"""Generate the row-major 25-plane RGB phase frame and scan it without changing the SLM."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from benchtop.hardware import Bench, encode_phase
from benchtop.optics import ASM, crop_center
from benchtop.stage import Stage, validate_target
from benchtop.readybeam import ReadyBeam, exclusive

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results'/'step01_canvas80'
HEIGHT, WIDTH = 960, 1728
ROI = np.s_[60:60+HEIGHT,96:96+WIDTH]
OUTSIDE = np.ones((1080,1920),dtype=bool)
OUTSIDE[ROI] = False
C = json.loads((ROOT/'configs/benchtop.json').read_text())
Z = np.linspace(0,10,25)
X_EDGES = np.rint(np.linspace(0,WIDTH,6)).astype(int)
Y_EDGES = np.rint(np.linspace(0,HEIGHT,6)).astype(int)
SITES = [(round((col+.5)*WIDTH/5-.5),round((row+.5)*HEIGHT/5-.5))
         for row in range(5) for col in range(5)]
ZERO = 100.780
SCAN_RANGE = (90.780,100.780)
WAVELENGTHS = (638e-9,520e-9,450e-9)

def png(path, image):
    if path.exists():raise FileExistsError(path)
    if not cv2.imwrite(str(path),image,[cv2.IMWRITE_PNG_COMPRESSION,1]):raise IOError(path)

def design():
    OUT.mkdir(exist_ok=False)
    y,x=np.indices((HEIGHT,WIDTH),dtype='f')
    target=np.zeros_like(x)
    for cx,cy in SITES:target[(x-cx)**2+(y-cy)**2<=100]=1
    assert cv2.connectedComponents(target.astype('uint8'))[0]==26
    png(OUT/'target.png',np.uint8(255*target))
    checker=np.where((x+y)%2==0,1.,-1.).astype('f')
    target_gpu=torch.as_tensor(target,device='cuda')
    checker_gpu=torch.as_tensor(checker,device='cuda')
    # All image, phase, and propagation coordinates below are on the native ROI canvas.
    xs,ys=X_EDGES,Y_EDGES
    cells=[]; weights=[]
    for j,(cx,cy) in enumerate(SITES):
        row,col=divmod(j,5)
        sl=np.s_[ys[row]:ys[row+1],xs[col]:xs[col+1]]
        cells.append(sl)
        weight=.008+np.exp(-((x[sl]-cx)**2+(y[sl]-cy)**2)/(2*16**2))
        weights.append(torch.as_tensor(weight/weight.sum(),device='cuda'))
    frames=[]
    for channel,wl in zip('RGB',WAVELENGTHS):
        config=dict(C,wavelength_m=wl,slm_shape=[HEIGHT,WIDTH])
        kernels=[]
        field=torch.zeros((HEIGHT,WIDTH),device='cuda',dtype=torch.complex64)
        # Sum the 25 back-propagated target fields across the entire native canvas.
        for j,depth in enumerate(Z):
            k=ASM(dict(config,distance_m=depth*.001),'cuda')
            kernels.append(k)
            one_dot=torch.zeros_like(target_gpu)
            one_dot[cells[j]]=target_gpu[cells[j]]
            spectrum=torch.fft.fft2(F.pad(one_dot,k.padding))
            field+=crop_center(torch.fft.ifft2(spectrum*k.kernel.conj()),(HEIGHT,WIDTH))
        angle=torch.where(field.abs()>.003,field.angle(),0.)
        phase=(angle+checker_gpu*torch.acos(field.abs().clamp(0,1))).detach().requires_grad_()
        optimizer=torch.optim.SGD([phase],lr=8.,momentum=.8)
        rng=np.random.default_rng(53)
        def loss(j):
            amp=kernels[j](phase).abs()[cells[j]]
            return (weights[j]*(amp-target_gpu[cells[j]]).square()).sum()
        with torch.no_grad(): before=float(torch.stack([loss(j) for j in range(25)]).mean())
        for epoch in range(12):
            for j in rng.permutation(25):
                optimizer.zero_grad(set_to_none=True);loss(j).backward();optimizer.step()
        with torch.no_grad(): after=float(torch.stack([loss(j) for j in range(25)]).mean())
        if not 0<=after<before:raise RuntimeError(f'{channel} SGD loss {before} -> {after}')
        print(f'{channel}: 300 SGD updates, mean design loss {before:.7f} -> {after:.7f}',flush=True)
        frames.append(encode_phase(phase.detach().cpu().numpy()))
        del kernels,optimizer,phase,field
    drive=np.stack(frames,axis=-1)
    png(OUT/'phase.png',cv2.cvtColor(drive,cv2.COLOR_RGB2BGR))
    assert np.array_equal(cv2.imread(str(OUT/'phase.png'))[...,::-1],drive)

def full_capture(b):
    b.verify_display();cam=b.camera
    cam.BeginAcquisition();b.acquiring=True
    try:
        for i in range(2):
            frame=cam.GetNextImage(4000)
            try:
                if frame.IsIncomplete():raise RuntimeError('Incomplete camera frame')
                if i:native=frame.GetNDArray().copy()
            finally:frame.Release()
    finally:
        try:cam.EndAcquisition()
        finally:b.acquiring=False;del cam
    b.verify_display()
    return native

def present_phase(b, path):
    from PySide6.QtGui import QImage
    bgr = cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if bgr is None or bgr.shape != (HEIGHT,WIDTH,3) or bgr.dtype != np.uint8:
        raise ValueError(f'Expected a native RGB phase canvas: {path}')
    drive = np.full((1080,1920,3),128,np.uint8)
    drive[ROI] = bgr[...,::-1]
    b.show_codes(drive)
    screen = b.label.screen().grabWindow(0).toImage().convertToFormat(QImage.Format_RGB888)
    actual = np.frombuffer(screen.constBits(),dtype='uint8').reshape(1080,screen.bytesPerLine())[:,:1920*3].reshape(drive.shape)
    if not np.array_equal(actual,drive):
        raise RuntimeError('SLM framebuffer pixel check failed')


def capture(output=OUT, depths=Z, phase_paths=None, *, register_on_move=False):
    frames=output/'camera_png';frames.mkdir(exist_ok=False)
    if phase_paths is None:
        phase_paths = [output/'phase.png']*len(depths)
    if len(phase_paths)!=len(depths) or not len(depths):
        raise ValueError('Each capture depth needs a phase frame')
    if register_on_move:
        from .geometry_registration import cached_patterns,move_and_align
        banks={z:cached_patterns(z,ROOT/'results'/'step03_geometry'/'pattern_cache') for z in set(depths)}
        registrations=[]
    vals=np.arange(65536,dtype='d')/65535
    lut=np.rint(255*np.where(vals<=.0031308,12.92*vals,1.055*vals**(1/2.4)-.055)).astype('uint8')
    with exclusive():
        laser=ReadyBeam();laser.configure('Digital')
        with Stage(serial='45897069',interval=SCAN_RANGE) as s:
            start=s.initial['position_mm']
            if abs(start-ZERO)>.002 or s.real(s.limits()[2])!=124 or s.dll.ISC_GetSoftLimitMode(s.serial)!=0:raise RuntimeError('Stage preflight failed')
            for z in depths:validate_target(round(ZERO-z,6),SCAN_RANGE,s.backlash_mm)
            try:
                with Bench(dict(C,exposure_us=15000.,gain_db=0.,trigger_delay_us=22.,settle_s=.22 if register_on_move else .45)) as b:
                    present_phase(b,phase_paths[0])
                    timing=b.set_timing(15000.,22.)
                    if abs(timing['exposure_us']-15000)>10:raise RuntimeError(f'Exposure readback: {timing}')
                    if (b.camera.Width.GetValue(),b.camera.Height.GetValue(),b.camera.OffsetX.GetValue(),b.camera.OffsetY.GetValue())!=(4096,3000,0,0):raise RuntimeError('Expected full frame')
                    print('Camera timing',timing,flush=True)
                    try:
                        print('Laser mode',laser.configure('Analog')['mode'],flush=True)
                        for i,z in enumerate(depths):
                            dest=round(ZERO-z,6)
                            if i and phase_paths[i]!=phase_paths[i-1]:present_phase(b,phase_paths[i])
                            if register_on_move and (not i or z!=depths[i-1]):
                                registration=move_and_align(s,b,z,banks[z])
                                registrations.append(registration)
                                temp=output/'geometry.tmp'
                                temp.write_text(json.dumps(registrations,indent=2),encoding='utf8')
                                temp.replace(output/'geometry.json')
                                if not registration['accepted']:
                                    raise RuntimeError(f'Geometry failed independent validation at {z} mm')
                            elif abs(s.status()['position_mm']-dest)>.002:s.move(dest)
                            state=s.status()
                            if state['moving'] or abs(state['position_mm']-dest)>.002:raise RuntimeError(f'Position not settled {state}')
                            native=full_capture(b)
                            assert native.shape==(3000,4096) and native.dtype==np.uint16
                            frac=float((native>=.98*65535).mean())
                            out=lut[cv2.cvtColor(native,cv2.COLOR_BayerRGGB2BGR)]
                            path=frames/f'{i+1:02d}_z{z:06.3f}mm.png'
                            png(path,out)
                            assert np.array_equal(cv2.imread(str(path)),out)
                            print(f'{path.name}  stage={state["position_mm"]:.4f} mm  near-saturation={frac:.6f}',flush=True)
                    finally:print('Standby',laser.configure('Digital'),flush=True)
            finally:
                if s.moves:s.require_stationary(s.status());s.move(start)
                print('End stage',s.status(),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('step',choices=['design','capture'])
    if ap.parse_args().step=='design':design()
    else:capture()
