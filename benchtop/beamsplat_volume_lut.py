"""Preserve BeamSplat complex fields in a fixed DPAC physical depth sweep."""
import argparse,json,hashlib,shutil
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from scipy.signal import resample
import cv2
from .joint_rgb_lut_ab import PhysicalAB,validate_joint,write
from .dpac_lut_session import Session,sha,now
from .dpac_lut_fit import apply_lut
from .hardware import encode_phase
from .optics import ASM,crop_center
from . import step01_dot_grid as grid
from . import geometry_registration as geo
from .rgb_roi_calibration import stats

SHAPE=(960,1728)
ROI=(414,227,900,506)

def prop(field,z,ch):
    m=ASM(dict(grid.C,slm_shape=list(SHAPE),distance_m=z/1000,wavelength_m=grid.WAVELENGTHS[ch]),'cuda')
    u=torch.as_tensor(field,device='cuda')
    return crop_center(torch.fft.ifft2(torch.fft.fft2(F.pad(u,m.padding))*m.kernel),SHAPE).cpu().numpy()

def prepare(root,source,package):
    tables,meta,checks=validate_joint(package)
    depths=list(range(1,10));scenes=[]
    (root/'patterns').mkdir();(root/'numerical').mkdir()
    snap=root/'source_snapshot';snap.mkdir()
    for n in ['beamsplat_volume_lut.py','joint_rgb_lut_ab.py','hardware.py','optics.py','stage.py','readybeam.py','dpac_lut_session.py']:
        shutil.copy2(Path(__file__).parent/n,snap/n)
    for s in sorted(source.iterdir()):
        if not(s/'field/U.npy').is_file():continue
        summary=json.loads((s/'summary.json').read_text());u=np.load(s/'field/U.npy',allow_pickle=False)
        assert u.shape==(3,1080,1920) and np.iscomplexobj(u) and np.isfinite(u).all()
        assert summary['optics']['pixel_pitch_um']==3
        assert np.allclose(summary['optics']['wavelengths_nm'],[638,520,450])
        resized=resample(resample(u,506,axis=1),900,axis=2).astype(np.complex64)
        target=np.zeros((3,*SHAPE),np.complex64);x,y,w,h=ROI;target[:,y:y+h,x:x+w]=resized
        back=np.stack([prop(target[c],-5,c) for c in range(3)])
        peak=float(np.abs(back).max());back/=peak
        yy,xx=np.indices(SHAPE);sign=np.where((yy+xx)%2==0,1,-1)
        phase=np.angle(back)+sign*np.arccos(np.clip(np.abs(back),0,1))
        codes=np.moveaxis(encode_phase(phase),0,-1)
        p=root/'patterns'/f'{s.name}.png';grid.png(p,codes[...,::-1])
        assert np.array_equal(cv2.imread(str(p))[...,::-1],codes)
        scores=[]
        for z in depths:
            ideal=np.stack([abs(prop(back[c],z,c))**2 for c in range(3)],-1).astype(np.float32)
            np.save(root/'numerical'/f'{s.name}_z{z}.npy',ideal)
            scores.append(dict(depth_mm=z,mean=ideal.mean(axis=(0,1)).tolist()))
        record=dict(scene=s.name,source=str(s.resolve()),source_U_sha256=sha(s/'field/U.npy'),
                    summary_sha256=sha(s/'summary.json'),summary=summary,pattern=str(p.relative_to(root)),
                    png_sha256=sha(p),rgb_sha256=hashlib.sha256(codes.tobytes()).hexdigest(),
                    common_RGB_backfield_peak=peak,resampled_mean_power_ratio=(np.mean(abs(resized)**2,axis=(1,2))/np.mean(abs(u)**2,axis=(1,2))).tolist())
        scenes.append(record);print('Prepared complex DPAC',s.name,flush=True)
    assert len(scenes)==3
    plan=dict(created_kst=now(),scenes=scenes,depths_mm=depths,methods={'A':'identity','B':'joint RGB LUT'},
              expected_scene_captures=108,repeats=2,modes=['RGB'],timing=[15000,22],
              package=str(package.resolve()),package_checksums=checks,teacher_metadata=meta,
              reference_depth_mm=5,content_roi_xywh=ROI,
              mapping='Complex Fourier resample 3um to6.4um,900x506 centered; vertical physical extent rounded by0.0494%; common RGB peak; negative5mm ASM then single checkerboard DPAC. Same fixed codes at all observation depths.',
              target='Ideal complex backfield propagated numerically through same ASM aperture, not camera or original pinhole ground truth',
              distinction='Three exports of same scene/view with different world_scale, not three independent scenes')
    write(root/'ab_plan.json',plan);write(root/'prepared.json',dict(plan_sha256=sha(root/'ab_plan.json')))
    shutil.copy2(package/'rgb_lut_candidate.npz',root/'rgb_lut_candidate.npz')

class Volume(PhysicalAB):
    def __init__(self,root):
        Session.__init__(self,root,'ab');self.plan=json.loads((root/'ab_plan.json').read_text())
        assert sha(root/'ab_plan.json')==json.loads((root/'prepared.json').read_text())['plan_sha256']
        self.tables,meta,_=validate_joint(Path(self.plan['package']));assert meta==self.plan['teacher_metadata']
        self.modes=dict(RGB=(15000,22));self.report.update(version='beamsplat-complex-volume-ab-v1',comparisons=[],requested_modes=self.modes)
        self.save()

    def depth(self,z,cal):
        self.report['state']=f'volume_scan_{z:g}mm';self.save();self.laser.configure('Analog')
        try:
            for i,s in enumerate(self.plan['scenes']):
                p=self.root/s['pattern'];assert sha(p)==s['png_sha256']
                codes=cv2.imread(str(p))[...,::-1].copy()
                assert hashlib.sha256(codes.tobytes()).hexdigest()==s['rgb_sha256']
                for repeat in range(2):
                    order=['A','B'] if (i+repeat)%2==0 else ['B','A'];pair={}
                    for method in order:
                        name=f'z{z:g}_{s["scene"]}_r{repeat}_RGB_{method}'
                        drive=apply_lut(codes,self.tables[method]);self.b.set_timing(15000,22);self.b.show_codes(geo.pack(drive))
                        raw=self.capture(name);q=stats(raw,cal['RGB']['mask'])
                        if any(q[c]['near_clip']>.0001 for c in 'RGB'):raise RuntimeError(f'Saturation {name}')
                        rec=dict(name=name,scene=s['scene'],depth_mm=z,repeat=repeat,mode='RGB',method=method,
                                 base_rgb_sha256=s['rgb_sha256'],drive_sha256=hashlib.sha256(drive.tobytes()).hexdigest(),roi=q)
                        self.report['records'][-1].update(rec);pair[method]=rec;self.save()
                    self.report['comparisons'].append(dict(scene=s['scene'],depth_mm=z,repeat=repeat,mode='RGB',order=order,pair=pair));self.save()
                print('Captured',s['scene'],'at',z,'mm',flush=True)
            self.drift(z,cal,'end')
        finally:self.laser.configure('Digital')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','acquire']);p.add_argument('root',type=Path);p.add_argument('--source',type=Path);p.add_argument('--package',type=Path);a=p.parse_args()
    if a.action=='prepare':prepare(a.root,a.source,a.package)
    else:Volume(a.root).execute()
