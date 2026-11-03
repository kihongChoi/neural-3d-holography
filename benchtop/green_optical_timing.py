"""Measure optical frame identities on a moving input with a camera-read margin tag."""
import threading
import time

import cv2
import numpy as np

from .hardware import encode_phase
from .measure_slm_plane import write
from .multiplane_green import kernel,t
from .paper_model import dpac
from .targets import srgb_to_linear

def timed_replay(bench,lut,prior,video,out,settle_s=.025,*,rgb_quality=False,correct=True):
    bench.set_timing(100.,6022.)
    cap=cv2.VideoCapture(str(video))
    inputs=[]
    try:
        for _ in range(48):
            ok,im=cap.read()
            if not ok: raise ValueError("Video too short")
            small=cv2.resize(im,(256,256),interpolation=cv2.INTER_AREA)
            inputs.append(small[...,::-1].copy() if rgb_quality else small[...,1].copy())
    finally:cap.release()
    asm=kernel(6.25)
    if rgb_quality:
        from .green_lut_output import nominal_codes, ASM, C
        kernels=[ASM(dict(C,slm_shape=[512,512],distance_m=.00625,wavelength_m=w),"cuda")
                 for w in (638e-9,520e-9,450e-9)]
    def generate(im):
        amplitude=np.sqrt(srgb_to_linear(im.astype("f")/255))
        if rgb_quality:
            patch=nominal_codes(amplitude,kernels,32)
            if correct:
                patch[...,1]=lut.apply(np.ascontiguousarray(patch[...,1]))
            return patch
        patch=encode_phase(dpac(asm,t(amplitude)).cpu().numpy())
        return lut.apply(patch) if correct else patch
    # Keep the readout cells outside the content patch so their optical
    # neighborhood is identical during calibration and video playback.
    centers=np.float32([(765+64*i,244) for i in range(7)])
    sensor=cv2.perspectiveTransform(centers[None],prior["H"])[0]*2
    camera=bench.camera
    old=[n.GetValue() for n in (camera.Width,camera.Height,camera.OffsetX,camera.OffsetY)]
    def set_roi(width,height,x,y):
        camera.OffsetX.SetValue(0);camera.OffsetY.SetValue(0)
        camera.Width.SetValue(width);camera.Height.SetValue(height)
        camera.OffsetX.SetValue(x);camera.OffsetY.SetValue(y)
    ox=int(sensor[:,0].mean()-512)//camera.OffsetX.GetInc()*camera.OffsetX.GetInc()
    oy=int(sensor[:,1].mean()-96)//camera.OffsetY.GetInc()*camera.OffsetY.GetInc()
    roi_xy=np.rint(sensor-[ox,oy]).astype(int)
    if not ((roi_xy>[12,12]).all() and (roi_xy<[1011,179]).all()):raise RuntimeError("Timing tag outside crop")
    checker=np.uint8(64+128*(np.indices((44,44)).sum(0)%2))
    def frame(idx,patch):
        full=np.full((1080,1920)+patch.shape[2:],128,np.uint8)
        full[284:796,704:1216]=patch
        gray=idx^(idx>>1)
        tag=full[...,1] if full.ndim==3 else full
        for bit,(x,y) in enumerate(centers.astype(int)):
            tag[y-22:y+22,x-22:x+22]=checker if gray&(1<<bit) else 128
        return full
    # Use the same 10x10 Bayer cells as bench.capture() uses for bright/dark.
    # An odd sum of the raw x/y origins would sample R/B instead of G.
    region=[np.s_[2*(y//2-5):2*(y//2+5),2*(x//2-5):2*(x//2+5)] for x,y in roi_xy]
    def sample(raw):
        return [float((raw[r][::2,1::2].mean()+raw[r][1::2,::2].mean())/2) for r in region]
    blank=np.full((512,512),128,np.uint8)
    prev_settle=bench.config["settle_s"]
    try:
        set_roi(1024,192,ox,oy)
        # All-zero and all-one Gray codes are produced by IDs 0 and 85.
        refs=[]
        for idx in (0,85):
            bench.show_codes(frame(idx,blank))
            raw=bench.capture(frames=3)[...,1]
            refs.append(np.array([raw[y//2-5:y//2+5,x//2-5:x//2+5].mean()*65535 for x,y in roi_xy]))
        bright,dark=refs
        if np.min((bright-dark)/np.maximum(bright,1))<.15: raise RuntimeError("Unresolved timing tag")
        records=[];errors=[];stop=threading.Event()
        def acquire():
            while not stop.is_set():
                try:
                    im=camera.GetNextImage(1500)
                    try:
                        if im.IsIncomplete(): raise RuntimeError("Incomplete timing frame")
                        records.append((int(im.GetTimeStamp()),int(im.GetFrameID()),sample(im.GetNDArray())))
                    finally:im.Release()
                except Exception as exc:
                    errors.append(str(exc));break
        bench.config["settle_s"]=settle_s
        for im in inputs[:8]:generate(im)
        camera.BeginAcquisition();bench.acquiring=True
        thread=threading.Thread(target=acquire)
        thread.start()
        starts=[]
        try:
            for idx,im in enumerate(inputs,1):
                starts.append(time.perf_counter_ns())
                bench.show_codes(frame(idx,generate(im)))
            time.sleep(.35)
        finally:
            stop.set();thread.join()
            camera.EndAcquisition();bench.acquiring=False
        if errors:raise RuntimeError(str(errors))
        decoded=[]
        for stamp,fid,vals in records:
            bits=(bright-vals)/(bright-dark)
            valid=bool(((bits<.3)|(bits>.7)).all())
            gray=sum(int(b>.5)<<j for j,b in enumerate(bits))
            idx=gray
            while gray:
                gray>>=1;idx^=gray
            decoded.append(dict(timestamp_ns=stamp,camera_id=fid,tag=idx,confident=valid))
        seen={}
        for row in decoded:
            if row["confident"] and 1<=row["tag"]<=len(inputs):
                seen.setdefault(row["tag"],row["timestamp_ns"])
        write(out/"optical_tag_trace.json",dict(starts_ns=starts,decoded=decoded,raw=records,
              bright=bright.tolist(),dark=dark.tolist(),xy=centers.tolist(),settle_s=settle_s))
        if 9 not in seen or 48 not in seen: raise RuntimeError("Optical endpoints missing")
        missing=sorted(set(range(9,49))-seen.keys())
        duration=(seen[48]-seen[9])/1e9
        order=[seen[i] for i in range(9,49) if i in seen]
        if len(order)<2 or not all(b>a for a,b in zip(order,order[1:])):
            raise RuntimeError("Optical frame IDs are not in presentation order")
        intervals_ms=np.diff(order)/1e6
        write(out/"optical_timing.json",dict(physical_update_hz=(39-len(missing))/duration,
             source="48 sequential 256x256 inputs, first 8 excluded; 7-bit Gray-code in margin and HDMI verification",
             pipeline="RGB DPAC with 32 green ASM iterations" if rgb_quality else "green single-pass DPAC",lut_on=correct,
             settle_s=settle_s,camera_sample_hz=float(1e9/np.median(np.diff([r[0] for r in records]))),
             optical_interval_mean_ms=float(intervals_ms.mean()),optical_interval_p95_ms=float(np.percentile(intervals_ms,95)),
             tag_xy_slm=centers.tolist(),sensor_roi=[ox,oy,1024,192],optically_unseen_ids=missing,
             unique_postwarm=40-len(missing),camera_id_gaps=sum(max(0,b[1]-a[1]-1) for a,b in zip(records,records[1:])),
             starts_ns=starts,decoded=decoded))
    finally:
        bench.config["settle_s"]=prev_settle
        set_roi(*old)
