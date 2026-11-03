"""Compare the unmodified camera PNGs with their center-cropped targets using held-out image landmarks."""
from pathlib import Path
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from .step02_div2k import OUT, PAIRS, stem
from .step01_dot_grid import WIDTH, HEIGHT

PTS=np.float32([(x,y) for y in np.linspace(90,HEIGHT-91,7) for x in np.linspace(90,WIDTH-91,9)]).round()
TRAIN=np.arange(len(PTS))%2==0
CORNERS=np.float32([[0,0],[WIDTH-1,0],[WIDTH-1,HEIGHT-1],[0,HEIGHT-1]])
def transform(h,p):
    return cv2.perspectiveTransform(np.float32(p)[None],h)[0]
def highpass(im):
    g=im.astype('f')
    return cv2.GaussianBlur(g,(0,0),1.2)-cv2.GaussianBlur(g,(0,0),6.)
def bootstrap(target,camera):
    detector=cv2.SIFT_create(nfeatures=6500)
    c=cv2.resize(camera,None,fx=.25,fy=.25,interpolation=cv2.INTER_AREA)
    kc,dc=detector.detectAndCompute(c,None)
    hypotheses=[]
    for flipped in (False,True):
        t=cv2.flip(target,1) if flipped else target
        t=cv2.resize(t,None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA)
        kt,dt=detector.detectAndCompute(t,None)
        pairs=[m for m,n in cv2.BFMatcher().knnMatch(dt,dc,k=2) if m.distance<.77*n.distance]
        if len(pairs)<4:continue
        a=np.float32([kt[m.queryIdx].pt for m in pairs])*2;b=np.float32([kc[m.trainIdx].pt for m in pairs])*4
        h,keep=cv2.findHomography(a,b,cv2.RANSAC,4.)
        if h is not None:hypotheses.append((int(keep.sum()),flipped,h))
    hypotheses.sort(key=lambda q:q[0],reverse=True)
    if not hypotheses or hypotheses[0][0]<9:raise RuntimeError('Insufficient registration features at reference plane')
    return hypotheses[0],[(count,flip) for count,flip,_ in hypotheses]

def landmarks(t,c,h):
    a=highpass(t)
    registered=cv2.warpPerspective(c,h,(WIDTH,HEIGHT),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
    b=highpass(registered)
    dst=np.full(PTS.shape,np.nan);conf=[]
    for i,(x,y) in enumerate(PTS.astype(int)):
        ref=a[y-36:y+37,x-36:x+37];area=b[y-70:y+71,x-70:x+71]
        ncc=cv2.matchTemplate(area,ref,cv2.TM_CCOEFF_NORMED)
        iy,ix=np.unravel_index(ncc.argmax(),ncc.shape);peak=float(ncc[iy,ix])
        rest=ncc.copy();rest[max(0,iy-7):iy+8,max(0,ix-7):ix+8]=-1
        conf.append(peak)
        if peak<.32 or peak-rest.max()<.04 or not (0<ix<68 and 0<iy<68) or ref.std()<1:continue
        def sub(v):
            l,m,r=v;return np.clip(.5*(l-r)/min(-1e-6,l-2*m+r),-.6,.6)
        dx=ix-34+sub(ncc[iy,ix-1:ix+2]);dy=iy-34+sub(ncc[iy-1:iy+2,ix])
        dst[i]=transform(h,[[x+dx,y+dy]])[0]
    return dst

def main():
    captures=sorted((OUT/'camera_png').glob('*.png'))
    assert len(captures)==len(PAIRS)
    images=[]
    for (depth,n),file in zip(PAIRS,captures):
        t=cv2.imread(str(OUT/'targets'/(stem(depth,n)+'.png')),0)
        c=cv2.imread(str(file),0)
        assert t.shape==(HEIGHT,WIDTH) and c.shape==(3000,4096)
        images.append((t,c))
    previous=OUT.parent/'step02_div2k_3planes'
    old_target=cv2.imread(str(previous/'targets'/'DIV2K_0802_z05mm.png'),0)
    old_camera=cv2.imread(str(previous/'camera_png'/'02_z05.000mm.png'),0)
    (n,flipped,h0),orient=bootstrap(old_target,old_camera)
    print('5 mm SIFT orientation, inliers',orient,flush=True)
    measurements=[]
    for t,c in images:
        if flipped:t=cv2.flip(t,1)
        dst=landmarks(t,c,h0);valid=np.isfinite(dst[:,0])
        fit=TRAIN&valid;held=~TRAIN&valid
        if fit.sum()<7 or held.sum()<4:raise RuntimeError(f'Sparse image registration: {fit.sum()} training, {held.sum()} independent')
        h,inliers=cv2.findHomography(PTS[fit],dst[fit],cv2.RANSAC,4.)
        if h is None or inliers.sum()<6:raise RuntimeError('Unstable geometry fit')
        measurements.append((h,t,c,dst,held,int(inliers.sum())))
    fig,axs=plt.subplots(3,3,figsize=(15.4,9.5),layout='constrained')
    info=[]
    for j,((d,n),(h,t,c,dst,held,nfit)) in enumerate(zip(PAIRS,measurements)):
        pred=transform(h,PTS[held]);delta=dst[held]-pred;err=np.linalg.norm(delta,axis=1)
        shared=np.linalg.norm(dst[held]-transform(measurements[1][0],PTS[held]),axis=1)
        quad=np.unique((PTS[held,0]>WIDTH/2)*1+(PTS[held,1]>HEIGHT/2)*2)
        corners=transform(h,CORNERS);rms=float(np.sqrt(np.mean(err**2)));p95=float(np.percentile(err,95))
        reg=cv2.warpPerspective(c,h,(WIDTH,HEIGHT),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
        eg=cv2.Canny(cv2.GaussianBlur(t,(0,0),1.6),38,80)>0
        # A light cyan contour overlays the actual camera intensity. The raw PNG is never altered.
        overlay=np.repeat(reg[...,None],3,axis=2);overlay[eg]=[0,225,245]
        a,b,e=axs[j]
        a.imshow(c[::4,::4],cmap='gray',vmin=0,vmax=255,extent=(0,4096,3000,0))
        poly=np.vstack([corners,corners[:1]]);a.plot(*poly.T,color='#4be399',lw=1.3)
        a.scatter(*dst[held].T,s=8,color='#ffd45b');a.axis('off');a.set_title(f'{d:g} mm | camera + target boundary')
        b.imshow(overlay[::2,::2]);b.axis('off');b.set_title(f'DIV2K {n:04d} | target contours in cyan')
        e.set(xlim=(0,4096),ylim=(3000,0),aspect='equal',xlabel='Sensor x (px)',ylabel='Sensor y (px)')
        e.scatter(*pred.T,color='#357493',s=6)
        e.quiver(pred[:,0],pred[:,1],delta[:,0]*30,delta[:,1]*30,angles='xy',scale_units='xy',scale=1,color='#ad442c',width=.005)
        e.set_title(f'Held-out residuals x30 | RMS {rms:.2f} px')
        span=np.ptp(corners,axis=0)/[4096,3000]
        item=dict(depth_mm=d,source=n,mirrored=flipped,train_inliers=nfit,holdout_n=int(held.sum()),holdout_quadrants=len(quad),rms_px=rms,p95_px=p95,shared_5mm_rms_px=float(np.sqrt(np.mean(shared**2))),coverage_xy=span.tolist(),corners=corners.tolist(),H_mirrored_target_to_full_sensor=h.tolist(),pass_3px=bool(len(quad)==4 and held.sum()>=10 and rms<=3 and p95<=6))
        info.append(item);print(item,flush=True)
    fig.suptitle('Target / camera alignment  |  green = projected canvas, gold = independent landmarks  |  full-resolution sensor pixels')
    fig.savefig(OUT/'alignment.png',dpi=150);plt.close(fig)
    (OUT/'alignment.json').write_text(json.dumps(dict(initialization='Independent previous 5 mm target/capture, then current training tiles only',orientation_inliers=orient,measurements=info),indent=2))
if __name__=='__main__':main()
