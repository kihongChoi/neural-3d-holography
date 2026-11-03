"""Fit measured mask coordinates, preserving native Bayer pixel units."""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def normalized(root, name):
    flat = (np.load(root/'flat_0_green_half.npy') + np.load(root/'flat_end_green_half.npy'))/2
    background = np.load(root/'rejected_green_half.npy')
    den = cv2.GaussianBlur(flat-background, (0,0), 1.5)
    delta = cv2.GaussianBlur(np.load(root/f'{name}_green_half.npy')-background, (0,0), 1.5)
    return np.clip(delta/np.maximum(den, .015), 0, 1), den>.025


def edges(profile, low, high, threshold):
    # Only use crossings surrounding the identified mask, excluding other islands.
    def crossing(guess, rising):
        indices = np.arange(max(0,int(guess)-12), min(len(profile)-1,int(guess)+13))
        candidates = indices[((profile[indices]<=threshold)&(profile[indices+1]>threshold)) if rising else
                             ((profile[indices]>=threshold)&(profile[indices+1]<threshold))]
        if not len(candidates):
            raise RuntimeError('No reliable mask edge crossing')
        i = candidates[np.argmin(abs(candidates-guess))]
        return float(i+(threshold-profile[i])/(profile[i+1]-profile[i]))
    return crossing(low, True), crossing(high, False)


def measure_mask(root, name, threshold=.5):
    z, valid = normalized(root, name)
    n, labels, stats, centers = cv2.connectedComponentsWithStats(np.uint8((z>.5)&valid))
    if n<2:
        raise RuntimeError('No mask component')
    i=1+np.argmax(stats[1:,4])
    x,y,w,h,area=stats[i]
    if area<1000:
        raise RuntimeError('Mask component too small')
    # Medians through the middle remove dust and avoid rounded mask corners.
    xp=np.median(z[y+h//4:y+3*h//4+1],axis=0)
    yp=np.median(z[:,x+w//4:x+3*w//4+1],axis=1)
    left,right=edges(xp,x,x+w-1,threshold)
    top,bottom=edges(yp,y,y+h-1,threshold)
    return dict(center_native_xy=[left+right+.5,top+bottom+.5],
                size_native_xy=[2*(right-left),2*(bottom-top)],
                bounds_native_xy=[2*left+.5,2*top+.5,2*right+.5,2*bottom+.5])


def analyze(root, threshold=.5):
    acquisition=json.loads((root/'acquisition.json').read_text())
    measurements=[]
    for row in acquisition['records']:
        if row['name'].startswith('patch_'):
            measurements.append(dict(name=row['name'],slm_xy=row['slm_center_xy'],
                                     **measure_mask(root,row['name'],threshold)))
    source=np.array([r['slm_xy'] for r in measurements])
    dest=np.array([r['center_native_xy'] for r in measurements])
    design=np.c_[source,np.ones(len(source))]
    affine=np.linalg.lstsq(design,dest,rcond=None)[0]
    residual=dest-design@affine
    pitch_ratio=acquisition['config']['camera_pixel_m']/acquisition['config']['slm_pitch_m']
    scale=np.linalg.norm(affine[:2],axis=1)
    inset=measure_mask(root,'inset_rectangle',threshold)
    drift=float(np.load(root/'flat_end_green_half.npy').mean()/np.load(root/'flat_0_green_half.npy').mean()-1)
    return dict(source=str(root.resolve()),threshold=threshold,measurements=measurements,
                affine_slm_to_native_sensor=affine.T.tolist(),
                sensor_pixels_per_slm_pixel_xy=scale.tolist(),
                physical_magnification_xy=(scale*pitch_ratio).tolist(),
                rms_native_sensor_px=float(np.sqrt(np.mean(np.sum(residual**2,axis=1)))),
                max_residual_native_sensor_px=float(np.linalg.norm(residual,axis=1).max()),
                inferred_full_slm_size_sensor_px=(scale*[1920,1080]).tolist(),
                inferred_full_slm_size_mm=(scale*[1920,1080]*acquisition['config']['camera_pixel_m']*1000).tolist(),
                independent_inset=inset,
                independent_inset_magnification_xy=(np.array(inset['size_native_xy'])/[1760,920]*pitch_ratio).tolist(),
                local_horizontal_magnification=[float(np.linalg.norm(dest[3*j+2]-dest[3*j])/1200*pitch_ratio) for j in range(3)],
                local_vertical_magnification=[float(np.linalg.norm(dest[j+6]-dest[j])/600*pitch_ratio) for j in range(3)],
                parity_determinant=float(np.linalg.det(affine[:2])),flat_mean_fractional_change=drift)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('runs',nargs='+')
    args=parser.parse_args()
    roots=[Path(r) for r in args.runs]
    reports=[analyze(p) for p in roots]
    sensitivity=[analyze(p,t)['physical_magnification_xy'] for p in roots for t in [.35,.65]]
    avg=np.mean([r['physical_magnification_xy'] for r in reports],axis=0)
    total=dict(runs=reports,mean_magnification_xy=avg.tolist(),
               edge_threshold_sensitivity_magnification_xy=sensitivity,
               nominal_slm_pitch_um=6.4,nominal_sensor_pitch_um=3.45,
               limitations='Geometric estimate at the current sensor plane using nominal manufacturer pixel pitches. Edge threshold sensitivity and repeatability are diagnostics, not a certified uncertainty or an axial focus measurement. No old calibration, distance, or magnification was used in fitting; no experiment config was updated.')
    out=roots[0]
    (out/'measurement.json').write_text(json.dumps(total,indent=2))
    r=reports[0]
    affine=np.array(r['affine_slm_to_native_sensor'])
    flat=np.load(out/'flat_0_green_half.npy')
    gray=np.clip(flat/max(float(np.quantile(flat,.999)),.001),0,1)**(1/2.2)
    fig=plt.figure(figsize=(13,8),facecolor='#f4f6fa')
    ax=fig.add_axes([.06,.12,.62,.79])
    ax.imshow(gray,cmap='gray',vmin=0,vmax=1,extent=[-.5,4095.5,2999.5,-.5])
    corners=np.array([[-.5,-.5],[1919.5,-.5],[1919.5,1079.5],[-.5,1079.5],[-.5,-.5]])
    mapped=np.c_[corners,np.ones(len(corners))]@affine.T
    ax.plot(mapped[:,0],mapped[:,1],color='#ffba42',lw=1.6,label='SLM active area inferred from 9 masks')
    for i,m in enumerate(r['measurements']):
        x,y=m['center_native_xy']
        ax.scatter(x,y,s=30,facecolor='#32e3d0',edgecolor='black',lw=.5)
        ax.annotate(str(i+1),(x,y),xytext=(8,-9),textcoords='offset points',color='#32e3d0',fontsize=10,weight='bold')
    ax.set(xlim=(0,4096),ylim=(3000,0),xlabel='Native sensor x [pixels]',ylabel='Native sensor y [pixels]')
    ax.set_title('Actual full-screen flat capture + measured mask centers',loc='left',fontsize=13,pad=15)
    ax.legend(loc='lower right',fontsize=8,framealpha=.85)
    panel=fig.add_axes([.72,.12,.26,.79]);panel.axis('off')
    width,height=np.mean([q['inferred_full_slm_size_sensor_px'] for q in reports],axis=0)
    text=(f'RELAY MAGNIFICATION\n\n'
          f'Horizontal   {avg[0]:.5f} x\nVertical       {avg[1]:.5f} x\n\n'
          f'Consistent with 1 : 1\n\n'
          f'SLM: 1920 x 1080\nPitch: 6.4 um\nSize: 12.288 x 6.912 mm\n\n'
          f'Sensor: 4096 x 3000\nPitch: 3.45 um\nBinning / decimation: 1\n\n'
          f'Inferred SLM image size:\n{width:.1f} x {height:.1f} sensor px\n\n'
          f'9 masks x {len(reports)} captures\nEach mask: 128 x 128 SLM px\n3-frame averages; green slot\nAffine residual: {r["rms_native_sensor_px"]:.2f} sensor px\n\n'
          f'Horizontal reflection observed\nNominal pixel pitches assumed')
    panel.text(0,1,text,va='top',fontsize=11,linespacing=1.45,color='#15243a')
    fig.suptitle('4f relay size measurement | 2026-10-30',x=.06,ha='left',fontsize=17,weight='bold',color='#15243a')
    fig.savefig(out/'relay_measurement.png',dpi=160,facecolor=fig.get_facecolor())
    print(json.dumps(total,indent=2))


if __name__=='__main__':
    main()
