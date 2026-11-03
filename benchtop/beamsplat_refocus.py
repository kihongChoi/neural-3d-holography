"""Numerical focus sweep of the ORIGINAL complex field, on its 3um grid."""
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .optics import crop_center
from .rgb_run import srgb_preview
from .paper_campaign import sha

source=Path(r'C:\Users\khchoi_exp\Documents\GitHub\beamSplat_2026\samples\20261026_134702')
out=Path('results/beamsplat_refocus_20261026')
out.mkdir(exist_ok=False)
u=np.load(source/'field/U.npy')
meta=json.loads((source/'summary.json').read_text())
h,w=u.shape[-2:]
pitch=meta['optics']['pixel_pitch_um']*1e-6
peak=float(abs(u).max())
rois={'left_plant':[30,40,550,620], 'rear_right':[1320,80,1880,530], 'table':[650,410,1230,650]}
spectra=[torch.fft.fft2(F.pad(torch.from_numpy(ch).cuda(),(w//2,w-w//2,h//2,h-h//2))) for ch in u]
fy=np.fft.fftfreq(2*h,pitch)[:,None]
fx=np.fft.fftfreq(2*w,pitch)[None,:]
r2=fx*fx+fy*fy
waves=meta['optics']['wavelengths_nm']
phase_per_m=[]
for wave in waves:
    wavelength=wave*1e-9
    root=np.sqrt(np.maximum(1/wavelength**2-r2,0))
    phase_per_m.append(torch.from_numpy((-2*np.pi*r2/(root+1/wavelength)).astype(np.float32)).cuda())

def propagate(dz_mm):
    fields=[]
    for spec,k in zip(spectra,phase_per_m):
        field=crop_center(torch.fft.ifft2(spec*torch.exp(1j*k*(dz_mm*1e-3))),(h,w))
        fields.append(field.cpu().numpy())
    return np.stack(fields)

rows=[]
grid=[-3.3,-2.75,-2.2,-1.65,-1.1,-.55,0.,.55,1.1,1.65,2.2,2.75,3.3,4.4,5.5,6.6]
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22)
sheet=Image.new('RGB',(1920,4*310),'#111722')
draw=ImageDraw.Draw(sheet)
for i,dz in enumerate(grid):
    field=propagate(dz)
    amp=np.moveaxis(abs(field),0,-1)/peak
    rgb=srgb_preview(amp)
    Image.fromarray(np.uint8(np.clip(rgb*255,0,255))).save(out/f'focus_{dz:+.2f}mm.png')
    gray=(amp**2*np.array([.2126,.7152,.0722])).sum(-1)
    smooth=cv2.GaussianBlur(gray.astype(np.float32),(0,0),1.2)
    gx=cv2.Sobel(smooth,cv2.CV_32F,1,0,ksize=3)
    gy=cv2.Sobel(smooth,cv2.CV_32F,0,1,ksize=3)
    scores={}
    for name,(x0,y0,x1,y1) in rois.items():
        g=smooth[y0:y1,x0:x1]
        scores[name]=float(np.mean((gx*gx+gy*gy)[y0:y1,x0:x1])/(g.mean()**2+1e-10))
    effective=meta['optics']['sensor_distance_mm']+dz
    focal=meta['optics']['focal_length_mm']
    obj_mm=1/(1/focal-1/effective)
    row=dict(dz_mm=dz,effective_sensor_distance_mm=effective,thin_lens_object_distance_mm=obj_mm,
             approximate_depth_ply=obj_mm/(meta['optics']['world_scale']*1000),focus_scores=scores)
    rows.append(row)
    im=Image.fromarray(np.uint8(np.clip(rgb*255,0,255)))
    im.thumbnail((474,270))
    x,y=(i%4)*480,(i//4)*310
    sheet.paste(im,(x,y+34))
    draw.text((x+5,y+4),f'dz {dz:+.2f} mm | d {row["approximate_depth_ply"]:.2f}',fill='white',font=font)
    print(json.dumps(row),flush=True)
sheet.save(out/'focus_sweep.png')
(out/'sweep.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source/'field/U.npy'),
    policy='Zero-padded angular spectrum propagation of original RGB complex field at source 3um pitch. Common original peak normalization. Thin-lens depths are approximate labels, not fresh scene rendering.',rois=rois,rows=rows),indent=2))
np.save(out/'U_original.npy',u)
