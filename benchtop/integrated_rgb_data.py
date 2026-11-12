"""Offline integrated-RGB cache. No hardware imports; test pixels remain sealed."""
import argparse, hashlib, io, json, os, time, traceback, multiprocessing
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path, PureWindowsPath
from functools import lru_cache
from datetime import datetime, timezone, timedelta
import cv2
import numpy as np

SHAPE=(960,1728)
HALF=np.array([[.5,0,-.25],[0,.5,-.25],[0,0,1.]])
def now(): return datetime.now(timezone(timedelta(hours=9))).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def array_sha(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def write_json(path,value):
    p=Path(path);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');tmp.replace(p)
def relative(root,name):
    p=PureWindowsPath(name)
    if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe relative path')
    result=Path(root).joinpath(*p.parts)
    if not result.resolve().is_relative_to(Path(root).resolve()):raise ValueError('Path escapes root')
    return result
def sensor_channel(raw,c):
    if c==1:return .5*(raw[::2,1::2].astype(np.float32)+raw[1::2,::2].astype(np.float32))
    return raw[c//2::2,c//2::2].astype(np.float32)
def rectify(raw,c,H):
    return cv2.warpPerspective(sensor_channel(raw,c),HALF@np.asarray(H),(SHAPE[1],SHAPE[0]),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
def read_raw(path,expected):
    payload=Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest()!=expected:raise ValueError('Raw SHA256 mismatch: '+str(path))
    with np.load(io.BytesIO(payload),allow_pickle=False) as f:
        raw=f['raw_bayer'].copy()
    if raw.shape!=(3000,4096) or raw.dtype!=np.uint16:raise ValueError('Unexpected Bayer frame')
    return raw
def validate_rows(rows):
    if len(rows)!=8200 or len({r['id'] for r in rows})!=8200:raise ValueError('Incomplete/duplicate bank')
    if Counter(r['split'] for r in rows)!={'fit':7500,'selection':300,'test':400}:raise ValueError('Split counts')
    parents={}
    for r in rows:
        if parents.setdefault(r['scene'],r['split'])!=r['split']:raise ValueError('Parent leakage')
        if r['split']!='test' and r['depth_mm'] not in (3.,7.,9.):raise ValueError('Held-out depth leakage')
        if r['calibration_session']!=r['raw_path'].replace('\\','/').split('/')[0]:raise ValueError('Calibration session mismatch')
        if r['timing']!={'exposure_us':15000.0,'trigger_delay_us':22.0}:raise ValueError('Mixed exposure')
        t=np.asarray(r['drive_lut'],dtype=np.int32)
        if t.shape!=(3,256) or np.any(t<0) or np.any(t>255) or np.any(np.diff(t,axis=1)<0):raise ValueError('Invalid LUT')
    if Counter(parents.values())!={'fit':742,'selection':25,'test':25}:raise ValueError('Parent counts')
    for split,n in [('fit',2500),('selection',100),('test',100)]:
        depths=(3.,5.,7.,9.) if split=='test' else (3.,7.,9.)
        if Counter(r['depth_mm'] for r in rows if r['split']==split)!={z:n for z in depths}:raise ValueError('Depth counts')

def make_references(dataset,out,rows,transfer_index):
    out.mkdir(exist_ok=True); refs={}
    for session,z in sorted({(r['calibration_session'],r['depth_mm']) for r in rows}):
        status_path=dataset/session/'status.json'
        assert sha(status_path)==transfer_index['dataset/'+session+'/status.json']['sha256']
        status=json.loads(status_path.read_text()); cal=[x for x in status['calibrations'] if x['mode']=='RGB' and x['depth_mm']==z]
        if len(cal)!=1 or not cal[0]['registration']['accepted'] or not status['routing']['RGB']['accepted']:raise ValueError('Unaccepted calibration')
        captures=[json.loads(s) for s in (dataset/session/'captures.jsonl').read_text().splitlines()]
        index={x['name']:x for x in captures}; darkrow=index[f'z{z:g}_RGB_dark'];whiterow=index[f'z{z:g}_RGB_white']
        dark=read_raw(relative(dataset,darkrow['raw_path']),darkrow['raw_sha256']);white=read_raw(relative(dataset,whiterow['raw_path']),whiterow['raw_sha256'])
        H=np.asarray(cal[0]['registration']['models']['pooled']); ds=[];ws=[];masks=[]
        bounds=cv2.warpPerspective(np.ones((1500,2048),np.float32),HALF@H,(1728,960),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)>.999
        for c in range(3):
            d=cv2.GaussianBlur(rectify(dark,c,H),(0,0),12)
            w=cv2.GaussianBlur(rectify(white,c,H),(0,0),12)-d
            valid=(w>max(32.,.1*float(np.median(w))))&bounds
            if valid.mean()<.98:raise ValueError('Calibration valid field below98%')
            valid[:16]=False;valid[-16:]=False;valid[:,:16]=False;valid[:,-16:]=False
            ds.append(d);ws.append(w);masks.append(valid)
        key=f'{session}_z{z:g}';file=out/(key+'.npz')
        np.savez(file,dark=np.stack(ds),white=np.stack(ws),valid=np.stack(masks),H=H)
        refs[key]=dict(file='references/'+file.name,sha256=sha(file),session=session,depth_mm=z,
                       valid_fraction=float(np.mean(masks)),dark_raw_sha256=darkrow['raw_sha256'],white_raw_sha256=whiterow['raw_sha256'],
                       normalization='max(rectified Bayer - sigma12 dark,0) / fixed sigma12 white-minus-dark; common16pixel boundary')
    write_json(out/'manifest.json',refs);return refs

_ROOT=None;_OUT=None
def init_worker(root,out):
    global _ROOT,_OUT
    _ROOT=Path(root);_OUT=Path(out);cv2.setNumThreads(1)
@lru_cache(maxsize=5)
def reference(key):
    with np.load(_OUT/'references'/f'{key}.npz') as f:return {k:f[k].copy() for k in f.files}
@lru_cache(maxsize=8)
def pattern(name,expected):
    p=relative(_ROOT,name);payload=p.read_bytes()
    if hashlib.sha256(payload).hexdigest()!=expected:raise ValueError('Pattern hash mismatch')
    bgr=cv2.imdecode(np.frombuffer(payload,np.uint8),cv2.IMREAD_COLOR)
    rgb=np.ascontiguousarray(bgr[...,::-1])
    if rgb.shape!=SHAPE+(3,) or rgb.dtype!=np.uint8:raise ValueError('Wrong pattern grid')
    return rgb
def preprocess_row(row):
    if row['split']=='test':raise ValueError('Test pixel decoding forbidden before model freeze')
    raw=read_raw(relative(_ROOT,row['raw_path']),row['raw_sha256']);base=pattern(row['pattern_path'],row['pattern_sha256'])
    if array_sha(base)!=row['base_rgb_sha256']:raise ValueError('Base RGB mismatch')
    table=np.asarray(row['drive_lut'],np.uint8)
    driven=np.stack([table[c,base[...,c]] for c in range(3)],-1)
    if array_sha(driven)!=row['displayed_rgb_sha256']:raise ValueError('Actual displayed codes mismatch')
    key=f"{row['calibration_session']}_z{row['depth_mm']:g}";ref=reference(key)
    measured=np.stack([np.maximum(rectify(raw,c,ref['H'])-ref['dark'][c],0)/np.maximum(ref['white'][c],1) for c in range(3)]).astype(np.float32)
    if not np.isfinite(measured).all():raise ValueError('Nonfinite measured intensity')
    name=f"{row['id']:05d}"; ip=_OUT/'intensity'/(name+'.npy');cp=_OUT/'codes'/(name+'.npy')
    np.save(ip,measured);np.save(cp,np.ascontiguousarray(driven.transpose(2,0,1)))
    meta={k:row[k] for k in ('id','scene','split','depth_mm','condition','augmentation','calibration_session','raw_sha256','pattern_sha256','source_sha256','displayed_rgb_sha256')}
    meta.update(intensity_path='intensity/'+ip.name,codes_path='codes/'+cp.name,reference_key=key,
                intensity_sha256=sha(ip),codes_sha256=sha(cp),intensity_mean=float(measured[ref['valid']].mean()))
    return meta
def preprocess(transfer,out,workers=4,limit=None):
    cv2.setNumThreads(1)
    transfer=Path(transfer).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();status=dict(state='preprocessing',started_kst=now(),processed=0,test_pixels_decoded=False)
    write_json(out/'status.json',status)
    try:
        manifest_path=transfer/'transfer_manifest.json';manifest=json.loads(manifest_path.read_text());complete=json.loads((transfer/'TRANSFER_COMPLETE.json').read_text())
        if not complete['complete'] or sha(manifest_path)!=complete['manifest_sha256']:raise ValueError('Transfer not verified')
        index={r['remote_rel']:r for r in manifest['files']};dataset=transfer/'dataset';rawmanifest=dataset/'dataset_raw.json'
        if sha(rawmanifest)!=index['dataset/dataset_raw.json']['sha256']:raise ValueError('Dataset manifest changed')
        data=json.loads(rawmanifest.read_text());validate_rows(data['records'])
        rows=[r for r in data['records'] if r['split'] in ('fit','selection')]
        # Verify metadata and source hashes against the fully verified transfer inventory.
        for r in rows:
            for a,b in [('path','png_sha256'),('raw_path','raw_sha256'),('pattern_path','pattern_sha256')]:
                name='dataset/'+r[a].replace('\\','/');entry=index[name]
                if entry['sha256']!=r[b] or (transfer/name).stat().st_size!=entry['size']:raise ValueError('Transfer correspondence changed')
            source=index['sources/DIV2K_train_HR/'+r['scene']+'.png']
            if source['sha256']!=r['source_sha256']:raise ValueError('Source hash mismatch')
        if limit is not None:
            # Diagnostic subset spans both sessions at3mm and all training depths; fit only.
            groups={}
            for r in rows:
                if r['split']=='fit':groups.setdefault((r['calibration_session'],r['depth_mm']),[]).append(r)
            rows=[r for group in groups.values() for r in group[:limit]]
        refs=make_references(dataset,out/'references',rows,index)
        for name in ('intensity','codes'): (out/name).mkdir()
        result=[]
        with (out/'records.jsonl').open('x',encoding='utf-8') as journal:
            with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=init_worker,initargs=(str(dataset),str(out))) as pool:
                for row in pool.map(preprocess_row,rows,chunksize=2):
                    result.append(row);journal.write(json.dumps(row)+'\n')
                    if len(result)%25==0 or len(result)==len(rows):
                        journal.flush();status.update(processed=len(result),expected=len(rows),updated_kst=now(),seconds=time.monotonic()-start);write_json(out/'status.json',status);print(json.dumps(status),flush=True)
        cache=dict(version='integrated-rgb-linear-v1',complete=True,source_dataset_sha256=sha(rawmanifest),transfer_manifest_sha256=complete['manifest_sha256'],
                   references=refs,records=result,counts=dict(Counter(r['split'] for r in result)),test_sealed_count=400,
                   test_pixels_decoded=False,shape=list(SHAPE),layout='CHW',normalization='fixed session/depth dark-white, no per-image fit',pilot_only=limit is not None)
        write_json(out/'manifest.json',cache);status.update(state='complete',updated_kst=now());write_json(out/'status.json',status)
    except Exception as e:
        status.update(state='failed',error=repr(e),traceback=traceback.format_exc(),updated_kst=now());write_json(out/'status.json',status);raise
def main():
    p=argparse.ArgumentParser();p.add_argument('transfer',type=Path);p.add_argument('out',type=Path);p.add_argument('--workers',type=int,default=4);p.add_argument('--pilot-per-group',type=int);a=p.parse_args();preprocess(a.transfer,a.out,a.workers,a.pilot_per_group)
if __name__=='__main__':main()
