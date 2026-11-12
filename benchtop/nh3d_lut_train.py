"""Train camera-calibrated NH3D teachers, then distil bounded code LUTs.

Explicit dataset argument only. No hardware control. Target images are used
only by the student, never as physical-model training labels.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from .nh3d_lut_data import CameraDataset
from .nh3d_lut_model import DepthSharedNH3D, CodeLUT, DEPTHS, optics, masked_mse, sha, write_json


def checkpoint(path, payload):
    tmp=path.with_suffix('.partial.pt')
    torch.save(payload,tmp)
    tmp.replace(path)


def guard(data):
    if (data.root/'STOP_REQUESTED').exists():
        raise RuntimeError('STOP_REQUESTED')
    data.check_unchanged()


def evaluate(net,data,rows):
    result=[]
    net.eval()
    with torch.no_grad():
        for row in rows:
            guard(data)
            codes,amp,mask=data.load(row,'cuda')
            prediction=net.amplitude(codes,row['depth_mm'])
            ideal=net.ideal_amplitude(codes,row['depth_mm'])
            result.append(dict(name=row['name'],scene=row['scene'],depth_mm=row['depth_mm'],variant=row['variant'],
                model_mse=float(masked_mse(prediction,amp,mask)),asm_mse=float(masked_mse(ideal,amp,mask))))
    return result


def train_teacher(data,out,tag,epochs,patience):
    destination=out/tag
    destination.mkdir(exist_ok=False)
    fit=[r for r in data.rows if r['channel']==tag and r['split']=='fit']
    selection=[r for r in data.rows if r['channel']==tag and r['split']=='selection']
    torch.manual_seed(1105+'RGB'.index(tag))
    net=DepthSharedNH3D(optics(tag),'cuda')
    optimizer=torch.optim.Adam(net.parameters(),lr=5e-4)
    best=float('inf'); stale=0; history=[]
    for epoch in range(epochs):
        net.train(); losses=[]; start=time.perf_counter()
        for index in np.random.default_rng(1105+epoch).permutation(len(fit)):
            guard(data)
            row=fit[int(index)]
            codes,measured,mask=data.load(row,'cuda')
            optimizer.zero_grad(set_to_none=True)
            loss=masked_mse(net.amplitude(codes,row['depth_mm']),measured,mask)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite physical-model loss')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(net.parameters(),10.,error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        checks=evaluate(net,data,selection)
        score=float(np.mean([r['model_mse'] for r in checks]))
        improved=score<best
        if improved:
            best=score; stale=0
            checkpoint(destination/'best.pt',dict(weights=net.learnable_state(),config=optics(tag),
                architecture=net.architecture,dataset_sha256=data.fingerprint,epoch=epoch+1,selection_mse=best,
                source_sha256={name:sha(Path(__file__).parent/name) for name in ('nh3d_lut_train.py','nh3d_lut_model.py','nh3d_lut_data.py')},cudnn_deterministic=True))
        else:stale+=1
        history.append(dict(epoch=epoch+1,train_mse=float(np.mean(losses)),selection_mse=score,
            best_selection_mse=best,seconds=time.perf_counter()-start))
        write_json(destination/'history.json',history)
        write_json(out/'status.json',dict(state='training_teacher',channel=tag,**history[-1]))
        print(tag,history[-1],flush=True)
        if stale>=patience:break
    saved=torch.load(destination/'best.pt',weights_only=True,map_location='cpu')
    net.load_learnable(saved['weights'])
    checks=evaluate(net,data,selection)
    by_depth={str(z):dict(model_mse=float(np.mean([r['model_mse'] for r in checks if r['depth_mm']==z])),
                         asm_mse=float(np.mean([r['asm_mse'] for r in checks if r['depth_mm']==z]))) for z in DEPTHS}
    probes=probe_check(net,data,selection)
    forward_ok=all(v['model_mse']<v['asm_mse'] for v in by_depth.values())
    # A held-out measured response check, not merely a good fit to appearance.
    response_ok=probes['count']>=12 and probes['significant_count']>=12 and probes['median_response_cosine']>0.25 and probes['target_delta_sign_agreement']>=.65
    report=dict(channel=tag,dataset_sha256=data.fingerprint,teacher_sha256=sha(destination/'best.pt'),
                by_depth=by_depth,records=checks,probes=probes,forward_ok=forward_ok,response_ok=response_ok,
                accepted_for_distillation=bool(forward_ok and response_ok),best_epoch=saved['epoch'])
    write_json(destination/'forward_checks.json',report)
    del net,optimizer;torch.cuda.empty_cache()
    return report


def probe_check(net,data,rows):
    groups={}
    for row in rows:groups.setdefault((row['scene'],row['depth_mm']),{})[row['variant']]=row
    comparisons=[]
    net.eval()
    with torch.no_grad():
        for (scene,z),group in groups.items():
            if 'identity' not in group:raise ValueError('No baseline for response check')
            base=group['identity']
            code,a,valid=data.load(base,'cuda')
            pred=net.amplitude(code,z)
            target=data.target(base,'cuda')
            for variant,row in group.items():
                if variant=='identity':continue
                code,b,mask=data.load(row,'cuda');mask=mask & valid
                pb=net.amplitude(code,z)
                measured_delta=(b-a)[mask]; predicted_delta=(pb-pred)[mask]
                denominator=measured_delta.norm()*predicted_delta.norm()
                cosine=float((measured_delta*predicted_delta).sum()/denominator.clamp_min(1e-12))
                dm=float(masked_mse(b,target,mask)-masked_mse(a,target,mask))
                dp=float(masked_mse(pb,target,mask)-masked_mse(pred,target,mask))
                comparisons.append(dict(scene=scene,depth_mm=z,variant=variant,response_cosine=cosine,
                    measured_target_mse_delta=dm,predicted_target_mse_delta=dp))
    significant=[r for r in comparisons if abs(r['measured_target_mse_delta'])>=.001]
    agreement=float(np.mean([np.sign(r['measured_target_mse_delta'])==np.sign(r['predicted_target_mse_delta']) for r in significant])) if significant else 0.
    return dict(count=len(comparisons),significant_count=len(significant),
                median_response_cosine=float(np.median([r['response_cosine'] for r in comparisons])) if comparisons else 0.,
                target_delta_sign_agreement=agreement,records=comparisons)


def distil(data,out,tag,epochs):
    teacher_dir=out/tag
    checks=json.loads((teacher_dir/'forward_checks.json').read_text())
    if checks['dataset_sha256']!=data.fingerprint or not checks['accepted_for_distillation']:
        raise RuntimeError(f'{tag}: physical-model validation/response gate failed; do not distil')
    if checks['teacher_sha256']!=sha(teacher_dir/'best.pt'):raise ValueError('Teacher changed after validation')
    saved=torch.load(teacher_dir/'best.pt',weights_only=True,map_location='cpu')
    teacher=DepthSharedNH3D(optics(tag),'cuda').eval().requires_grad_(False)
    teacher.load_learnable(saved['weights'])
    student=CodeLUT().cuda()
    optimizer=torch.optim.Adam(student.parameters(),lr=.025)
    fit=[r for r in data.rows if r['channel']==tag and r['split']=='fit' and r['variant']=='identity']
    selection=[r for r in data.rows if r['channel']==tag and r['split']=='selection' and r['variant']=='identity']
    destination=teacher_dir/'distillation'
    destination.mkdir(exist_ok=False)
    def score():
        values=[]
        with torch.no_grad():
            for row in selection:
                guard(data)
                base,_,mask=data.load(row,'cuda');target=data.target(row,'cuda')
                values.append(float(masked_mse(teacher.amplitude(student(base),row['depth_mm']),target,mask)))
        return float(np.mean(values))
    identity_score=score();best=identity_score;table=student.export();history=[]
    for epoch in range(epochs):
        losses=[]
        for index in np.random.default_rng(2205+epoch).permutation(len(fit)):
            guard(data);row=fit[int(index)]
            base,_,mask=data.load(row,'cuda');target=data.target(row,'cuda')
            optimizer.zero_grad(set_to_none=True)
            loss=masked_mse(teacher.amplitude(student(base),row['depth_mm']),target,mask)
            regularizer=1e-4*(student.table(False)-torch.arange(256,device='cuda')).square().mean()/144
            (loss+regularizer).backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();losses.append(float(loss.detach()))
        val=score()
        if val<best:best=val;table=student.export()
        history.append(dict(epoch=epoch+1,train_predicted_mse=float(np.mean(losses)),selection_predicted_mse=val,best_predicted_mse=best))
        write_json(destination/'history.json',history)
        print('distil',tag,history[-1],flush=True)
    if any(p.grad is not None for p in teacher.parameters()):raise RuntimeError('Teacher was not frozen')
    meta=dict(channel=tag,dataset_sha256=data.fingerprint,teacher_sha256=sha(teacher_dir/'best.pt'),
        origin='distilled through frozen camera-trained NH3D; not an identified physical phase response',
        direction='inverted uint8 DPAC code -> uint8 HDMI code',application='table[original_code] exactly once',
        wavelength_nm=round(optics(tag)['wavelength_m']*1e9),active_xywh=[96,60,1728,960],
        training_depths_mm=list(DEPTHS),held_out_depth_mm=5.,identity_predicted_mse=identity_score,
        candidate_predicted_mse=best,measured_AB='pending',bound_codes=12.,
        scope='DIV2K design-plane data; GS transfer and same-hologram multiplane generalization unverified')
    np.savez(destination/'lut.npz',table=table,identity=np.arange(256,dtype=np.uint8),metadata=json.dumps(meta))
    write_json(destination/'metadata.json',dict(meta,npz_sha256=sha(destination/'lut.npz')))
    del teacher,student,optimizer;torch.cuda.empty_cache()
    return table,meta


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--label',default='nh3d_training_01')
    p.add_argument('--stage',choices=['teacher','distil'],required=True)
    p.add_argument('--channels',default='RGB',choices=['RGB','R','G','B'])
    p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--patience',type=int,default=6)
    p.add_argument('--distil-epochs',type=int,default=12)
    a=p.parse_args()
    if a.epochs<1 or a.patience<1 or a.distil_epochs<1:raise ValueError('Positive iteration limits required')
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    data=CameraDataset(a.root)
    if Path(a.label).name!=a.label:raise ValueError('Label must be a single directory name')
    out=data.root/a.label
    if a.stage=='teacher':out.mkdir(exist_ok=False)
    elif not out.is_dir():raise ValueError('Teacher directory missing')
    tables=[];metas=[]
    for tag in a.channels:
        if a.stage=='teacher':train_teacher(data,out,tag,a.epochs,a.patience)
        else:
            table,meta=distil(data,out,tag,a.distil_epochs);tables.append(table);metas.append(meta)
    if a.stage=='distil' and a.channels=='RGB':
        path=out/'rgb_lut_candidate.npz'
        if path.exists():raise FileExistsError(path)
        np.savez(path,table=np.stack(tables),identity=np.tile(np.arange(256,dtype=np.uint8),(3,1)),metadata=json.dumps(metas))
    write_json(out/'status.json',dict(state=a.stage+'_complete',channels=a.channels,measured_AB='pending'))

if __name__=='__main__':main()
