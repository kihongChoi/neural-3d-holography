"""Warm-start model update with old/new replay and bounded LUT redistillation."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from .nh3d_lut_data import CameraDataset
from .nh3d_lut_model import DepthSharedNH3D,CodeLUT,KNOTS,DEPTHS,optics,masked_mse,sha,write_json
from .nh3d_lut_train import evaluate,probe_check,checkpoint

class ReplayData:
    def __init__(self,root):
        self.root=Path(root).resolve();self.plan=json.loads((self.root/'collection_plan.json').read_text())
        self.banks=[CameraDataset(self.plan['base_bank']),CameraDataset(self.root)]
        if self.banks[0].fingerprint!=self.plan['base_bank_sha256']:raise ValueError('Base bank changed')
        self.rows=[dict(r,_bank=i,name=f'b{i}_{r["name"]}') for i,b in enumerate(self.banks) for r in b.rows]
        seen={}
        forbidden=set(self.plan['final_test_scenes'])
        for r in self.rows:
            if seen.setdefault(r['scene'],r['split'])!=r['split'] or r['scene'] in forbidden or r['depth_mm']==5:raise ValueError('Scene or depth leakage')
        self.manifest_path=self.root/'combined_dataset.json'
        spec=dict(version='model-update-replay-v1',banks=[dict(path=str(b.root),sha256=b.fingerprint) for b in self.banks],plan_sha256=sha(self.root/'collection_plan.json'),records=len(self.rows),new_test_scenes=self.plan['final_test_scenes'])
        if self.manifest_path.exists():
            if json.loads(self.manifest_path.read_text())!=spec:raise ValueError('Replay spec changed')
        else:write_json(self.manifest_path,spec)
        self.fingerprint=sha(self.manifest_path)
    def load(self,row,device):return self.banks[row['_bank']].load(row,device)
    def original(self,row,device):return self.banks[row['_bank']].original(row,device)
    def target(self,row,device):return self.banks[row['_bank']].target(row,device)
    def check_unchanged(self):
        if (self.root/'STOP_REQUESTED').exists():raise RuntimeError('STOP_REQUESTED')
        if sha(self.manifest_path)!=self.fingerprint:raise ValueError('Replay changed')
        for b in self.banks:b.check_unchanged()

def scores(net,data,rows):
    result=evaluate(net,data,rows)
    by_depth={str(z):dict(model_mse=float(np.mean([r['model_mse'] for r in result if r['depth_mm']==z])),asm_mse=float(np.mean([r['asm_mse'] for r in result if r['depth_mm']==z]))) for z in DEPTHS}
    return dict(mean_mse=float(np.mean([r['model_mse'] for r in result])),by_depth=by_depth,records=result)

def train(data,out,tag,epochs=8):
    dest=out/tag;dest.mkdir(exist_ok=False)
    with np.load(data.plan['baseline_candidate'],allow_pickle=False) as f:parent_meta=json.loads(str(f['metadata']))['RGB'.index(tag)]
    old_path=Path(parent_meta.get('checkpoint_path',str(Path(data.plan['baseline_candidate']).parent/tag/'best.pt')))
    if sha(old_path)!=parent_meta['teacher_sha256']:raise ValueError('Warm-start teacher provenance mismatch')
    saved=torch.load(old_path,weights_only=True,map_location='cpu')
    net=DepthSharedNH3D(optics(tag),'cuda');net.load_learnable(saved['weights'])
    train_rows=[[r for r in data.rows if r['channel']==tag and r['split']=='fit' and r['_bank']==i] for i in [0,1]]
    val_rows=[[r for r in data.rows if r['channel']==tag and r['split']=='selection' and r['_bank']==i] for i in [0,1]]
    initial=[scores(net,data,r) for r in val_rows]
    write_json(dest/'initial_evaluation.json',dict(checkpoint=str(old_path),checkpoint_sha256=sha(old_path),old=initial[0],new=initial[1]))
    best=float(np.mean([x['mean_mse'] for x in initial]));history=[];stale=0
    def save_best(epoch):
        checkpoint(dest/'best.pt',dict(weights=net.learnable_state(),config=optics(tag),architecture=net.architecture,dataset_sha256=data.fingerprint,epoch=epoch,selection_mse=best,parent_checkpoint_sha256=sha(old_path),source_sha256=sha(__file__)))
    save_best(0)
    optimizer=torch.optim.Adam(net.parameters(),lr=1e-4)
    print(tag,'initial old/new MSE',initial[0]['mean_mse'],initial[1]['mean_mse'],flush=True)
    for epoch in range(epochs):
        start=time.monotonic();net.train();rng=np.random.default_rng(61106+epoch);losses=[]
        selected=[]
        for pool in train_rows:
            if len(pool)<216:raise ValueError('Insufficient replay samples')
            selected.extend(pool[i] for i in rng.permutation(len(pool))[:216])
        for i in rng.permutation(len(selected)):
            data.check_unchanged();row=selected[i];codes,measured,mask=data.load(row,'cuda')
            optimizer.zero_grad(set_to_none=True);loss=masked_mse(net.amplitude(codes,row['depth_mm']),measured,mask)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite update loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),10.,error_if_nonfinite=True);optimizer.step();losses.append(float(loss.detach()))
        current=[scores(net,data,r) for r in val_rows]
        score=float(np.mean([v['mean_mse'] for v in current]))
        retained=all(current[0]['by_depth'][str(z)]['model_mse']<=1.05*initial[0]['by_depth'][str(z)]['model_mse'] for z in DEPTHS)
        improved=retained and score<best
        if improved:best=score;save_best(epoch+1);stale=0
        else:stale+=1
        entry=dict(epoch=epoch+1,updates=len(selected),train_mse=float(np.mean(losses)),old_mse=current[0]['mean_mse'],new_mse=current[1]['mean_mse'],balanced_selection_mse=score,old_depth_retention_passed=retained,selected=improved,seconds=time.monotonic()-start)
        history.append(entry);write_json(dest/'history.json',history);print(tag,entry,flush=True)
        if stale>=3:break
    saved=torch.load(dest/'best.pt',weights_only=True,map_location='cpu');net.load_learnable(saved['weights'])
    final=[scores(net,data,r) for r in val_rows]
    # Rename only the probe grouping key; measured drive_lut remains unchanged.
    response_rows=[dict(r,variant='identity' if r['variant']=='current_lut' else r['variant']) for r in val_rows[1]]
    probes=probe_check(net,data,response_rows)
    response_ok=probes['significant_count']>=12 and probes['median_response_cosine']>.25 and probes['target_delta_sign_agreement']>=.65
    forward_ok=all(v['model_mse']<v['asm_mse'] for x in final for v in x['by_depth'].values())
    retention=all(final[0]['by_depth'][str(z)]['model_mse']<=1.05*initial[0]['by_depth'][str(z)]['model_mse'] for z in DEPTHS)
    new_improved=final[1]['mean_mse']<initial[1]['mean_mse']
    checks=dict(channel=tag,dataset_sha256=data.fingerprint,teacher_sha256=sha(dest/'best.pt'),accepted_for_distillation=bool(response_ok and forward_ok and retention and new_improved),forward_ok=bool(forward_ok),response_ok=bool(response_ok),old_retention=bool(retention),new_improved=bool(new_improved),initial_old_mse=initial[0]['mean_mse'],initial_new_mse=initial[1]['mean_mse'],final_old=final[0],final_new=final[1],probes=probes,best_epoch=saved['epoch'])
    write_json(dest/'forward_checks.json',checks);print(tag,'update gates',checks['accepted_for_distillation'],'best epoch',saved['epoch'],flush=True)
    del net,optimizer;torch.cuda.empty_cache();return checks

def distil(data,out,tag,checks,epochs=6):
    idx='RGB'.index(tag)
    with np.load(data.plan['baseline_candidate'],allow_pickle=False) as f:baseline=f['table'][idx].copy();prior=json.loads(str(f['metadata']))[idx]
    old_dir=Path(data.plan['baseline_candidate']).parent/tag
    prior_checkpoint=Path(prior.get('checkpoint_path',str(old_dir/'best.pt')))
    prior_checks=Path(prior.get('forward_checks_path',str(old_dir/'forward_checks.json')))
    common=dict(channel=tag,update_protocol='model_update_v1',update_round=str(data.root),combined_dataset_path=str(data.manifest_path),combined_dataset_sha256=data.fingerprint,baseline_candidate_sha256=data.plan['baseline_candidate_sha256'],direction='original inverted DPAC uint8 code -> HDMI uint8 code; one lookup',bound_codes=12.,wavelength_nm={'R':638,'G':520,'B':450}[tag],active_xywh=[96,60,1728,960],training_depths_mm=list(DEPTHS),held_out_depth_mm=5.,application='render RGB -> linear intensity -> amplitude -> single-pass wavelength-specific DPAC -> inverted uint8 encoding -> channel LUT once -> HDMI',scope='Global effective code correction; not an identified physical phase response')
    if not checks['accepted_for_distillation']:
        return baseline,{**prior,**common,'checkpoint_path':str(prior_checkpoint),'forward_checks_path':str(prior_checks),'retained_baseline':True,'retention_reason':'Updated teacher failed preregistered gate; retain original teacher/LUT provenance'}
    dest=out/tag/'distillation';dest.mkdir(exist_ok=False)
    teacher=DepthSharedNH3D(optics(tag),'cuda').eval().requires_grad_(False)
    teacher.load_learnable(torch.load(out/tag/'best.pt',weights_only=True,map_location='cpu')['weights'])
    student=CodeLUT().cuda()
    with torch.no_grad():student.delta.copy_(torch.atanh(torch.as_tensor(np.clip((baseline[KNOTS.astype(int)][1:-1].astype(float)-KNOTS[1:-1])/12,-.9999,.9999),device='cuda',dtype=torch.float32)))
    optimizer=torch.optim.Adam(student.parameters(),lr=.0125)
    fit=[r for r in data.rows if r['_bank']==0 and r['split']=='fit' and r['variant']=='identity' and r['channel']==tag]
    selection=[r for r in data.rows if r['_bank']==0 and r['split']=='selection' and r['variant']=='identity' and r['channel']==tag]
    def score(table):
        values=[]
        with torch.no_grad():
            t=torch.as_tensor(table,device='cuda')
            for row in selection:
                data.check_unchanged();base,_,mask=data.load(row,'cuda');target=data.target(row,'cuda')
                values.append(float(masked_mse(teacher.amplitude(t[base.long()],row['depth_mm']),target,mask)))
        return float(np.mean(values))
    original_score=score(baseline);best=original_score;table=baseline.copy();history=[]
    for epoch in range(epochs):
        losses=[];start=time.monotonic()
        for i in np.random.default_rng(61116+epoch).permutation(len(fit)):
            data.check_unchanged();row=fit[i];base,_,mask=data.load(row,'cuda');target=data.target(row,'cuda')
            optimizer.zero_grad(set_to_none=True);loss=masked_mse(teacher.amplitude(student(base),row['depth_mm']),target,mask)
            regularizer=1e-4*(student.table(False)-torch.arange(256,device='cuda')).square().mean()/144
            (loss+regularizer).backward();torch.nn.utils.clip_grad_norm_(student.parameters(),1.,error_if_nonfinite=True);optimizer.step();losses.append(float(loss.detach()))
        current=student.export();val=score(current)
        if val<best:best=val;table=current.copy()
        history.append(dict(epoch=epoch+1,train_predicted_mse=float(np.mean(losses)),selection_predicted_mse=val,best_predicted_mse=best,seconds=time.monotonic()-start));write_json(dest/'history.json',history);print('update distil',tag,history[-1],flush=True)
    assert all(p.grad is None for p in teacher.parameters())
    metadata=dict(common,dataset_sha256=data.fingerprint,teacher_sha256=sha(out/tag/'best.pt'),checkpoint_path=str(out/tag/'best.pt'),forward_checks_path=str(out/tag/'forward_checks.json'),retained_baseline=bool(np.array_equal(table,baseline)),baseline_predicted_mse=original_score,candidate_predicted_mse=best,origin='Redistilled through frozen updated camera-trained NH3D; no camera during gradient updates')
    np.savez(dest/'lut.npz',table=table,metadata=json.dumps(metadata));write_json(dest/'metadata.json',metadata)
    del teacher,student,optimizer;torch.cuda.empty_cache();return table,metadata

def run(root):
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    data=ReplayData(root);out=data.root/'updated_training_01';out.mkdir(exist_ok=False)
    tables=[];metadata=[];checks=[]
    for tag in 'RGB':
        torch.manual_seed(61106+'RGB'.index(tag))
        c=train(data,out,tag);checks.append(c)
        t,m=distil(data,out,tag,c);tables.append(t);metadata.append(m)
    np.savez(out/'rgb_lut_candidate.npz',table=np.stack(tables),identity=np.tile(np.arange(256,dtype=np.uint8),(3,1)),metadata=json.dumps(metadata))
    with np.load(data.plan['baseline_candidate']) as f:changed=int(np.sum(np.stack(tables)!=f['table']))
    write_json(out/'completed.json',dict(state='updated_models_and_lut_complete',candidate_sha256=sha(out/'rgb_lut_candidate.npz'),changed_entries=changed,teacher_gates={c['channel']:c['accepted_for_distillation'] for c in checks},dataset_sha256=data.fingerprint))
    print('UPDATE COMPLETE',out,'changed entries',changed,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);run(p.parse_args().root)
