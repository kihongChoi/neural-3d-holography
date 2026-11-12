"""Bounded camera/model update experiment; explicit provenance and existing-LUT baseline."""
import argparse, hashlib, json, shutil, time
from pathlib import Path
from datetime import datetime, timezone, timedelta
import numpy as np
import torch
from .dpac_lut_dataset import new_run
from .nh3d_lut_model import KNOTS,DEPTHS,optics,sha,write_json

def update_tables(center,scene,z):
    rng=np.random.default_rng(int(hashlib.sha256(f'{scene}|{z}|update-r1'.encode()).hexdigest()[:16],16))
    delta=rng.uniform(-6,6,(3,7));tables={'current_lut':center.copy()}
    for name,sign in [('probe_plus',1),('probe_minus',-1)]:
        channels=[]
        for c in range(3):
            y=center[c,KNOTS.astype(int)].astype(float)
            y[1:-1]=np.clip(y[1:-1]+sign*delta[c],KNOTS[1:-1]-12,KNOTS[1:-1]+12)
            channels.append(np.rint(np.interp(np.arange(256),KNOTS,y)).astype(np.uint8))
        tables[name]=np.stack(channels)
    for t in tables.values():
        assert t.shape==(3,256) and np.all(np.diff(t.astype(int),axis=1)>=0)
        assert np.max(np.abs(t.astype(int)-np.arange(256)))<=12
        assert np.all(t[:,0]==0) and np.all(t[:,-1]==255)
    return tables

def prepare(bank,candidate):
    from .nh3d_lut_ab import validate_candidate
    bank=Path(bank).resolve();candidate=Path(candidate).resolve()
    center,_=validate_candidate(bank,candidate)
    old=json.loads((bank/'manifest.json').read_text())
    assert len(old['splits']['fit'])==48 and len(old['splits']['selection'])==8
    spec=json.loads(json.dumps(old));spec['splits']['fit']=old['splits']['fit'][:24]
    reserved=set(sum(old['splits'].values(),[]))
    paths=sorted(p for p in (Path(old['dataset'])/'DIV2K_train_HR').glob('*.png') if p.stem not in reserved)
    selected=[paths[i] for i in np.random.default_rng(61106).permutation(len(paths))[:8]]
    spec['splits']['test']=[p.stem for p in selected]
    for p in selected:spec['sources'][p.stem]=p.relative_to(old['dataset']).as_posix()
    assert not(set(spec['splits']['test'])&reserved)
    spec['patterns']=[]
    for split in ['fit','selection','test']:
        for scene in spec['splits'][split]:
            for z in ((3.,5.,7.,9.) if split=='test' else DEPTHS):
                spec['patterns'].append(dict(id=len(spec['patterns']),scene=scene,split=split,method='DPAC',design_mm=z,iterations=0))
    root=new_run(Path('results'),'nh3d-model-update-r1').resolve()
    write_json(root/'manifest.json',spec)
    shutil.copyfile(candidate,root/'baseline_lut.npz')
    plan=dict(version='nh3d-camera-v4',optics={c:optics(c) for c in 'RGB'},fit_scenes=24,selection_scenes=8,held_out_scenes=8,depths_mm=list(DEPTHS),variants=['current_lut','probe_plus','probe_minus'],baseline_variant='current_lut',expected_capture_count=864,storage='full-frame PNG and user-approved linear NPZ',requested_modes=dict(R=[175,22],G=[175,6222],B=[85,12272],RGB=[15000,22]),update_protocol='model_update_v1',base_bank=str(bank),base_bank_sha256=sha(bank/'model_dataset.json'),baseline_candidate=str(candidate),baseline_candidate_sha256=sha(candidate),max_teacher_epochs=8,max_distillation_epochs=6,teacher_lr=1e-4,fit_steps_per_channel_epoch=432,old_new_sampling=[216,216],teacher_selection='Equal weight old and new selection mean MSE; initial teacher is eligible; old selection at each depth <= 1.05 times starting value',teacher_gate='New selection MSE improves, no >5% old depth regression, ASM superiority and original probe response gates',round2_policy='Only if round1 new-vs-current physical improvement exceeds within-scene repeat SD and >=120 minutes remain before deadline; otherwise finish round1',final_test_scenes=spec['splits']['test'],experiment_started_kst='2026-11-06T11:05:18+09:00',deadline_kst='2026-11-06T16:05:18+09:00')
    write_json(root/'collection_plan.json',plan)
    tables={str(r['id']):{k:v.tolist() for k,v in update_tables(center,r['scene'],r['design_mm']).items()} for r in spec['patterns'] if r['split']!='test'}
    write_json(root/'drive_tables.json',tables)
    snap=root/'source_snapshot';snap.mkdir()
    for name in ['nh3d_update.py','nh3d_lut_collect.py','nh3d_lut_data.py','nh3d_lut_model.py','dpac_lut_session.py','dpac_lut_fit.py','stage.py','hardware.py','readybeam.py']:
        shutil.copyfile(Path(__file__).parent/name,snap/name)
    with Path('docs/rgb-asm-nh3d-lut-plan.md').open('a',encoding='utf-8') as f:
        f.write('\n\n### 2026-11-06 모델 업데이트 실험 승인 및 사전 계획\n\n- 사용자가 모델 업데이트 실험을 승인했다. 새 실행 '+root.name+'. 11:05 KST부터 최대 5시간 규모로 1차 반복을 검증하고 조건부 2차를 수행한다. 기준은 기존 48장 모델과 동결 LUT e0f0f6506b767261b22314c35038f7a108b3e917be430aa39cb87c96cbe90415.\n- 1차는 기존 fit 원본 앞 24개와 selection 8개, 3/7/9 mm에서 현재 RGB LUT 및 주변 두 탐침을 촬영한다. 총 864장, 선형 학습 NPZ 저장 승인 유지. 탐침은 각 채널 9제어점 중 내부 7점에 결정적 ±6코드 변형을 더하되 기존 항등 대비 ±12 범위로 제한한다. 현재 LUT 자체는 그대로 측정하며 최종 재생은 항상 원래 DPAC 코드에서 표 한 번 조회한다. 청색 요청 85/12272 us, 실제 노출 82 us, fresh 0 mm 분리와 각 깊이 dark/white/정합 및 3% 안정성 검사를 요구한다.\n- 기존 데이터 216개와 새 데이터 216개를 채널/epoch당 섞어 총 432회 업데이트, 기존 체크포인트에서 Adam 1e-4 최대 8 epoch로 이어 학습한다. 모델 선택은 기존/신규 selection 오차를 동등 가중하며 각 기존 깊이 오차가 초기 대비 5% 이상 악화한 모델은 선택하지 않는다. 초기 모델도 후보로 보존한다. 새 모델은 신규 selection 예측 개선 및 기존 ASM/탐침 응답 기준을 통과해야 증류한다. 증류 최대 6 epoch, 기존 LUT를 정확한 비교 후보로 포함하고 원래 DPAC 코드에 직접 적용한다.\n- 라운드 선택은 기존 selection 8원본 × 3깊이 × 4모드 × 두 순서에서 기존 LUT 대 신규 LUT를 비교한다. 광학 개선이 원본별 반복 표준편차를 넘고 최종 평가 시간을 포함해 120분 이상 남을 때만 2차 반복을 진행한다. 2차를 생략해도 1차 결과와 이유를 기록한다. 이 설계는 업데이트 전체 효과를 평가하며 추가 학습 시간만의 효과를 분리하는 대조군 실험은 아니다.\n- 최종 시험은 기존 64원본과 겹치지 않는 새 DIV2K 원본 8개 '+','.join(spec['splits']['test'])+'를 사전 고정한다. 3/5/7/9 mm × 네 모드 × 기존/신규 LUT × 두 순서로 512장 평가한다. 이 원본과 5 mm는 모델 학습·선택에 사용하지 않는다. 기존 검증 8원본은 과거 결과로 보존한다.\n')
    print(root,flush=True)
    from .rgb_study_patterns import build
    build(root/'manifest.json',root/'patterns',0,None)
    write_json(root/'prepared_verification.json',dict(patterns=128,scene_captures=864,baseline_sha256=sha(root/'baseline_lut.npz'),probe_tables_sha256=sha(root/'drive_tables.json'),fresh_test_scenes=spec['splits']['test']))
    return root

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('bank',type=Path);p.add_argument('candidate',type=Path);a=p.parse_args();prepare(a.bank,a.candidate)
