"""Visualize teacher predictions before/after update against identical measured drives."""
import json,csv,shutil
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from .dpac_lut_dataset import new_run
from .nh3d_lut_data import CameraDataset
from .nh3d_lut_model import DepthSharedNH3D,optics,masked_mse,sha,write_json

CROP=(16,16,1696,928)
def display(a):
    v=np.clip(a*a,0,1)
    return np.uint8(np.rint(255*np.where(v<=.0031308,12.92*v,1.055*v**(1/2.4)-.055)))
def panel(out,images,metrics,keys,name,zoom=False):
    width,height=(432,237) if not zoom else (432,243)
    margin,gap,top,rowh=20,16,154,height+102
    canvas=Image.new('RGB',(margin*2+4*width+3*gap,top+rowh*len(keys)+16),'white');d=ImageDraw.Draw(canvas)
    font=lambda n:ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',n)
    d.text((margin,12),'추가학습 전후 모델 예측 비교',font=font(28),fill='#111111')
    d.text((margin,55),'동일 DPAC 구동·동일 실측 기준 | 광학 LUT 전후 촬영이 아닙니다',font=font(21),fill='#333333')
    d.text((margin,88),'실측은 독립 R/G/B 슬롯을 합성한 표시용 영상 | 후 모델: R 7 epoch · G 3 epoch · B 0 epoch 유지',font=font(17),fill='#555555')
    d.text((margin,116),'공통 정렬·고정 백색 정규화·동일 sRGB 표시, 개별 밝기/색 맞춤 없음'+(' | 공통 중앙 512×288 확대' if zoom else ''),font=font(16),fill='#555555')
    for row,key in enumerate(keys):
        scene,z=key;y=top+row*rowh
        for col,(method,label) in enumerate([('clean','원본 clean'),('camera','동일 실측 (슬롯 합성)'),('before','추가학습 전 예측'),('after','추가학습 후 선택모델 예측')]):
            x=margin+col*(width+gap);d.text((x,y),f'{scene} · {z:g} mm | {label}',font=font(18),fill='#111111')
            arr=images[key][method]
            if zoom:
                yy=(arr.shape[0]-288)//2;xx=(arr.shape[1]-512)//2;arr=arr[yy:yy+288,xx:xx+512]
            im=Image.fromarray(arr).resize((width,height),Image.Resampling.LANCZOS);canvas.paste(im,(x,y+29))
            if method in ('before','after'):
                values=[metrics[(scene,z,c)][method+'_crop_mse'] for c in 'RGB']
                d.text((x,y+height+34),f'{"전체 크롭 MSE" if zoom else "실측 대비 진폭 MSE"} {np.mean(values):.6f}',font=font(17),fill='#222222')
                d.text((x,y+height+60),'R/G/B  '+ ' / '.join(f'{v:.5f}' for v in values),font=font(14),fill='#444444')
    canvas.save(out/name)

def run(source):
    source=Path(source).resolve();out=new_run(Path('results'),'model-update-before-after-montage').resolve();(out/'aligned').mkdir()
    write_json(out/'analysis_plan.json',dict(source_run=str(source),split='selection',variant='current_lut',depths_mm=[3,7,9],crop_xywh=CROP,checkpoint_epochs=dict(R=7,G=3,B=0),preview='First two manifest selection scenes at 3 mm; all eight scenes also shown at each measured depth',display='Fixed white-normalized amplitude squared -> clipped sRGB, independently captured RGB slot channels composed for display, not integrated RGB camera capture',geometry='Existing independent per-channel/depth calibration in camera NPZ, shared native DPAC coordinates, common 16px border crop'))
    shutil.copy2(__file__,out/'source.py')
    with Path('docs/rgb-asm-nh3d-lut-plan.md').open('a',encoding='utf-8') as f:
        f.write('\n- 사용자 요청으로 추가학습 전후 모델 예측 몽타주를 생성한다. 분석 '+out.name+'. 신규 selection 8원본의 current_lut 구동만 3/7/9 mm에서 사용한다. 원본 clean, 기존 정합/고정 백색 정규화 실측 슬롯 합성, 기존 교사 예측, 선택 교사 R7/G3/B0 예측을 공통 [16,16,1696,928] 크롭으로 비교한다. 새 광학 A/B 사진이 아니며 B는 업데이트 전 모델 유지라고 명시한다. 최종 시험 원본/5 mm는 사용하지 않고 추가 학습·장비 구동은 하지 않는다.\n')
    print(out,flush=True)
    data=CameraDataset(source);manifest=json.loads((source/'manifest.json').read_text());plan=json.loads((source/'collection_plan.json').read_text())
    with np.load(plan['baseline_candidate'],allow_pickle=False) as f:parents=json.loads(str(f['metadata']))
    keys=[(scene,z) for z in (3.,7.,9.) for scene in manifest['splits']['selection']]
    images={key:{method:np.zeros((928,1696,3),np.uint8) for method in ['clean','camera','before','after']} for key in keys};metrics={};provenance={};audit=[]
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    for channel,tag in enumerate('RGB'):
        rows=[r for r in data.rows if r['channel']==tag and r['split']=='selection' and r['variant']=='current_lut']
        directory=source/'updated_training_01'/tag;checks=json.loads((directory/'forward_checks.json').read_text());initial=json.loads((directory/'initial_evaluation.json').read_text())
        old=Path(initial['checkpoint']);new=directory/'best.pt'
        assert sha(old)==parents[channel]['teacher_sha256'] and sha(new)==checks['teacher_sha256']
        provenance[tag]=dict(before_checkpoint=str(old),before_sha256=sha(old),after_checkpoint=str(new),after_sha256=sha(new),after_epoch=checks['best_epoch'],lut_distillation_eligible=checks['accepted_for_distillation'])
        net=DepthSharedNH3D(optics(tag),'cuda').eval().requires_grad_(False)
        for method,checkpoint,reference in [('before',old,initial['new']['records']),('after',new,checks['final_new']['records'])]:
            saved=torch.load(checkpoint,weights_only=True,map_location='cpu');net.load_learnable(saved['weights']);del saved
            expected={r['name']:r['model_mse'] for r in reference}
            with torch.no_grad():
                for row in rows:
                    codes,measured,mask=data.load(row,'cuda');predicted=net.amplitude(codes,row['depth_mm']);target=data.target(row,'cuda')
                    full=float(masked_mse(predicted,measured,mask));delta=abs(full-expected['b1_'+row['name']]);assert delta<1e-7,(tag,row['name'],delta)
                    y,x=16,16;crop=(slice(y,y+928),slice(x,x+1696));valid=mask[crop];loss=float(masked_mse(predicted[crop],measured[crop],valid))
                    key=(row['scene'],row['depth_mm']);mkey=(*key,tag);record=metrics.setdefault(mkey,dict(scene=key[0],depth_mm=key[1],channel=tag,camera_path=str(source/row['camera_path']),camera_sha256=row['camera_sha256'],array_path=str(source/row['array_path']),array_sha256=row['array_sha256'],displayed_channel_sha256=row['displayed_channel_sha256'],valid_crop_fraction=float(valid.float().mean())))
                    record[method+'_crop_mse']=loss;record[method+'_full_mse']=full;audit.append(delta)
                    images[key][method][...,channel]=display(predicted[crop].cpu().numpy())
                    if method=='before':
                        images[key]['camera'][...,channel]=display(measured[crop].cpu().numpy());images[key]['clean'][...,channel]=display(target[crop].cpu().numpy())
        del net;torch.cuda.empty_cache();print(tag,'24 before/after predictions verified against saved evaluation',flush=True)
    for key,panels in images.items():
        for method,img in panels.items():Image.fromarray(img).save(out/'aligned'/f'{key[0]}_z{key[1]:g}_{method}.png')
    scenes=manifest['splits']['selection']
    for z in (3.,7.,9.):panel(out,images,metrics,[(s,z) for s in scenes],f'all_selection_{z:g}mm.png')
    preview=[(s,3.) for s in scenes[:2]];panel(out,images,metrics,preview,'preview_3mm.png');panel(out,images,metrics,preview,'detail_3mm.png',zoom=True)
    summaries={}
    for tag in 'RGB':
        group=[r for r in metrics.values() if r['channel']==tag];before=float(np.mean([r['before_crop_mse'] for r in group]));after=float(np.mean([r['after_crop_mse'] for r in group]));summaries[tag]=dict(before=before,after=after,reduction=1-after/before)
    assert all(np.array_equal(v['before'][...,2],v['after'][...,2]) for v in images.values())
    write_json(out/'results.json',dict(source_run=str(source),dataset_sha256=data.fingerprint,source_status_sha256=sha(source/'model_acquisition/status.json'),provenance=provenance,summary_current_lut_only=summaries,crop_xywh=CROP,full_prediction_mse_max_difference_from_saved_evaluation=max(audit),b_display_unchanged=True,rows=list(metrics.values()),note='Prediction error against the same measured camera response; not optical quality improvement. Slot composite is not an integrated RGB camera photo. Metrics shown on detail montage still refer to the full common crop.'))
    with (out/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(next(iter(metrics.values()))));w.writeheader();w.writerows(metrics.values())
    write_json(out/'checksums.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
    with Path('docs/rgb-asm-nh3d-lut-plan.md').open('a',encoding='utf-8') as f:
        f.write('\n- 모델 예측 몽타주 완료 '+out.name+'. current_lut 조건 선택 장면 24개에 대해 채널별 전/후 예측 144건을 계산하고 저장된 full-field 평가값과 오차 <1e-7로 일치함을 확인했다. 3/7/9 mm별 전체 8장 몽타주, 첫 두 선택 원본의 3 mm 미리보기·공통 중앙 확대, 정렬 표시 PNG 96장을 저장했다. B 전후 표시 픽셀은 동일함을 확인했다. 지표는 current_lut 조건의 공통 크롭 실측 예측 오차이며 이전 보고의 세 탐침 전체 평균과 구분한다. 파일/모델/촬영 출처와 해시는 results.json, metrics.csv, checksums.json에 저장했다.\n')
    print(json.dumps(dict(output=str(out),summary=summaries,max_mse_reproduction_error=max(audit))),flush=True)
    return out

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);run(p.parse_args().source)
