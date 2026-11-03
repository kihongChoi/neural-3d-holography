"""Render diagnostic measurements without touching models or hardware."""
import argparse
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def image(ax,path,label):
    with Image.open(path) as source:
        im=source.convert('RGB');im.thumbnail((800,460),Image.Resampling.LANCZOS)
    ax.imshow(im);ax.set_xticks([]);ax.set_yticks([]);ax.set_title(label,fontsize=10,pad=8)


def legend(ax,**kwargs):
    lg=ax.legend(facecolor='#142330',edgecolor='#526575',**kwargs)
    plt.setp(lg.get_texts(),color='#eef3f7')


def render(root):
    root=Path(root).resolve();out=root/'report';out.mkdir(exist_ok=True)
    font_manager.fontManager.addfont('C:/Windows/Fonts/malgun.ttf')
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,
        'figure.facecolor':'#101923','axes.facecolor':'#142330','text.color':'#eef3f7',
        'axes.labelcolor':'#d7e3ec','xtick.color':'#b7c8d6','ytick.color':'#b7c8d6',
        'axes.edgecolor':'#526575','font.size':10})
    stamp=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M KST')
    stability=read(root/'stability_summary.json');fixed=rows(root/'stability.jsonl')
    reference=rows(root/'references.jsonl');protocol=read(root/'protocol.json')
    gradients=read(root/'gradient_summary.json');conditions=read(root/'citl_summary.json')
    preservation=read(root/'source_preservation.json') if (root/'source_preservation.json').exists() else {'pending':True}
    fig=plt.figure(figsize=(15,10),dpi=165)
    fig.suptitle('고정 위상 반복 측정 · 단기 잡음과 실험 중 드리프트',x=.035,y=.975,ha='left',fontsize=19,weight='bold')
    fig.text(.035,.929,stamp+f'  |  4개 고정 위상 × {protocol["stability_rounds"]}회  |  최적화 사이에 동일 0846 기준 위상 재촬영',color='#b7c8d6',fontsize=11)
    gs=fig.add_gridspec(3,2,left=.065,right=.98,top=.86,bottom=.09,height_ratios=[1,1,.7],hspace=.5,wspace=.25)
    ax=fig.add_subplot(gs[0,0])
    for j,identity in enumerate(stability):
        selected=[r for r in fixed if r['id']+'_'+r['phase']==identity]
        ax.plot([(r['time']-fixed[0]['time'])/60 for r in selected],[r['amplitude_mse'] for r in selected],'.-',label=identity)
    ax.set(xlabel='반복 측정 시작 이후 [분]',ylabel='진폭 MSE',title='같은 위상에서도 변하는 오차');ax.grid(alpha=.15);legend(ax,fontsize=8)
    ax=fig.add_subplot(gs[0,1])
    for ci,(ch,color) in enumerate(zip(('R','G','B'),('#ef8b83','#83dca3','#80b3ed'))):
        times=[(r['time']-reference[0]['time'])/60 for r in reference]
        values=np.array([r['mean_intensity_rgb'][ci] for r in reference]);ax.plot(times,100*(values/values[0]-1),'.-',label=ch,color=color)
    ax.set(xlabel='실험 시작 이후 [분]',ylabel='고정 기준 위상의 평균 광량 변화 [%]',title='실험 전체에 걸친 기준 위상 응답');ax.grid(alpha=.15);legend(ax,fontsize=9)
    for j,identity in enumerate(('0846_initial','0898_initial')):
        with np.load(root/f'stability_{identity}.npz') as values:noise=values['std_amplitude'].astype(np.float32)
        ax=fig.add_subplot(gs[1,j]);vmax=max(float(np.percentile(noise,99)),.001)
        im=ax.imshow(np.sqrt(np.mean(noise**2,axis=-1)),vmin=0,vmax=vmax,cmap='magma');ax.set_xticks([]);ax.set_yticks([])
        ax.set_title(identity+' · 픽셀별 진폭 표준편차',fontsize=11);fig.colorbar(im,ax=ax,fraction=.026)
    ax=fig.add_subplot(gs[2,:]);ax.axis('off')
    lines=['고정 위상별 평균 광량 CV [R / G / B]       |       MSE 평균 ± 표준편차']
    for name,s in stability.items():
        cv=' / '.join(f'{100*v:.2f}%' for v in s['intensity_cv_rgb'])
        lines.append(f'{name:18s}   {cv}       |       {s["mse_mean"]:.5f} ± {s["mse_std"]:.5f}')
    ax.text(0,1,'\n'.join(lines),va='top',fontsize=10,linespacing=1.8)
    fig.text(.035,.025,'픽셀 표준편차는 광학·센서·타이밍·환경 변동을 합친 측정치이며, 센서 단독 read noise 측정은 아닙니다.',fontsize=10,color='#b7c8d6')
    fig.savefig(out/'stability_montage.png',bbox_inches='tight',pad_inches=.25);plt.close(fig)

    # Predeclared direction 0 and epsilon .1, not a direction selected for effect size.
    fig=plt.figure(figsize=(15,10),dpi=165)
    fig.suptitle('실측과 모델의 위상 민감도 비교 · 중앙 유한차분',x=.035,y=.975,ha='left',fontsize=19,weight='bold')
    fig.text(.035,.929,'0846 · 고정 direction 0 / ε = 0.1 rad  |  카메라 2쌍 평균  |  영상: G 채널 / 요약: RGB 전체',fontsize=11,color='#b7c8d6')
    gs=fig.add_gridspec(3,3,left=.055,right=.97,top=.86,bottom=.09,height_ratios=[1,1,.8],hspace=.48,wspace=.16)
    for i,base in enumerate(('initial','old_best')):
        row=next(r for r in gradients if r['base']==base and r['direction']==0 and r['epsilon']==.1)
        with np.load(root/f'fd_{base}_d0_e0.1.npz') as values:
            arrays={name:values[name].astype(np.float32)[...,1] for name in ('camera_fd_mean','asm_fd','cnn_fd')}
        vmax=max(float(np.percentile(np.abs(np.concatenate([a.ravel() for a in arrays.values()])),99)),1e-5)
        for j,(name,label) in enumerate((('camera_fd_mean','실제 카메라'),('asm_fd','ASM'),('cnn_fd','CNN'))):
            ax=fig.add_subplot(gs[i,j]);im=ax.imshow(arrays[name],cmap='RdBu_r',vmin=-vmax,vmax=vmax)
            ax.set_xticks([]);ax.set_yticks([])
            detail=f'응답/잡음 {row["models"]["cnn"]["response_to_noise"]:.1f}' if j==0 else f'G cosine {row["models"]["asm" if j==1 else "cnn"]["cosine_rgb"][1]:.3f}'
            base_label='초기 위상' if base=='initial' else '기존 최적 위상'
            ax.set_title(f'{base_label} · {label}\n{detail}',fontsize=10)
            if j==2:fig.colorbar(im,ax=ax,fraction=.024,label='진폭 변화 / rad')
    ax=fig.add_subplot(gs[2,:2])
    physical=np.array([r['physical_loss_derivative'] for r in gradients])
    for name,color in (('asm','#edbd6a'),('cnn','#79dbc4')):
        values=np.array([r['models'][name]['autodiff_loss_derivative'] for r in gradients])
        ax.errorbar(physical,values,xerr=[r['physical_loss_derivative_repeat_se'] for r in gradients],
                    fmt='o',label=name.upper(),color=color,markersize=4,capsize=2,alpha=.8)
    lo=min(physical.min(),*[r['models'][n]['autodiff_loss_derivative'] for r in gradients for n in ('asm','cnn')]);hi=max(physical.max(),*[r['models'][n]['autodiff_loss_derivative'] for r in gradients for n in ('asm','cnn')])
    ax.plot([lo,hi],[lo,hi],'--',color='#8294a5',lw=1);ax.axhline(0,color='#8294a5',lw=.6);ax.axvline(0,color='#8294a5',lw=.6)
    ax.set(xlabel='실측 loss 방향 미분',ylabel='CITL 모델의 방향 미분',title='실측 loss 미분과 autodiff의 일치 여부');ax.grid(alpha=.15);legend(ax,fontsize=9)
    ax=fig.add_subplot(gs[2,2]);ax.axis('off')
    lines=[]
    for name in ('asm','cnn'):
        early=[r['models'][name]['cosine'] for r in gradients if r['base']=='initial']
        late=[r['models'][name]['cosine'] for r in gradients if r['base']=='old_best']
        lines.append(f'{name.upper()} · 응답 cosine 중앙값\n초기 위상 {np.median(early):.3f}\n기존 최적 위상 {np.median(late):.3f}')
    ax.text(0,1,'\n\n'.join(lines),va='top',fontsize=10,linespacing=1.6)
    fig.text(.035,.025,'2개 위상 × 3개 방향 × 2개 ε의 국소 검사. 가로 오차 막대: 2회 반복의 SE (신뢰구간 아님). 같은 방향의 두 ε는 독립 표본이 아닙니다.',fontsize=10,color='#edbd6a')
    fig.savefig(out/'gradient_montage.png',bbox_inches='tight',pad_inches=.25);plt.close(fig)
    if len(conditions)!=4:
        print('Stability and gradient figures ready; waiting for four learning-rate conditions.')
        return

    # Matched-budget visual comparison: all conditions, no best-case cherry-picking.
    budget=protocol['steps']
    fig=plt.figure(figsize=(17,9.5),dpi=165)
    fig.suptitle(f'학습률 대조 실험 · 동일 초기 위상 / 동일 모델 / 각 {budget}회',x=.025,y=.98,ha='left',fontsize=20,weight='bold')
    fig.text(.025,.936,stamp+'  |  최적·마지막 위상을 각각 3회 재촬영  |  고정 노출·정규화·표시 범위',color='#b7c8d6',fontsize=11)
    gs=fig.add_gridspec(4,5,left=.025,right=.985,top=.89,bottom=.08,height_ratios=[.2,1.65,.2,1.65],hspace=.30,wspace=.045)
    effects=[]
    for i,identity in enumerate(('0846','0898')):
        high=next(r for r in conditions if r['id']==identity and r['lr']==.02)
        low=next(r for r in conditions if r['id']==identity and r['lr']==.005)
        reduction=100*(1-low['remeasured_mean_mse']['last']/high['remeasured_mean_mse']['last'])
        best_reduction=100*(1-low['remeasured_mean_mse']['best']/high['remeasured_mean_mse']['best'])
        effects.append(dict(id=identity,last_mse_reduction_percent=reduction,best_mse_reduction_percent=best_reduction))
        title=fig.add_subplot(gs[2*i,:]);title.axis('off')
        end_text=f'{abs(reduction):.1f}% '+('감소' if reduction>=0 else '증가')
        best_text=f'{abs(best_reduction):.1f}% '+('감소' if best_reduction>=0 else '증가')
        title.text(0,.5,f'{identity}.png · 낮은 학습률: 종료 위상 MSE {end_text} · 최적 위상 MSE {best_text}',
                   fontsize=13,weight='bold',color='#79dbc4',va='center')
        panels=[(root/f'citl_{identity}_lr0.02'/'target.png','목표 영상',None),
                (root/f'citl_{identity}_lr0.02'/'best_remeasured.png',f'lr 0.02 · 최적 step {high["best_step"]}',high['remeasured_mean_mse']['best']),
                (root/f'citl_{identity}_lr0.02'/'last_remeasured.png',f'lr 0.02 · 종료 step {budget}',high['remeasured_mean_mse']['last']),
                (root/f'citl_{identity}_lr0.005'/'best_remeasured.png',f'lr 0.005 · 최적 step {low["best_step"]}',low['remeasured_mean_mse']['best']),
                (root/f'citl_{identity}_lr0.005'/'last_remeasured.png',f'lr 0.005 · 종료 step {budget}',low['remeasured_mean_mse']['last'])]
        for j,(path,label,score) in enumerate(panels):
            ax=fig.add_subplot(gs[2*i+1,j]);image(ax,path,label)
            ax.set_xlabel('DIV2K' if score is None else f'재촬영 평균 진폭 MSE  {score:.5f}',fontsize=9,labelpad=7)
            if j>=3:
                for spine in ax.spines.values():spine.set_color('#79dbc4')
    fig.text(.025,.025,'개선율은 기존 결과가 아닌 이번 실험의 동시대 대조군과 비교. 타깃별 실행 순서 반전. 2개 타깃·1회 최적화이므로 일반화/수렴의 입증은 아님.',fontsize=10,color='#edbd6a')
    fig.savefig(out/'learning_rate_montage.png',bbox_inches='tight',pad_inches=.25);plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(15,9),dpi=165)
    fig.subplots_adjust(left=.065,right=.98,top=.84,bottom=.10,hspace=.45,wspace=.25)
    fig.suptitle('낮은 학습률이 실측 loss와 위상 이동에 미친 영향',x=.035,y=.97,ha='left',fontsize=19,weight='bold')
    fig.text(.035,.915,stamp+'  |  각 조건마다 Adam 초기화 · CNN 가중치 고정 · 고정 기준 위상 재촬영 삽입',fontsize=11,color='#b7c8d6')
    for i,identity in enumerate(('0846','0898')):
        for lr,color in ((.02,'#edbd6a'),(.005,'#79dbc4')):
            h=rows(root/f'citl_{identity}_lr{lr:g}'/'history.jsonl');x=[r['step'] for r in h];y=[r['amplitude_mse'] for r in h]
            axes[i,0].plot(x,y,color=color,label=f'lr {lr:g}',lw=1.5)
            best=int(np.argmin(y));axes[i,0].scatter([x[best]],[y[best]],color=color,s=28)
            axes[i,1].plot(x,[float(np.sqrt(np.mean(np.array(r['phase_delta_rms_rgb'])**2))) for r in h],color=color,label=f'lr {lr:g}',lw=1.5)
        axes[i,0].set(title=identity+'.png · 실제 카메라 오차',xlabel='CITL iteration',ylabel='평균 진폭 MSE')
        axes[i,1].set(title=identity+'.png · 위상 이동량',xlabel='CITL iteration',ylabel='초기 대비 wrapped RMS [rad]')
        for ax in axes[i]:ax.grid(alpha=.15);legend(ax,fontsize=9)
    fig.text(.035,.025,'점은 각 실행의 관측 최저 오차. 느린 위상 이동과 최종 품질은 별도 지표이며, 낮은 학습률만으로 모델 gradient가 정확해지는 것은 아닙니다.',fontsize=10,color='#b7c8d6')
    fig.savefig(out/'learning_rate_curves.png',bbox_inches='tight',pad_inches=.25);plt.close(fig)

    summary=dict(as_of=stamp,preservation=preservation,effects=effects,stability=stability,
        reference_duration_minutes=(reference[-1]['time']-reference[0]['time'])/60,
        reference_intensity_change_percent=(100*(np.array(reference[-1]['mean_intensity_rgb'])/np.array(reference[0]['mean_intensity_rgb'])-1)).tolist(),
        reference_intensity_peak_to_peak_percent=(100*np.ptp(np.array([r['mean_intensity_rgb'] for r in reference]),axis=0)/np.array(reference[0]['mean_intensity_rgb'])).tolist(),
        reference_mse_range=[min(r['amplitude_mse'] for r in reference),max(r['amplitude_mse'] for r in reference)],
        gradient={name:dict(median_cosine=float(np.median([r['models'][name]['cosine'] for r in gradients])),
                    sign_agreement=sum(r['models'][name]['autodiff_sign_agrees'] for r in gradients),count=len(gradients)) for name in ('asm','cnn')},
        minimum_fd_response_to_noise=min(r['models']['cnn']['response_to_noise'] for r in gradients),
        conditions=conditions)
    if (root/'gradient_spatial_scale_check.json').exists():
        summary['gradient_spatial_scale_check']=read(root/'gradient_spatial_scale_check.json')
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('conditions','stability')},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True)
    render(parser.parse_args().out)
