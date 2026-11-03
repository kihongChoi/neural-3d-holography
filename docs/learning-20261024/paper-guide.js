const P=D.paper;
const PM=['dpac','sgd_asm','sgd_cnnpropcnn'];
const PL={dpac:'DPAC',sgd_asm:'SGD-ASM',sgd_cnnpropcnn:'SGD-CNNpropCNN'};
const PC={dpac:'#a34a27',sgd_asm:'#657788',sgd_cnnpropcnn:'#126d6a'};
chart($('paper-train-chart'),['red','green','blue'].map((c,i)=>({color:['#a34a27','#126d6a','#345ba0'][i],points:P.models[c].curve})),{
 xmax:Math.max(...Object.values(P.models).map(m=>m.epoch)),ymax:Math.max(...Object.values(P.models).flatMap(m=>m.curve.map(r=>r[1])))*1.12,
 xlabel:'전체 학습 데이터 반복 횟수 (epoch)',ylabel:'22쌍 검증 진폭 L1 (무차원)'});
function paperBars(){
 const metric=$('paper-metric').value,ssim=metric.includes('ssim');
 const label=ssim?'보정 sRGB SSIM · 클수록 좋음':metric==='fixed_white_mse'?'고정 백색 진폭 MSE · 작을수록 좋음':'배율 보정 진폭 MSE · 작을수록 좋음';
 const e=$('paper-result-chart'),W=940,H=370,L=76,R=24,T=38,B=65,h=H-T-B,w=W-L-R;
 const ymax=ssim?1:Math.max(...P.targets.flatMap(t=>PM.map(m=>t.means[m][metric])))*1.15;e.replaceChildren();
 e.append(node('text',{x:L,y:20},label+' (무차원)'));
 for(let k=0;k<=4;k++){let v=k*ymax/4,yy=T+h*(1-v/ymax);e.append(node('line',{x1:L,x2:W-R,y1:yy,y2:yy,class:'grid'}));e.append(node('text',{x:L-10,y:yy+5,'text-anchor':'end'},fmt(v,3)));}
 const group=w/8;
 P.targets.forEach((t,i)=>{PM.forEach((m,j)=>{const yy=T+h*(1-t.means[m][metric]/ymax);const rect=node('rect',{x:L+i*group+12+j*24,y:yy,width:19,height:T+h-yy,fill:PC[m]});rect.append(node('title',{},t.id+' '+PL[m]+': '+fmt(t.means[m][metric])));e.append(rect);});e.append(node('text',{x:L+(i+.5)*group,y:H-B+24,'text-anchor':'middle'},t.id));});
 e.append(node('text',{x:W/2,y:H-10,'text-anchor':'middle'},'DIV2K 새 평가 영상 ID'));
 $('paper-metric-readout').textContent='8장 × RGB × 재촬영 3회 평균: '+PM.map(m=>PL[m]+' '+fmt(P.means[m][metric])).join(' / ')+'. '+(ssim?'평균 SSIM은 DPAC이 가장 높습니다.':'두 SGD의 초기 위상과 1,000회 반복을 동일하게 맞췄습니다.');
}
$('paper-metric').onchange=paperBars;$('paper-metric-reset').onclick=()=>{$('paper-metric').value='fixed_white_mse';paperBars();};paperBars();
P.targets.forEach(t=>$('paper-target').add(new Option(t.id,t.id)));
function paperImages(){
 const t=P.targets.find(t=>t.id===$('paper-target').value),fitted=$('paper-display').value==='fitted';
 $('paper-images').replaceChildren(fig(t.images.target,t.id+' · 디지털 목표'),...PM.map(m=>fig(t.images[m+(fitted?'_fitted':'')],PL[m]+' · '+(fitted?'배율 보정':'고정 백색')+' · 첫 번째 재촬영')));
 const metric=fitted?'fitted_amplitude_mse':'fixed_white_mse';
 $('paper-image-readout').textContent=t.id+' · 원본 '+t.metadata.source_size_wh.join('×')+' → 리사이즈·중앙 자르기 → 1680×960. 세 재촬영 평균 '+(fitted?'보정':'고정')+' 진폭 MSE: '+PM.map(m=>PL[m]+' '+fmt(t.means[m][metric])).join(' / ')+'.'+(fitted?' CNN 평균 진폭 배율 '+fmt(t.means.sgd_cnnpropcnn.fitted_amplitude_scale,3)+' (색·반복 평균).':'');
}
$('paper-target').onchange=paperImages;$('paper-display').onchange=paperImages;$('paper-target-reset').onclick=()=>{$('paper-target').value='0822';$('paper-display').value='fixed';paperImages();};paperImages();
