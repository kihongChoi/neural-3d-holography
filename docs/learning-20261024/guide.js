'use strict';
const D=JSON.parse(document.getElementById('guide-data').textContent);
const $=id=>document.getElementById(id), fmt=(x,n=6)=>Number(x).toFixed(n);
const COLORS={asm:'#657788',cnn:'#b56e2e',cnn_citl:'#126d6a'};
const LABELS={asm:'ASM',cnn:'CNN',cnn_citl:'CNN+CITL · 최적 위상',target:'디지털 목표'};
const svgNS='http://www.w3.org/2000/svg';
function node(name,attrs,text){const n=document.createElementNS(svgNS,name);Object.entries(attrs||{}).forEach(([k,v])=>n.setAttribute(k,v));if(text!==undefined)n.textContent=text;return n;}
function chart(el,series,{xmax,ymax,xlabel,ylabel,ymin=0,point=null,reference=null}){
 const box=el.viewBox.baseVal,W=box.width,H=box.height,L=76,R=25,T=38,B=58,w=W-L-R,h=H-T-B;el.replaceChildren();
 const x=v=>L+v/xmax*w,y=v=>T+h-(v-ymin)/(ymax-ymin)*h;
 el.append(node('text',{x:L,y:21},ylabel));
 for(let k=0;k<=4;k++){let val=ymin+(ymax-ymin)*k/4;el.append(node('line',{x1:L,x2:W-R,y1:y(val),y2:y(val),class:'grid'}));el.append(node('text',{x:L-10,y:y(val)+5,'text-anchor':'end'},fmt(val,ymax<=.3?3:1)));}
 for(let k=0;k<=4;k++){let val=xmax*k/4;el.append(node('text',{x:x(val),y:H-B+24,'text-anchor':'middle'},fmt(val,xmax<10?2:0)));}
 el.append(node('text',{x:L+w/2,y:H-8,'text-anchor':'middle'},xlabel));
 el.append(node('line',{x1:L,x2:L,y1:T,y2:H-B,class:'axis'}));
 if(reference!==null)el.append(node('line',{x1:L,x2:W-R,y1:y(reference),y2:y(reference),stroke:'#a34a27','stroke-dasharray':'7 5'}));
 for(const s of series){let path='',pen=false;for(const [a,b] of s.points){if(!Number.isFinite(a)||!Number.isFinite(b)){pen=false;continue;}path+=(pen?' L':'M')+x(a).toFixed(2)+','+y(b).toFixed(2);pen=true;}el.append(node('path',{d:path,fill:'none',stroke:s.color,'stroke-width':2.5}));}
 if(point)el.append(node('circle',{cx:x(point[0]),cy:y(point[1]),r:6,fill:'#a34a27',stroke:'white','stroke-width':2}));
}
function wave(){
 const delta=Number($('phase').value),r=Number($('ratio').value),intensity=1+r*r+2*r*Math.cos(delta);
 $('phase-value').textContent=fmt(delta,3)+' rad';$('ratio-value').textContent=fmt(r,2);
 chart($('wave-chart'),[{color:'#126d6a',points:Array.from({length:301},(_,k)=>{const q=k/300*2*Math.PI;return[q,1+r*r+2*r*Math.cos(q)];})}],{xmax:2*Math.PI,ymax:4,xlabel:'상대 위상 δ (rad)',ylabel:'첫 파동 대비 광강도 I / I₁ (무차원)',point:[delta,intensity]});
 $('wave-readout').textContent='현재 광강도: 첫 파동만 있을 때의 '+fmt(intensity,3)+'배 · 진폭 비 '+fmt(r,2)+', 상대 위상 '+fmt(delta,3)+' rad.';
}
function wavePreset(delta,r){$('phase').value=delta;$('ratio').value=r;wave();}
['phase','ratio'].forEach(id=>$(id).addEventListener('input',wave));
$('construct').onclick=()=>wavePreset(0,1);$('destruct').onclick=()=>wavePreset(Math.PI,1);$('onewave').onclick=()=>wavePreset(Math.PI/2,0);$('wave-reset').onclick=()=>wavePreset(Math.PI/2,1);wave();
for(const [ch,name] of [['red','R · 638 nm'],['green','G · 520 nm'],['blue','B · 450 nm']]){
 const r=D.prediction[ch],tr=document.createElement('tr');tr.innerHTML='<td>'+name+'</td><td class="number">'+fmt(r.asm)+'</td><td class="number">'+fmt(r.cnn)+'</td><td class="number">'+fmt((1-r.cnn/r.asm)*100,1)+'%</td>';$('pred-table').tBodies[0].append(tr);
}
const tbody=$('final-table').tBodies[0];
for(const r of D.final){
 let tr=document.createElement('tr');tr.innerHTML='<td>'+r.id+'</td>'+['asm','cnn','cnn_citl'].map(k=>'<td class="number">'+fmt(r.means[k])+'</td>').join('')+'<td>'+(r.termination.reason==='iteration_limit'?'1000회 상한':'960회 정체 조건')+'</td><td class="number">'+r.termination.best_step+'</td>';tbody.append(tr);
 $('target-select').add(new Option(r.id+(r.id==='0846'?' · 건물':r.id==='0898'?' · 꽃':''),r.id));
}
let total=document.createElement('tr');total.className='total';total.innerHTML='<td>8장 평균</td>'+['asm','cnn','cnn_citl'].map(k=>'<td class="number">'+fmt(D.means[k])+'</td>').join('')+'<td colspan="2">각 이미지 3회 평균을 동일 가중</td>';tbody.append(total);
function bars(){
 const e=$('final-chart'),W=940,H=370,L=76,R=24,T=38,B=65,ymax=.2,h=H-T-B,w=W-L-R;e.replaceChildren();
 e.append(node('text',{x:L,y:20},'실제 출력 진폭 MSE · 3회 평균 (무차원)'));
 for(let k=0;k<=4;k++){let v=k*.05,yy=T+h*(1-v/ymax);e.append(node('line',{x1:L,x2:W-R,y1:yy,y2:yy,class:'grid'}));e.append(node('text',{x:L-10,y:yy+5,'text-anchor':'end'},fmt(v,2)));}
 const group=w/8;
 D.final.forEach((r,i)=>{for(const [j,m] of ['asm','cnn','cnn_citl'].entries()){let yy=T+h*(1-r.means[m]/ymax);e.append(node('rect',{x:L+i*group+12+j*24,y:yy,width:19,height:T+h-yy,fill:COLORS[m]}));}e.append(node('text',{x:L+(i+.5)*group,y:H-B+24,'text-anchor':'middle'},r.id));});
 e.append(node('text',{x:W/2,y:H-10,'text-anchor':'middle'},'DIV2K 최종 평가 영상 ID'));
}bars();
function fig(src,label){let f=document.createElement('figure'),im=document.createElement('img'),c=document.createElement('figcaption');im.src=src;im.alt=label;im.loading='lazy';c.textContent=label;f.append(im,c);return f;}
function result(){
 const r=D.final.find(x=>x.id===$('target-select').value);
 $('measured-images').replaceChildren(...['target','asm','cnn','cnn_citl'].map(k=>fig(r.images[k],r.id+' · '+LABELS[k]+(k==='target'?'':' · MSE '+fmt(r.means[k])))));
 $('image-context').textContent='출처: DIV2K validation '+r.id+'.png · 원본 '+r.source_size[0]+'×'+r.source_size[1]+' → 종횡비 유지 리사이즈·중앙 crop → 1680×960. 전체 최종 집합은 결과 확인 전에 정했으며 여덟 장 모두 선택 가능합니다. 미리보기는 저장된 평균 진폭의 sRGB 표시를 축소한 영상입니다.';
 let best=r.history.reduce((a,b)=>b[1]<a[1]?b:a),last=r.history[r.history.length-1],ymax=Math.max(...r.history.map(x=>x[1]))*1.12;
 chart($('citl-chart'),[{color:'#126d6a',points:r.history}],{xmax:1000,ymax,xlabel:'CITL 업데이트 횟수 (회)',ylabel:'실측 RGB 진폭 MSE (무차원)',reference:best[1],point:best});
 $('citl-readout').textContent=r.id+' · 시작 MSE '+fmt(r.history[0][1])+' → 관측 최저 '+fmt(best[1])+' ('+best[0]+'회) → 마지막 '+fmt(last[1])+' ('+last[0]+'회). 보존된 최적 위상의 3회 재촬영 평균 MSE '+fmt(r.means.cnn_citl)+'.';
}
$('target-select').onchange=result;$('target-reset').onclick=()=>{$('target-select').value='0846';result();};result();
function diagnosis(){
 const id=$('diag-select').value,rows=D.diagnosis.filter(x=>x.id===id).sort((a,b)=>b.lr-a.lr),colors=['#a34a27','#345ba0'];
 chart($('diag-chart'),rows.map((r,k)=>({color:colors[k],points:r.history})),{xmax:200,ymax:Math.max(...rows.flatMap(r=>r.history.map(p=>p[1])))*1.12,xlabel:'CITL 업데이트 횟수 (회)',ylabel:'실측 RGB 진폭 MSE (무차원)'});
 $('diag-table').tBodies[0].replaceChildren();
 rows.forEach(r=>{let tr=document.createElement('tr');tr.innerHTML='<td>'+id+' · '+r.lr+'</td><td class="number">'+r.best_step+'</td><td class="number">'+fmt(r.remeasured_mean_mse.best)+'</td><td class="number">'+fmt(r.remeasured_mean_mse.last)+'</td>';$('diag-table').tBodies[0].append(tr);});
 const high=rows[0],low=rows[1],last=100*(1-low.remeasured_mean_mse.last/high.remeasured_mean_mse.last),best=100*(1-low.remeasured_mean_mse.best/high.remeasured_mean_mse.best);
 $('diag-readout').textContent=id+' · 낮춘 학습률의 종료 위상 재촬영 MSE는 '+fmt(last,1)+'% 감소. 최적 위상 재촬영 MSE는 '+fmt(Math.abs(best),1)+'% '+(best>=0?'감소':'증가')+'. 네 실험 모두 200회 제한이며 수렴 검증이 아닙니다.';
 $('diag-images').replaceChildren(...rows.flatMap(r=>['best_remeasured','last_remeasured'].map(k=>fig(r.images[k],id+' · 학습률 '+r.lr+' · '+(k==='best_remeasured'?'최적 위상':'종료 위상')+' 재촬영'))));
}
$('diag-select').onchange=diagnosis;$('diag-reset').onclick=()=>{$('diag-select').value='0846';diagnosis();};diagnosis();
const tip=$('tooltip');let activeTip=null,tipTimer;
function closeTip(){tip.hidden=true;if(activeTip){activeTip.setAttribute('aria-expanded','false');activeTip.dataset.pinned='0';}activeTip=null;}
function showTip(b){clearTimeout(tipTimer);if(activeTip&&activeTip!==b)activeTip.setAttribute('aria-expanded','false');activeTip=b;tip.textContent=b.dataset.tip;tip.hidden=false;b.setAttribute('aria-expanded','true');const q=b.getBoundingClientRect();tip.style.left=Math.max(12,Math.min(q.left,innerWidth-tip.offsetWidth-12))+'px';tip.style.top=Math.max(12,Math.min(q.bottom+8,innerHeight-tip.offsetHeight-12))+'px';}
for(const b of document.querySelectorAll('.tip-btn')){
 b.setAttribute('aria-describedby','tooltip');b.addEventListener('mouseenter',()=>showTip(b));b.addEventListener('focus',()=>showTip(b));
 b.addEventListener('mouseleave',()=>{if(b.dataset.pinned!=='1')tipTimer=setTimeout(closeTip,220);});
 b.addEventListener('blur',()=>{tipTimer=setTimeout(closeTip,220);});
 b.addEventListener('click',()=>{if(activeTip===b&&b.dataset.pinned==='1'){closeTip();}else{b.dataset.pinned='1';showTip(b);}});
}
tip.onmouseenter=()=>clearTimeout(tipTimer);tip.onmouseleave=()=>{if(activeTip?.dataset.pinned!=='1')tipTimer=setTimeout(closeTip,220);};
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeTip();});
document.addEventListener('click',e=>{if(!e.target.closest('.tip-btn')&&!tip.contains(e.target))closeTip();});
window.addEventListener('resize',closeTip);
for(const b of document.querySelectorAll('[data-answer]'))b.onclick=()=>{$('quiz-answer').textContent=b.dataset.answer==='yes'?'맞습니다. 전방 예측은 고정 위상의 영상값을, 새 위상 생성은 모델의 변화 방향까지 이용합니다. 실제 출력 평가가 따로 필요합니다.':'다시 생각해 보세요. 같은 위치에서 함수값이 같아도 기울기는 반대일 수 있습니다. 02 단원의 1변수 반례를 펼쳐 보세요.';};
document.querySelectorAll('details').forEach(d=>d.addEventListener('toggle',()=>{if(d.open&&window.MathJax?.typesetPromise)MathJax.typesetPromise([d]).catch(mathFailed);}));
setTimeout(()=>{if(!document.querySelector('mjx-container'))mathFailed();},20000);
