(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const D = JSON.parse($('evidence-data').textContent);
  const NS = 'http://www.w3.org/2000/svg';
  const colors = ['#536573', '#b64d20', '#176477'];
  const names = {target:'타깃',asm:'ASM',cnn:'CNN',cnn_citl:'CNN+CITL'};
  const fixed = n => Number(n).toFixed(6);
  $('mode').addEventListener('click', () => {
    const on = document.body.classList.toggle('with-analogies');
    $('mode').setAttribute('aria-pressed',String(on));
    $('mode').textContent = on ? '전문 설명만 보기' : '비유 함께 보기';
  });
  $('expand').addEventListener('click', () => {
    const open = ![...document.querySelectorAll('details')].every(d=>d.open);
    document.querySelectorAll('details').forEach(d=>d.open=open);
    $('expand').textContent = open ? '풀이 모두 접기' : '풀이 모두 펼치기';
  });
  $('print').addEventListener('click',()=>window.print());
  let savedDetails;
  window.addEventListener('beforeprint',()=> {
    savedDetails=[...document.querySelectorAll('details')].map(d=>d.open);
    document.querySelectorAll('details').forEach(d=>d.open=true);
  });
  window.addEventListener('afterprint',()=> {
    document.querySelectorAll('details').forEach((d,i)=>d.open=savedDetails?.[i]??false);
  });
  document.querySelectorAll('.exercise').forEach(box=>{
    box.querySelectorAll('button[data-choice]').forEach(button=>button.addEventListener('click',()=>{
      const correct=button.dataset.choice===box.dataset.answer;
      box.querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));
      const feedback=box.querySelector('.feedback');
      feedback.textContent=correct?box.querySelector('template').content.textContent:'다시 생각해 보세요. 무엇을 측정하고, 어떤 변수를 업데이트하는지 확인해 보세요.';
      feedback.className='feedback '+(correct?'pass':'fail');
    }));
  });
  function el(svg, tag, attrs={}, text=''){
    const node=document.createElementNS(NS,tag);
    for(const [k,v] of Object.entries(attrs))node.setAttribute(k,v);
    if(text)node.textContent=text;
    svg.appendChild(node);return node;
  }
  function prepare(svg,height=286){
    const width=Math.max(250,svg.getBoundingClientRect().width);
    svg.setAttribute('viewBox','0 0 '+width+' '+height);
    svg.style.height=height+'px'; svg.replaceChildren();
    el(svg,'title',{},svg.getAttribute('aria-label'));
    return width;
  }
  function lineChart(id,series,{xLabel='최적화 step',yLabel='진폭 MSE (무차원)',marker=null,yMin=0}={}){
    const svg=$(id);if(!svg)return;
    const w=prepare(svg),h=286,left=60,right=w-18,top=38,bottom=234;
    const all=series.flatMap(s=>s.points);
    const xmax=Math.max(...all.map(p=>p[0]),1);
    const ymax=Math.max(...all.map(p=>p[1]),.0001)*1.08;
    const sx=x=>left+(right-left)*x/xmax,sy=y=>bottom-(bottom-top)*(y-yMin)/(ymax-yMin);
    el(svg,'text',{x:left,y:18},yLabel);
    for(let i=0;i<=4;i++){
      const v=yMin+(ymax-yMin)*i/4,y=sy(v);
      el(svg,'line',{x1:left,y1:y,x2:right,y2:y,class:'gridline'});
      el(svg,'text',{x:left-9,y:y+4,'text-anchor':'end'},ymax>1?v.toFixed(1):v.toFixed(3));
    }
    const ticks=w<450?3:5;
    for(let i=0;i<ticks;i++){
      const v=xmax*i/(ticks-1),x=sx(v);
      el(svg,'text',{x,y:bottom+21,'text-anchor':i===0?'start':i===ticks-1?'end':'middle'},xmax<10?v.toFixed(2):String(Math.round(v)));
    }
    el(svg,'text',{x:(left+right)/2,y:h-9,'text-anchor':'middle'},xLabel);
    el(svg,'line',{x1:left,y1:bottom,x2:right,y2:bottom,class:'axis'});
    for(const s of series){
      const path=s.points.map((p,i)=>(i?'L':'M')+sx(p[0]).toFixed(2)+','+sy(p[1]).toFixed(2)).join(' ');
      const p=el(svg,'path',{d:path,fill:'none',stroke:s.color,'stroke-width':2.2});
      el(p,'title',{},s.name);
    }
    if(marker){
      el(svg,'circle',{cx:sx(marker.x),cy:sy(marker.y),r:5,fill:'#27744c',stroke:'#fff','stroke-width':1.5});
      if(marker.label)el(svg,'text',{x:Math.max(left+4,Math.min(right-135,sx(marker.x)+10)),y:Math.max(top+15,sy(marker.y)-14)},marker.label);
    }
  }
  function bars(item){
    const svg=$('result-bars'),w=prepare(svg,248),left=108,right=w-78,max=Math.max(...Object.values(item.mse))*1.08;
    el(svg,'text',{x:0,y:20},item.id+' · 실제 진폭 MSE (낮을수록 좋음)');
    ['asm','cnn','cnn_citl'].forEach((v,i)=>{
      const y=46+i*48,len=(right-left)*item.mse[v]/max;
      el(svg,'text',{x:left-12,y:y+19,'text-anchor':'end'},names[v]);
      el(svg,'rect',{x:left,y,width:Math.max(1,len),height:28,fill:colors[i]});
      el(svg,'text',{x:left+len+7,y:y+19},fixed(item.mse[v]));
    });
    el(svg,'line',{x1:left,y1:191,x2:right,y2:191,class:'axis'});
    [0,.5,1].forEach(t=>el(svg,'text',{x:left+(right-left)*t,y:211,'text-anchor':t===0?'start':t===1?'end':'middle'},(max*t).toFixed(3)));
    el(svg,'text',{x:(left+right)/2,y:239,'text-anchor':'middle'},'진폭 MSE (무차원)');
  }
  function drawResult(){
    const item=D.items.find(x=>x.id===$('target-select').value);
    bars(item);
    $('image-grid').innerHTML=['target','asm','cnn','cnn_citl'].map(v=>'<figure><img src="'+item.images[v]+'" alt="'+item.id+' '+names[v]+' '+(v==='target'?'목표 영상':'실제 카메라 평균 영상')+'"><figcaption>'+names[v]+(v==='target'?' · 선형 진폭에서 sRGB 표시':' · MSE '+fixed(item.mse[v]))+'</figcaption></figure>').join('');
    lineChart('citl-curve',[{points:item.curve,color:colors[2],name:'카메라 진폭 MSE'}],{
      marker:{x:item.stop.best_step,y:item.stop.best_mse,label:'best: '+item.stop.best_step+' step'}
    });
    const last=item.curve.at(-1)[1];
    $('citl-detail').textContent=item.id+': 최적 기록 '+item.stop.best_step+' step / MSE '+fixed(item.stop.best_mse)+' → 마지막 '+item.stop.steps+' step / MSE '+fixed(last)+'. 종료: '+item.stop.reason+'. 위 막대는 best 위상을 따로 재촬영한 평균이다.';
  }
  function drawDiagnosis(){
    const id=$('diagnosis-select').value;
    const rows=D.diagnosis.filter(x=>x.id===id).sort((a,b)=>b.lr-a.lr);
    lineChart('diagnosis-curve',rows.map((x,i)=>({points:x.curve,color:colors[i+1],name:'lr '+x.lr})));
  }
  function drawInterference(){
    const phi=Number($('phase-slider').value),intensity=2+2*Math.cos(phi);
    $('phase-value').value=phi.toFixed(2);
    lineChart('interference-chart',[{name:'두 파의 합성 강도',color:colors[2],points:Array.from({length:201},(_,i)=>{
      const x=i*2*Math.PI/200;return [x,2+2*Math.cos(x)];
    })}],{xLabel:'상대 위상 Δφ (rad)',yLabel:'합성 강도 I (정규화)',marker:{x:phi,y:intensity}});
    $('interference-result').textContent='Δφ = '+phi.toFixed(2)+' rad → 합성 강도 I = '+intensity.toFixed(3)+'. 개별 강도는 각각 1이지만 간섭 항 때문에 합은 0에서 4까지 변한다.';
  }
  function drawDepth(){
    const depth=Number($('depth-slider').value),planes=[0,5,10,15,20,25,30,35];
    const chosen=planes.reduce((a,b)=>Math.abs(b-depth)<Math.abs(a-depth)?b:a);
    $('depth-value').value=depth.toFixed(1);
    $('mask-board').innerHTML=planes.map(z=>'<span class="'+(z===chosen?'selected':'')+'">'+z+' mm<br>m='+(z===chosen?1:0)+'</span>').join('');
    $('mask-result').textContent='D='+depth.toFixed(1)+' mm인 영역은 '+chosen+' mm 평면에서 m=1이다. 나머지 7개 평면에서는 이 영역의 진폭이 직접 감독되지 않는다.';
  }
  function drawArchitecture(){
    const key=$('architecture-select').value;
    const options={
      paper:{steps:['SLM 복소장','SLM CNN (Re/Im)','ASM → 여러 깊이','공유 target CNN (Re/Im)','예측 진폭'],note:'논문: 복소장 보정 + 물리 전파. 깊이별 영상으로 학습하고, 제외 깊이에서 일반화를 검증한다.'},
      current:{steps:['SLM 위상','고정 ASM → 15 mm','ROI 진폭 1채널','소형 CNN + gain','예측 진폭'],note:'현재: 전파 후 진폭의 곱셈 보정. 전파는 복소 연산이지만 보정 CNN은 위상 채널을 입력받지 않는다.'},
      asm:{steps:['SLM 위상','복소장 exp(iφ)','분석적 ASM','예측 진폭'],note:'ASM 기준: 학습된 보정 없이 설정된 광학 파라미터로 전파한다. 실제 광학계와의 불일치가 남을 수 있다.'}
    };
    $('architecture-flow').innerHTML=options[key].steps.map(x=>'<span>'+x+'</span>').join('<b aria-hidden="true">→</b>');
    $('architecture-note').textContent=options[key].note;
  }
  let redraw;
  if(document.body.dataset.guide==='paper-vs-results'){
    $('final-table').querySelector('tbody').innerHTML=D.items.map(x=>'<tr><td>'+x.id+'</td>'+['asm','cnn','cnn_citl'].map(v=>'<td class="numeric">'+fixed(x.mse[v])+'</td>').join('')+'<td>'+x.stop.best_step+'</td><td>'+(x.stop.reason==='converged'?'plateau · 960':'상한 · 1000')+'</td></tr>').join('')+'<tr><th>8장 평균</th>'+['asm','cnn','cnn_citl'].map(v=>'<th>'+fixed(D.mean[v])+'</th>').join('')+'<td colspan="2">방식별 3회 재촬영</td></tr>';
    $('target-select').innerHTML=D.items.map(x=>'<option value="'+x.id+'">'+x.id+'</option>').join('');
    $('diagnosis-table').querySelector('tbody').innerHTML=[...D.diagnosis].sort((a,b)=>a.id.localeCompare(b.id)||b.lr-a.lr).map(x=>'<tr><td>'+x.id+'</td><td>'+x.lr+'</td><td>'+x.best_step+'</td><td>'+fixed(x.mse.best)+'</td><td>'+fixed(x.mse.last)+'</td></tr>').join('');
    $('target-select').addEventListener('change',drawResult);
    $('diagnosis-select').addEventListener('change',drawDiagnosis);
    redraw=()=>{drawResult();drawDiagnosis();};redraw();
  }else{
    $('phase-slider').addEventListener('input',drawInterference);
    $('phase-zero').addEventListener('click',()=>{$('phase-slider').value=0;drawInterference();});
    $('phase-pi').addEventListener('click',()=>{$('phase-slider').value=Math.PI;drawInterference();});
    $('depth-slider').addEventListener('input',drawDepth);
    $('architecture-select').addEventListener('change',drawArchitecture);
    drawDepth();drawArchitecture();redraw=drawInterference;redraw();
  }
  let pending;
  const resize=new ResizeObserver(()=>{cancelAnimationFrame(pending);pending=requestAnimationFrame(redraw);});
  document.querySelectorAll('svg.chart').forEach(svg=>resize.observe(svg));
})();
