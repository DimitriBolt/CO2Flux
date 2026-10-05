/* Direct Fourier engine shared by both HTML views; tested against Python. */
'use strict';
function wrapPhase(phi) { return ((phi + Math.PI) % (2*Math.PI) + 2*Math.PI) % (2*Math.PI) - Math.PI; }
function calculatePhases(data, windowDays=30, stepDays=1) {
  if (![windowDays,stepDays].every(x=>Number.isInteger(x) && x>0))
    throw Error('Длина и шаг должны быть положительными целыми числами суток');
  const width=windowDays*12, step=stepDays*12, total=data.t.length;
  if(width>total) throw Error('INSUFFICIENT DATA: окно длиннее кандидатного календаря');
  const results=[];
  for(let lo=0; lo+width<=total; lo+=step) {
    const hi=lo+width, used=[], counts=[0,0,0,0], usedCounts=[0,0,0,0], qChannel=[0,0,0,0];
    let gap=0, longestGap=0, doubtful=0, doubtfulAll=0;
    for(let i=lo;i<hi;i++) {
      const complete=data.y.every(a=>a[i]!==null && Number.isFinite(a[i]));
      const review=data.q.some(a=>a[i]);
      if(review) doubtfulAll++;
      for(let j=0;j<4;j++) counts[j]+=data.n[j][i];
      if(complete) {
        used.push(i); gap=0;
        if(review) doubtful++;
        for(let j=0;j<4;j++) {usedCounts[j]+=data.n[j][i]; if(data.q[j][i]) qChannel[j]++;}
      } else {gap++; longestGap=Math.max(longestGap,gap);}
    }
    const n=used.length, F=Array.from({length:4},()=>[null,null]), phase=[null,null,null], pairReasons=['','',''];
    let reason='';
    if(n<2) reason='INSUFFICIENT DATA: менее двух общих ячеек; после центрирования нет ненулевой фазы';
    else {
      for(let j=0;j<4;j++) {
        let mean=0, re=0, im=0;
        for(const i of used) mean+=data.y[j][i];
        mean/=n;
        for(const i of used) {
          const angle=-2*Math.PI*data.t[i], centered=data.y[j][i]-mean;
          re+=centered*Math.cos(angle); im+=centered*Math.sin(angle);
        }
        F[j]=[re/n,im/n];
      }
      for(let j=0;j<3;j++) {
        const [ar,ai]=F[j], [br,bi]=F[j+1];
        if(Math.hypot(ar,ai)===0 || Math.hypot(br,bi)===0) {
          reason='INSUFFICIENT DATA: нулевой коэффициент хотя бы одного канала; часть фаз не определена';
          pairReasons[j]='Нулевой Fourier-коэффициент; аргумент не определён';
        } else phase[j]=wrapPhase(Math.atan2(ai*br-ar*bi,ar*br+ai*bi));
      }
    }
    if(n<2) pairReasons.fill(reason);
    const date=days=>new Date(Date.parse(data.start+'T00:00:00Z')+days*86400000).toISOString().slice(0,19);
    results.push({window:results.length,start:date(lo/12),end:date(hi/12),center:date((lo+hi)/24),
      possible:width,n,coverage:100*n/width,longestGapHours:2*longestGap,doubtful,doubtfulAll,
      counts,usedCounts,qChannel,F,phase,pairReasons,reason,
      status:reason?'INSUFFICIENT DATA':(doubtful?'ВЫЧИСЛЕНО / ТРЕБУЕТ ПРОВЕРКИ':'ВЫЧИСЛЕНО / НАУЧНАЯ НАДЁЖНОСТЬ НЕ ОЦЕНЕНА')});
  }
  return results;
}
function phaseLine(rows,pair) {
  const x=[], y=[], custom=[]; let prev=null;
  for(const r of rows) {
    const value=r.phase[pair];
    if(prev!==null && value!==null && Math.abs(value-prev)>Math.PI) {x.push(r.center); y.push(null); custom.push(null);}
    x.push(r.center); y.push(value); custom.push(r); prev=value;
  }
  return {x,y,custom};
}
if(typeof module!=='undefined') module.exports={calculatePhases,wrapPhase,phaseLine};
