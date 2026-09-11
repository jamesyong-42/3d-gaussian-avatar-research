import {copyPose,forwardKinematics,axisQuat,quatAxis,quatMul,quatMatrix} from './engine.mjs';

export function applyClip(base,clip,time){
  const p=copyPose(base),a=p.angles,s=Math.sin(time*3.5);
  if(clip==='wave'){a[16*3+2]=-1.25;a[17*3+2]=-0.2;a[19*3+2]=-1.45+s*0.25;a[21*3+1]=s*0.4;a[15*3+1]=s*0.1;}
  if(clip==='walk'){a[1*3]=s*0.5;a[2*3]=-s*0.5;a[4*3]=Math.max(0,-s)*0.8;a[5*3]=Math.max(0,s)*0.8;a[16*3]=s*0.3;a[17*3]=-s*0.3;p.rootPosition[2]=base.rootPosition[2]+Math.sin(time*0.5)*0.7;}
  if(clip==='squat'){const t=(Math.sin(time*2)+1)/2;a[3]=a[6]=-t*0.75;a[12]=a[15]=t*1.45;a[9]=t*0.2;p.rootPosition[1]=base.rootPosition[1]-t*0.25;}
  if(clip==='turn'){a[1]=Math.sin(time)*1.1;a[15*3+1]=Math.sin(time*1.3)*0.3;}
  return p;
}
const sub=(a,b)=>a.map((x,i)=>x-b[i]);
function rotationBetween(a,b){
  const al=Math.hypot(...a),bl=Math.hypot(...b);if(al<1e-6||bl<1e-6)return [0,0,0,1];a=a.map(x=>x/al);b=b.map(x=>x/bl);
  const dot=a.reduce((s,x,i)=>s+x*b[i],0);
  if(dot < -0.9999){let axis=Math.abs(a[0])<0.8?[0,a[2],-a[1]]:[-a[2],0,a[0]];const n=Math.hypot(...axis);return [...axis.map(x=>x/n),0];}
  const q=[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0],1+dot],n=Math.hypot(...q);return q.map(x=>x/n);
}

// Targets are world-space joint anchors, not raw device origins. A future device
// adapter must apply user calibration and headset/controller-to-joint offsets.
export function solveTargets(asset,base,targets){
  const p=copyPose(base),root=p.rootPosition;p.rootPosition=[0,0,0];p.rootRotation=[0,0,0,1];
  const invRoot=[-base.rootRotation[0],-base.rootRotation[1],-base.rootRotation[2],base.rootRotation[3]],matrix=quatMatrix(invRoot);
  const toActor=target=>{const v=sub(target,root);return [0,1,2].map(r=>matrix[r*3]*v[0]+matrix[r*3+1]*v[1]+matrix[r*3+2]*v[2]);};
  // Solve the torso before arms, since moving the spine moves both shoulders.
  for(const [end,chain,worldTarget] of [[15,[12,9,6],targets.head],[20,[18,16,13],targets.left],[21,[19,17,14],targets.right]]){
    if(!worldTarget)continue;const target=toActor(worldTarget);
    for(let iteration=0;iteration<12;iteration++)for(const joint of chain){
      const fk=forwardKinematics(asset,p),j=Array.from(fk.positions.slice(joint*3,joint*3+3)),e=Array.from(fk.positions.slice(end*3,end*3+3));
      const parent=asset.arrays.parents[joint],m=parent<0?null:fk.global.subarray(parent*12,parent*12+12);
      const local=v=>m?[m[0]*v[0]+m[4]*v[1]+m[8]*v[2],m[1]*v[0]+m[5]*v[1]+m[9]*v[2],m[2]*v[0]+m[6]*v[1]+m[10]*v[2]]:v;
      const q=rotationBetween(local(sub(e,j)),local(sub(target,j))),old=axisQuat(...p.angles.slice(joint*3,joint*3+3));
      let aa=quatAxis(quatMul(q,old)),length=Math.hypot(...aa),limit=joint===18||joint===19?2.5:1.8;
      if(length>limit)aa=aa.map(x=>x*limit/length);p.angles.set(aa,joint*3);
    }
  }
  const orient=(joint,rotation)=>{
    if(!rotation)return;
    const parent=asset.arrays.parents[joint],lineage=[];let j=parent;
    while(j>=0){lineage.unshift(j);j=asset.arrays.parents[j];}
    let parentQ=[0,0,0,1];for(const ancestor of lineage)parentQ=quatMul(parentQ,axisQuat(...p.angles.slice(ancestor*3,ancestor*3+3)));
    const inverse=[-parentQ[0],-parentQ[1],-parentQ[2],parentQ[3]],actorQ=quatMul(invRoot,axisQuat(...rotation));
    p.angles.set(quatAxis(quatMul(inverse,actorQ)),joint*3);
  };
  orient(15,targets.headRotation);
  for(const [side,start,wrist] of [['left',25,20],['right',40,21]]){
    const curl=targets[`${side}Grip`]??0;
    for(let j=start;j<start+15;j++)p.angles[j*3+2]=(side==='left'?1:-1)*curl*0.9;
    orient(wrist,targets[`${side}Rotation`]);
  }
  p.rootPosition=root;p.rootRotation=base.rootRotation;
  return p;
}
