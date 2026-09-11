// Optional, explicit downloads. See THIRD_PARTY_NOTICES.md before running.
import {createHash} from 'node:crypto';
import {mkdir,readFile,writeFile,rename,cp} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const args=process.argv.slice(2);
if(!args.length||args.some(a=>!['--camera','--mixamo'].includes(a)))throw Error('Usage: node scripts/setup-optional-assets.mjs --camera [--mixamo]. Review THIRD_PARTY_NOTICES.md first.');
const assets=[];
if(args.includes('--camera')){
  assets.push(
    ['models/pose_landmarker_lite.task','https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task','59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbfb6753c459690d574a'],
    ['models/face_landmarker.task','https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task','64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff'],
    ['models/hand_landmarker.task','https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task','fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1']);
}
if(args.includes('--mixamo'))assets.push(['motions/samba.fbx','https://raw.githubusercontent.com/mrdoob/three.js/r185/examples/models/fbx/Samba%20Dancing.fbx','b9003ee562c87bf03051c3a502411b0808d3513f1d74a2011f7530d9f067069f']);
const digest=data=>createHash('sha256').update(data).digest('hex');
for(const [name,url,expected] of assets){
  const target=path.join(root,'web/public',name);
  let existing;
  try{existing=await readFile(target);}catch(error){if(error.code!=='ENOENT')throw error;}
  if(existing){if(digest(existing)!==expected)throw Error(`Existing file differs; not overwritten: ${name}`);continue;}
  console.log(`Downloading optional ${name}`);
  const response=await fetch(url,{signal:AbortSignal.timeout(120000)});
  if(!response.ok)throw Error(`${name}: HTTP ${response.status}`);
  const data=new Uint8Array(await response.arrayBuffer());
  if(data.byteLength>80*1024*1024||digest(data)!==expected)throw Error(`Unexpected bytes for ${name}`);
  await mkdir(path.dirname(target),{recursive:true});await writeFile(target,data,{flag:'wx'});
}
if(args.includes('--camera'))await cp(path.join(root,'web/node_modules/@mediapipe/tasks-vision/wasm'),path.join(root,'web/public/models/wasm'),{recursive:true,force:false,errorOnExist:false});
console.log('Optional assets are local and git-ignored. Rebuild: npm --prefix web run build');
