import {createHash} from 'node:crypto';
import {mkdir,readFile,writeFile} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createSyntheticPackage} from '../web/src/synthetic.mjs';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const data=process.argv[2] || process.env.GSAVATAR_DATA_DIR || path.join(root,'local/data');
const folder=path.resolve(root,data,'avatars/synthetic-demo');
try {
  const existing=JSON.parse(await readFile(path.join(folder,'avatar.json'),'utf8'));
  if(existing.generation?.engine!=='synthetic-demo')throw Error('Refusing to overwrite a non-demo avatar');
  console.log('Synthetic demo already exists; no files changed.');
} catch(error) {
  if(error.code!=='ENOENT')throw error;
  await mkdir(folder,{recursive:true});
  const {meta,buffer}=createSyntheticPackage();
  meta.sha256=createHash('sha256').update(new Uint8Array(buffer)).digest('hex');
  // Exclusive writes: never replace an existing user asset or incomplete package.
  await writeFile(path.join(folder,'avatar.bin'),new Uint8Array(buffer),{flag:'wx'});
  await writeFile(path.join(folder,'avatar.json'),JSON.stringify(meta,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify({demo:'synthetic',gaussians:meta.nGaussians,bytes:meta.bytes,folder}));
}
