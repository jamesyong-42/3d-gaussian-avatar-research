import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {startServer,validateFolder} from './lab_server.mjs';
if(process.argv.length!==3)throw Error('Supply the native deformation export directory');
const folder=validateFolder(process.argv[2]),here=path.dirname(fileURLToPath(import.meta.url));
const lab=await startServer(folder,fs.readFileSync(path.join(here,'deformation_demo.html'),'utf8'));
console.log(lab.url+' — local IDOL deformation lab; Ctrl+C to stop');
process.once('SIGINT',()=>lab.server.close());process.once('SIGTERM',()=>lab.server.close());
