import fs from 'node:fs';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {decodeAsset,attachValidation,checkReferences} from '../web/src/engine.mjs';
const folder=process.argv[2];
const meta=JSON.parse(fs.readFileSync(path.join(folder,process.argv.includes('--pending')?'avatar.pending.json':'avatar.json')));
const bytes=fs.readFileSync(path.join(folder,'avatar.bin'));
const checksum=createHash('sha256').update(bytes).digest('hex')===meta.sha256;
const asset=decodeAsset(meta,bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength));
let validationChecksum=true;
if(meta.validation){const b=fs.readFileSync(path.join(folder,'validation.bin'));validationChecksum=createHash('sha256').update(b).digest('hex')===meta.validation.sha256;attachValidation(asset,b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));}
const checks=checkReferences(asset);
const finite=Object.values(asset.arrays).every(array=>array.every(Number.isFinite));
const passed=checksum&&validationChecksum&&finite&&checks.length===7&&checks.every(c=>c.pass);
console.log(JSON.stringify({pass:passed,checksum,validationChecksum,finite,model:meta.model,bytes:meta.bytes,nGaussians:meta.nGaussians,checks}));
process.exitCode=passed?0:1;
