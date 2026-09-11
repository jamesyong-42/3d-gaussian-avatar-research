import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import {fileURLToPath} from 'node:url';
const here=path.dirname(fileURLToPath(import.meta.url)),reports=path.resolve(here,'../reports/idol');
export function validateFolder(value){
  const folder=fs.realpathSync(path.resolve(value));
  if(path.dirname(folder)!==reports||!/^deformation-\d{8}-\d{6}-[0-9a-f]{6}$/.test(path.basename(folder)))throw Error('Expected a scoped deformation run');
  return folder;
}
export async function startServer(folder,html='<!doctype html><title>IDOL deformation lab</title><h1>IDOL deformation lab</h1>'){
  const read=n=>JSON.parse(fs.readFileSync(path.join(folder,n),'utf8')),validation=read('validation.json'),record=read('metrics.json');
  if(record.status!=='native-deformation-exported'||read('run.json').exitCode!==0)throw Error('Native export did not pass');
  const files=new Map(['rig.json','correctives.json','correctives.bin','validation.json',...record.assets.flatMap((a,i)=>['asset-'+i+'.json','asset-'+i+'.bin']),...validation.cases.map(c=>c.file)].map(name=>[name,path.join(folder,name)]));
  for(const name of ['corrective_gpu.mjs','deformation_cpu.mjs','deformation_gpu.mjs','neighbors_gpu.mjs','lab_io.mjs','deformation_check.mjs','deformation_demo.mjs','covariance_viewer.mjs','resident_viewer.mjs'])files.set(name,path.join(here,name));
  const server=http.createServer((req,res)=>{
    try{
      if(req.method!=='GET'){res.writeHead(405);res.end();return;}
      const key=decodeURIComponent(new URL(req.url,'http://localhost').pathname).slice(1);
      res.setHeader('Cache-Control','no-store');res.setHeader('Vary','Range');
      if(key===''){res.setHeader('Content-Type','text/html; charset=utf-8');res.end(html);return;}
      const target=files.get(key);if(!target){res.writeHead(404);res.end();return;}
      const size=fs.statSync(target).size;res.setHeader('Content-Type',key.endsWith('.mjs')?'text/javascript':key.endsWith('.json')?'application/json':'application/octet-stream');
      const range=/^bytes=(\d+)-(\d+)$/.exec(req.headers.range??'');
      if(range){
        const begin=Number(range[1]),end=Number(range[2]);
        if(!Number.isSafeInteger(begin)||!Number.isSafeInteger(end)||begin>end||end>=size||end-begin+1>4*1024**2){res.writeHead(416);res.end();return;}
        const bytes=Buffer.alloc(end-begin+1),fd=fs.openSync(target,'r');
        try{if(fs.readSync(fd,bytes,0,bytes.length,begin)!==bytes.length)throw Error('Short read');}finally{fs.closeSync(fd);}
        res.writeHead(206,{'Content-Range':`bytes ${begin}-${end}/${size}`,'Content-Length':bytes.length});res.end(bytes);
      }else{
        if(size>4*1024**2){res.writeHead(413);res.end('Use bounded Range requests');return;}
        const bytes=fs.readFileSync(target);res.setHeader('Content-Length',bytes.length);res.end(bytes);
      }
    }catch(error){if(!res.headersSent)res.writeHead(400);res.end(String(error));}
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  return{server,url:'http://127.0.0.1:'+server.address().port,record,validation,files};
}
