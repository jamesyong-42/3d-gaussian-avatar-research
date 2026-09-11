import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {execFileSync} from 'node:child_process';
import {chromium} from '../../web/node_modules/playwright-core/index.mjs';
import {validateFolder,startServer} from './lab_server.mjs';
if(process.argv.length!==3)throw Error('Supply the native deformation export directory');
const folder=validateFolder(process.argv[2]),here=path.dirname(fileURLToPath(import.meta.url));
if(execFileSync('C:/Program Files/Docker/Docker/resources/bin/docker.exe',['ps','--filter','label=research.task=idol','--format','{{.Names}}'],{encoding:'utf8',windowsHide:true}).trim())throw Error('Native IDOL generation is still running');
const evidence={date:new Date().toISOString(),nativeRun:path.basename(folder),errors:[],cases:[],
  scope:'Complete geometry parity; not a main-app backend or renderer-parity claim',
  sources:Object.fromEntries(['deformation_cpu.mjs','deformation_gpu.mjs','neighbors_gpu.mjs','corrective_gpu.mjs','deformation_check.mjs','lab_io.mjs','lab_server.mjs','check_deformation.mjs'].map(n=>[n,crypto.createHash('sha256').update(fs.readFileSync(path.join(here,n))).digest('hex')]))};
const destination=path.join(folder,'browser-'+evidence.date.replace(/[:.]/g,'-')+'.json');
const server=await startServer(folder);let browser;
const save=()=>fs.writeFileSync(destination,JSON.stringify(evidence,null,2));
try{
  browser=await chromium.launch({executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE,args:['--disable-background-timer-throttling','--disable-renderer-backgrounding']});
  const page=await browser.newPage();page.on('pageerror',error=>evidence.errors.push(String(error)));
  await page.exposeFunction('reportProgress',async progress=>{
    if(progress.case){evidence.cases.push(progress.case);if(evidence.cases.length%10===1||!progress.case.pass)console.log(JSON.stringify(progress.case));}
    else{Object.assign(evidence,progress);console.log(JSON.stringify(progress));}save();
  });
  await page.goto(server.url);
  const result=await page.evaluate(async()=>{const {runChecks}=await import('/deformation_check.mjs');return runChecks(window.reportProgress);});
  Object.assign(evidence,result);evidence.pass=result.pass&&!evidence.errors.length;
  console.log(JSON.stringify({pass:evidence.pass,cases:evidence.cases.length,benchmark:{...evidence.benchmark,samples:undefined},adapter:evidence.adapter}));
}catch(error){evidence.pass=false;evidence.errors.push(String(error));console.error(String(error));}
finally{save();if(browser)await browser.close();await new Promise(resolve=>server.server.close(resolve));console.log(destination);}
if(!evidence.pass)process.exitCode=1;
