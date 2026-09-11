import crypto from 'node:crypto';
import {createGateway} from './public_gateway.mjs';

if(process.argv.length!==3)throw Error('Usage: node scripts/run_public_gateway.mjs <exact-device-name.ts.net>');
const publicHost=process.argv[2],password=crypto.randomBytes(24).toString('base64url');
const gateway=createGateway({publicHost,password}),health=await fetch('http://127.0.0.1:8765/api/health');
if(!health.ok||(await health.json()).ok!==true)throw Error('Start the local prototype on port 8765 first');
await gateway.listen(8766);
// Deliberate one-time handoff to the owner. Never put credentials in URLs,
// source, a disk file, access logs, or upstream requests. Restart rotates them.
console.log(JSON.stringify({gateway:'http://127.0.0.1:8766',publicUrl:gateway.publicOrigin,username:gateway.username,password,notice:'Gateway ready, not yet published. Ctrl+C stops it; restart generates a new password.'}));
let stopping=false;
async function stop(){if(stopping)return;stopping=true;await gateway.close();process.exit(0);}
process.once('SIGINT',stop);process.once('SIGTERM',stop);

// Diagnostic commands over this owned process's stdin, without putting the
// generated password on another process's command line or saving it to disk.
let pending='';
process.stdin.setEncoding('utf8');process.stdin.on('data',async chunk=>{
  pending+=chunk.replace(/\r\n?/g,'\n');
  while(pending.includes('\n')){
    const at=pending.indexOf('\n'),command=pending.slice(0,at).trim();pending=pending.slice(at+1);
    if(command==='check-local'||command==='check-public'){
      try{const {verifyGateway}=await import('./verify_public_gateway.mjs?check='+Date.now());await verifyGateway({baseUrl:command==='check-public'?gateway.publicOrigin:'http://127.0.0.1:8766',publicOrigin:gateway.publicOrigin,username:gateway.username,password,publicCheck:command==='check-public'});}
      catch(error){console.error('GATEWAY_CHECK_FAILED '+String(error));}
    }else if(command==='stop')await stop();
  }
});
