// Invoked inside the owning gateway process: the password never enters a
// subprocess command line, report, screenshot or browser storage-state file.
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import https from 'node:https';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
import {chromium} from '../node_modules/playwright-core/index.mjs';

export async function verifyGateway({baseUrl,publicOrigin,username,password,publicCheck}){
  const output=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../artifacts/public-access');fs.mkdirSync(output,{recursive:true});
  const record={date:new Date().toISOString(),baseUrl,publicCheck,checks:[],errors:[],generationTriggered:false};
  const authorization='Basic '+Buffer.from(username+':'+password).toString('base64');
  const checked=(name,actual,expected)=>{record.checks.push({name,actual,expected,pass:actual===expected});assert.equal(actual,expected,name);};
  let browser;
  const get=(url,options={})=>fetch(url,{...options,signal:AbortSignal.timeout(30000)});
  try{
    for(const route of ['/','/api/health','/api/avatars','/api/avatars/example/avatar.json','/api/avatars/example/data','/api/avatars/example/source-photos','/api/avatars/example/source-photos/0'])checked('unauthenticated '+route,(await get(baseUrl+route)).status,401);
    checked('wrong password',(await get(baseUrl+'/',{headers:{authorization:'Basic '+Buffer.from(username+':wrong').toString('base64')}})).status,401);
    checked('unauthenticated upload',(await get(baseUrl+'/api/jobs',{method:'POST',body:'blocked before upload parsing'})).status,401);
    const health=await get(baseUrl+'/api/health',{headers:{authorization}});checked('authenticated health',health.status,200);checked('backend healthy',(await health.json()).ok,true);
    const sourceCatalog=await get(baseUrl+'/api/avatars/example/source-photos',{headers:{authorization}});checked('authenticated source photo list',sourceCatalog.status,200);
    const sourcePhoto=(await sourceCatalog.json()).photos[0];checked('saved source photo available',sourcePhoto.available,true);
    const sourceImage=await get(baseUrl+sourcePhoto.url,{headers:{authorization}});checked('authenticated source photo bytes',sourceImage.status,200);checked('source photo MIME',sourceImage.headers.get('content-type'),'image/png');
    checked('source photo is private',sourceImage.headers.get('cache-control'),'private, no-store');await sourceImage.arrayBuffer();
    checked('cross-origin write',(await get(baseUrl+'/api/jobs',{method:'POST',headers:{authorization,origin:'https://invalid.example'},body:'blocked'})).status,403);
    checked('same-origin invalid upload reaches validation',(await get(baseUrl+'/api/jobs',{method:'POST',headers:{authorization,origin:publicOrigin},body:new FormData()})).status,400);
    const range=await get(baseUrl+'/api/avatars/example/data',{headers:{authorization,range:'bytes=0-63'}});
    checked('authenticated binary response',[200,206].includes(range.status),true);
    record.backendSupportsRange=range.status===206;
    // This installed backend may return the whole file instead of honoring
    // Range. Confirm real protected bytes, then cancel, without buffering 61 MB.
    const reader=range.body.getReader(),first=await reader.read();checked('protected binary has data',!first.done&&first.value.byteLength>0,true);await reader.cancel();
    const deniedUpgrade=await new Promise((resolve,reject)=>{
      const url=new URL(baseUrl+'/ws/unauth-check?role=receiver'),client=url.protocol==='https:'?https:http;
      const req=client.request(url,{headers:{origin:publicOrigin,connection:'Upgrade',upgrade:'websocket','sec-websocket-key':crypto.randomBytes(16).toString('base64'),'sec-websocket-version':'13'}},res=>{res.resume();resolve(res.statusCode);});
      req.on('upgrade',(_res,socket)=>{socket.destroy();resolve(101);});req.on('error',reject);req.setTimeout(15000,()=>req.destroy(new Error('Upgrade check timed out')));req.end();
    });checked('unauthenticated WebSocket',deniedUpgrade,401);
    if(publicCheck){
      browser=await chromium.launch({executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE,args:['--disable-background-timer-throttling','--disable-renderer-backgrounding']});
      const context=await browser.newContext({httpCredentials:{username,password,origin:publicOrigin},viewport:{width:1440,height:1000}});
      // No password is put in a URL; external fonts cannot receive credentials.
      const page=await context.newPage();page.on('pageerror',error=>record.errors.push(String(error)));
      await page.goto(baseUrl,{waitUntil:'domcontentloaded',timeout:60000});
      await page.waitForFunction(()=>window.__lab?.state.loaded&&window.__lab.state.renderedFrames>5,null,{timeout:180000});
      record.avatar=await page.evaluate(()=>({id:window.__lab.state.assetId,loaded:window.__lab.state.loaded,frames:window.__lab.state.renderedFrames,rawReadbackPixels:window.__lab.visiblePixels()}));
      checked('saved example loaded',record.avatar.id,'example');
      await page.waitForFunction(()=>document.getElementById('photo-label').textContent.includes('saved source photo')&&!document.getElementById('photo-preview').hidden&&document.getElementById('photo-preview').naturalWidth>0,null,{timeout:30000});
      checked('saved source photo visible after HTTPS login',true,true);
      // Use the existing research harness's composited-image negative control.
      // Raw WebGL readPixels can see a discarded buffer after presentation.
      await page.evaluate(()=>window.__lab.setClip('none'));await page.waitForTimeout(500);
      const rendered=await page.locator('#viewport').screenshot();let hidden;
      try{await page.evaluate(()=>window.__lab.setAvatarVisible(false));await page.waitForTimeout(250);hidden=await page.locator('#viewport').screenshot();}
      finally{await page.evaluate(()=>window.__lab.setAvatarVisible(true));}
      record.avatar.visiblePixels=await page.evaluate(async images=>{
        const decoded=await Promise.all(images.map(async data=>{
          const bitmap=await createImageBitmap(await(await fetch('data:image/png;base64,'+data)).blob());
          const canvas=new OffscreenCanvas(bitmap.width,bitmap.height),ctx=canvas.getContext('2d');ctx.drawImage(bitmap,0,0);
          const pixels=ctx.getImageData(0,0,bitmap.width,bitmap.height);bitmap.close();return pixels;
        }));
        const[a,b]=decoded;let count=0;
        if(a.width!==b.width||a.height!==b.height)throw Error('Screenshot size changed');
        for(let y=Math.floor(a.height*.15);y<a.height*.95;y++)for(let x=Math.floor(a.width*.1);x<a.width*.9;x++){
          const i=(y*a.width+x)*4;if(Math.abs(a.data[i]-b.data[i])+Math.abs(a.data[i+1]-b.data[i+1])+Math.abs(a.data[i+2]-b.data[i+2])>30)count++;
        }return count;
      },[rendered.toString('base64'),hidden.toString('base64')]);
      record.avatar.visibilityMethod='Composited screenshot against avatar-hidden negative control';
      checked('visible splats',record.avatar.visiblePixels>1000,true);
      await page.locator('[data-clip="wave"]').click();const before=await page.evaluate(()=>JSON.stringify(window.__lab.state.probe));
      await page.waitForFunction(value=>JSON.stringify(window.__lab.state.probe)!==value,before,{timeout:15000});checked('animated browser geometry',true,true);
      const room='public-check-'+crypto.randomBytes(6).toString('hex');await page.evaluate(id=>window.__lab.connectRoom(id),room);
      await page.waitForFunction(()=>document.getElementById('connection').textContent==='Sending',null,{timeout:15000});
      const received=await page.evaluate(async({baseUrl,room})=>new Promise((resolve,reject)=>{
        const ws=new WebSocket(baseUrl.replace(/^http/,'ws')+'/ws/'+room+'?role=receiver');
        const timer=setTimeout(()=>{ws.close();reject(Error('No public WebSocket pose received'));},15000);
        ws.onmessage=event=>{if(typeof event.data!=='string'){clearTimeout(timer);ws.close();resolve(true);}};
        ws.onerror=()=>{clearTimeout(timer);reject(Error('WebSocket failed'));};
      }),{baseUrl,room});checked('authenticated WSS pose relay',received,true);
      await page.screenshot({path:path.join(output,'public-prototype.png'),fullPage:true});
      record.browser='Chromium: public HTTPS login, example avatar, visible animation and WSS pose relay';
      record.sessionCookie=await context.cookies().then(c=>c.filter(x=>x.name==='__Host-AvatarLabSession').map(({name,secure,httpOnly,sameSite})=>({name,secure,httpOnly,sameSite})));
    }
    record.pass=record.checks.every(c=>c.pass)&&!record.errors.length;
  }catch(error){record.pass=false;record.errors.push(String(error));if(error.cause)record.cause={message:error.cause.message,code:error.cause.code};}
  finally{if(browser)await browser.close();const filename=(publicCheck?'public':'local')+'-'+record.date.replace(/[:.]/g,'-')+'.json';fs.writeFileSync(path.join(output,filename),JSON.stringify(record,null,2));console.log('GATEWAY_CHECK '+JSON.stringify(record));}
  if(!record.pass)throw Error('Gateway checks did not all pass');
}
