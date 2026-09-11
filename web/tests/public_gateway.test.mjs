import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import crypto from 'node:crypto';
import {createGateway} from '../scripts/public_gateway.mjs';

const password='unit-test-only-strong-password',publicHost='gateway.example.ts.net',origin='https://'+publicHost;
const authorization='Basic '+Buffer.from('avatar:'+password).toString('base64');
async function fixture(t){
  const requests=[],sockets=new Set();
  const upstream=http.createServer((req,res)=>{let body='';req.on('data',chunk=>body+=chunk);req.on('end',()=>{requests.push({url:req.url,headers:req.headers,body});res.setHeader('content-type','application/json');res.end(JSON.stringify({ok:true,bytes:body.length}));});});
  upstream.on('connection',socket=>{sockets.add(socket);socket.on('close',()=>sockets.delete(socket));});
  upstream.on('upgrade',(req,socket)=>{
    requests.push({url:req.url,headers:req.headers,upgrade:true});
    const accept=crypto.createHash('sha1').update(req.headers['sec-websocket-key']+'258EAFA5-E914-47DA-95CA-C5AB0DC85B11').digest('base64');
    socket.write(`HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: ${accept}\r\n\r\n`);
  });
  await new Promise(resolve=>upstream.listen(0,'127.0.0.1',resolve));
  const gateway=createGateway({publicHost,password,upstreamPort:upstream.address().port,maxUploadBytes:1024});
  await gateway.listen(0);
  t.after(async()=>{await gateway.close();for(const socket of sockets)socket.destroy();await new Promise(resolve=>upstream.close(resolve));});
  const request=(path='/',headers={},method='GET',body)=>new Promise((resolve,reject)=>{
    const req=http.request({host:'127.0.0.1',port:gateway.server.address().port,path,method,headers},res=>{let text='';res.on('data',chunk=>text+=chunk);res.on('end',()=>resolve({status:res.statusCode,headers:res.headers,body:text}));});
    req.on('error',reject);req.end(body);
  });
  const upgrade=headers=>new Promise((resolve,reject)=>{
    const req=http.request({host:'127.0.0.1',port:gateway.server.address().port,path:'/ws/gateway-test?role=sender',headers:{connection:'Upgrade',upgrade:'websocket','sec-websocket-key':'dGhlIHNhbXBsZSBub25jZQ==','sec-websocket-version':'13',...headers}});
    req.on('upgrade',(res,socket)=>{socket.destroy();resolve(res.statusCode);});req.on('response',res=>{res.resume();resolve(res.statusCode);});req.on('error',reject);req.end();
  });
  return{gateway,request,upgrade,requests};
}
test('all HTTP routes and WebSocket upgrades require authentication before upstream access',async t=>{
  const f=await fixture(t);
  for(const path of ['/','/api/health','/api/avatars','/api/avatars/example/data','/api/avatars/example/source-photos','/api/avatars/example/source-photos/0','/models/pose_landmarker_lite.task'])assert.equal((await f.request(path)).status,401);
  assert.equal((await f.request('/api/jobs',{},'POST','private photo')).status,401);
  assert.equal((await f.request('/',{},'OPTIONS')).status,401);
  assert.equal((await f.request('/',{authorization:'Basic '+Buffer.from('avatar:wrong').toString('base64')})).status,401);
  assert.equal(await f.upgrade({origin}),401);assert.equal(f.requests.length,0);
});
test('valid login issues protected session cookie; credentials and forwarded identity never reach upstream',async t=>{
  const f=await fixture(t),res=await f.request('/',{authorization,'x-forwarded-host':'evil.invalid','x-forwarded-for':'1.2.3.4',forwarded:'host=evil.invalid','tailscale-user-login':'spoof'});
  assert.equal(res.status,200);assert.match(res.headers['set-cookie'][0],/Secure; HttpOnly; SameSite=Strict/);
  assert.equal(res.headers['cache-control'],'private, no-store');
  const h=f.requests[0].headers;assert.equal(h.authorization,undefined);assert.equal(h.cookie,undefined);assert.equal(h['x-forwarded-host'],undefined);assert.equal(h.forwarded,undefined);assert.equal(h['tailscale-user-login'],undefined);
  assert.match(h.host,/^127\.0\.0\.1:\d+$/);
  const cookie=res.headers['set-cookie'][0].split(';')[0];assert.equal((await f.request('/api/avatars',{cookie})).status,200);
  assert.equal((await f.request('/',{cookie:cookie+'x'})).status,401);assert.equal((await f.request('/',{cookie:cookie+'; '+cookie})).status,401);
});
test('host/origin validation, denied methods and oversized uploads precede backend forwarding',async t=>{
  const f=await fixture(t);
  for(const headers of [{host:'evil.invalid'},{origin:'https://evil.invalid'},{origin:origin+'.evil.invalid'},{'sec-fetch-site':'cross-site'}])assert.ok([400,403].includes((await f.request('/',{authorization,...headers})).status));
  assert.equal((await f.request('/api/jobs',{authorization},'POST','x')).status,403);
  assert.equal((await f.request('/api/jobs',{authorization,origin},'DELETE')).status,405);
  assert.equal((await f.request('/not-an-api',{authorization,origin},'POST','x')).status,405);
  assert.equal((await f.request('/api/jobs',{authorization,origin,'content-length':1025},'POST','x'.repeat(1025))).status,413);
  assert.equal(f.requests.length,0);
  assert.equal((await f.request('/api/jobs',{authorization,origin},'POST','test')).status,200);
  assert.match(f.requests[0].headers.origin,/^http:\/\/127\.0\.0\.1:\d+$/);assert.equal(f.requests[0].body,'test');
});
test('WebSocket relay accepts an authenticated same-origin session and rejects other origins',async t=>{
  const f=await fixture(t),res=await f.request('/',{authorization}),cookie=res.headers['set-cookie'][0].split(';')[0];
  assert.equal(await f.upgrade({cookie,origin:'https://evil.invalid'}),403);assert.equal(await f.upgrade({cookie}),403);
  assert.equal(await f.upgrade({cookie,origin}),101);
  const request=f.requests.find(r=>r.upgrade);assert.ok(request);assert.equal(request.headers.cookie,undefined);assert.match(request.headers.origin,/^http:\/\/127\.0\.0\.1:/);
});
test('configuration rejects weak passwords and unscoped hosts',()=>{
  for(const options of [{publicHost:'example.com',password},{publicHost,password:'weak'},{publicHost,password,upstreamPort:80},{publicHost,password,username:'name:password'}])assert.throws(()=>createGateway(options));
});
test('session cookies from a different gateway process are rejected even with the same password',async t=>{
  const first=await fixture(t),second=await fixture(t),login=await first.request('/',{authorization});
  const cookie=login.headers['set-cookie'][0].split(';')[0];
  assert.equal((await first.request('/',{cookie})).status,200);
  assert.equal((await second.request('/',{cookie})).status,401);
  assert.equal(second.requests.length,0);
});
test('absolute and authority-form targets cannot turn the gateway into an open proxy',async t=>{
  const f=await fixture(t);
  for(const target of ['http://evil.invalid/','//evil.invalid/','/\\evil.invalid/'])assert.equal((await f.request(target,{authorization})).status,400);
  assert.equal(f.requests.length,0);
});
test('chunked uploads are bounded even without Content-Length',async t=>{
  const f=await fixture(t);
  assert.equal((await f.request('/api/jobs',{authorization,origin,'transfer-encoding':'chunked'},'POST','x'.repeat(1025))).status,413);
  assert.equal(f.requests.length,0);
});
