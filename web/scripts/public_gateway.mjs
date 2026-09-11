// Password-protected, loopback-only gateway for this lab's Tailscale Funnel.
// No dependencies, password files, upstream changes or public auth bypasses.
import http from 'node:http';
import crypto from 'node:crypto';

const hash=value=>crypto.createHash('sha256').update(value).digest();
const constantEqual=(a,b)=>crypto.timingSafeEqual(hash(a),hash(b));
const cookieName='__Host-AvatarLabSession',sessionSeconds=8*60*60;
const securityHeaders={
  'cache-control':'private, no-store','vary':'Authorization, Cookie',
  'x-content-type-options':'nosniff','x-frame-options':'DENY',
  'referrer-policy':'no-referrer','content-security-policy':"frame-ancestors 'none'",
  'permissions-policy':'camera=(self), microphone=(self)',
};

export function createGateway({publicHost,upstreamPort=8765,username='avatar',password,maxUploadBytes=65*1024*1024}){
  if(!/^[a-z0-9][a-z0-9.-]*\.ts\.net$/.test(publicHost))throw Error('Expected the exact Tailscale DNS name');
  if(!Number.isInteger(upstreamPort)||upstreamPort<1024||upstreamPort>65535)throw Error('Invalid upstream port');
  if(!/^[a-zA-Z0-9_-]{1,40}$/.test(username)||typeof password!=='string'||password.length<20)throw Error('Strong password required');
  const publicOrigin='https://'+publicHost,expected=hash(username+':'+password),sessionKey=crypto.randomBytes(32);
  const sockets=new Set(),upstreams=new Set();
  function basic(req){
    const auth=req.headers.authorization;
    if(typeof auth!=='string'||auth.length>2048||!/^Basic [a-zA-Z0-9+/=]+$/i.test(auth))return false;
    return crypto.timingSafeEqual(expected,hash(Buffer.from(auth.slice(6),'base64')));
  }
  const sign=value=>crypto.createHmac('sha256',sessionKey).update(value).digest('base64url');
  function session(req){
    const matches=(req.headers.cookie??'').split(';').map(s=>s.trim()).filter(s=>s.startsWith(cookieName+'='));
    if(matches.length!==1)return false;
    const token=matches[0].slice(cookieName.length+1),parts=token.split('.');
    if(parts.length!==3||!/^\d{10}$/.test(parts[0])||! /^[a-zA-Z0-9_-]{16}$/.test(parts[1])||! /^[a-zA-Z0-9_-]{43}$/.test(parts[2]))return false;
    const now=Math.floor(Date.now()/1000),expiry=Number(parts[0]);
    return expiry>now&&expiry<=now+sessionSeconds+5&&constantEqual(sign(parts[0]+'.'+parts[1]),parts[2]);
  }
  function cookie(){
    const value=(Math.floor(Date.now()/1000)+sessionSeconds)+'.'+crypto.randomBytes(12).toString('base64url');
    return `${cookieName}=${value}.${sign(value)}; Path=/; Secure; HttpOnly; SameSite=Strict; Max-Age=${sessionSeconds}`;
  }
  function gate(req,websocket=false){
    // Authenticate *every* route, including binaries, API, OPTIONS and upgrades.
    const authenticatedByBasic=basic(req);
    if(!authenticatedByBasic&&!session(req))return{status:401,message:'Password required.',headers:{'www-authenticate':'Basic realm="Gaussian Avatar Lab", charset="UTF-8"'}};
    const address=server.address(),host=req.headers.host;
    if(![publicHost,publicHost+':443','127.0.0.1:'+address.port,'localhost:'+address.port].includes(host))return{status:400,message:'Unrecognized host.'};
    if(!req.url?.startsWith('/')||req.url.startsWith('//')||/[\r\n\\]/.test(req.url))return{status:400,message:'Invalid request target.'};
    const origin=req.headers.origin;
    if(origin&&origin!==publicOrigin)return{status:403,message:'Origin not allowed.'};
    if((websocket||!['GET','HEAD','OPTIONS'].includes(req.method))&&origin!==publicOrigin)return{status:403,message:'Same-origin request required.'};
    if(req.headers['sec-fetch-site']==='cross-site')return{status:403,message:'Cross-site request blocked.'};
    if(websocket){
      if(req.method!=='GET'||req.headers.upgrade?.toLowerCase()!=='websocket'||!/^\/ws\/[a-zA-Z0-9_-]{1,40}(?:\?|$)/.test(req.url))return{status:400,message:'Invalid WebSocket route.'};
    }else if(!['GET','HEAD','OPTIONS','POST'].includes(req.method)||(req.method==='POST'&&req.url.split('?')[0]!=='/api/jobs'))return{status:405,message:'Method not allowed.'};
    if(req.headers['content-length']&&(!/^\d+$/.test(req.headers['content-length'])||Number(req.headers['content-length'])>maxUploadBytes))return{status:413,message:'Upload too large.'};
    return{authenticatedByBasic};
  }
  function headers(req,websocket=false){
    const out={...req.headers};
    const hop=new Set(['connection','proxy-connection','keep-alive','transfer-encoding','te','trailer','upgrade','authorization','proxy-authorization','cookie','forwarded',
      ...(req.headers.connection??'').split(',').map(s=>s.trim().toLowerCase())]);
    for(const key of Object.keys(out))if(hop.has(key)||key.startsWith('x-forwarded-')||key.startsWith('tailscale-'))delete out[key];
    out.host='127.0.0.1:'+upstreamPort;
    // Only an already-validated exact public origin is translated for the
    // unchanged local-only backend; arbitrary Origin headers are never trusted.
    if(req.headers.origin)out.origin='http://127.0.0.1:'+upstreamPort;
    if(websocket){out.connection='Upgrade';out.upgrade='websocket';}
    return out;
  }
  const deny=(res,result)=>{res.writeHead(result.status,{...securityHeaders,'content-type':'text/plain; charset=utf-8',...result.headers});res.end(result.message);};
  const server=http.createServer({maxHeaderSize:16384},(req,res)=>{
    const access=gate(req);if(access.status){deny(res,access);req.resume();return;}
    let bytes=0;
    const upstream=http.request({hostname:'127.0.0.1',port:upstreamPort,path:req.url,method:req.method,headers:headers(req)},incoming=>{
      const out={...incoming.headers,...securityHeaders};delete out['set-cookie'];
      if(access.authenticatedByBasic)out['set-cookie']=cookie();
      res.writeHead(incoming.statusCode,out);incoming.pipe(res);
      incoming.on('error',()=>res.destroy());
    });
    upstreams.add(upstream);upstream.on('close',()=>upstreams.delete(upstream));
    upstream.setTimeout(120000,()=>upstream.destroy());
    upstream.on('error',()=>{if(!res.headersSent)deny(res,{status:502,message:'Local prototype unavailable.'});else res.destroy();});
    req.on('data',chunk=>{bytes+=chunk.length;if(bytes>maxUploadBytes){req.unpipe(upstream);if(!res.headersSent)deny(res,{status:413,message:'Upload too large.'});upstream.destroy();req.resume();}});
    req.on('aborted',()=>upstream.destroy());req.on('error',()=>upstream.destroy());
    res.on('close',()=>upstream.destroy());req.pipe(upstream);
  });
  server.on('upgrade',(req,socket,head)=>{
    const access=gate(req,true);
    if(access.status){socket.end(`HTTP/1.1 ${access.status} Rejected\r\nConnection: close\r\nContent-Length: 0\r\n\r\n`);return;}
    const upstream=http.request({hostname:'127.0.0.1',port:upstreamPort,path:req.url,method:'GET',headers:headers(req,true)});
    upstreams.add(upstream);upstream.on('close',()=>upstreams.delete(upstream));
    upstream.setTimeout(15000,()=>upstream.destroy());
    upstream.on('upgrade',(response,remote,remoteHead)=>{
      sockets.add(remote);remote.on('close',()=>sockets.delete(remote));remote.setTimeout(0);
      socket.write('HTTP/1.1 101 Switching Protocols\r\n'+Object.entries(response.headers).map(([k,v])=>`${k}: ${v}\r\n`).join('')+'\r\n');
      if(remoteHead.length)socket.write(remoteHead);if(head.length)remote.write(head);
      socket.pipe(remote);remote.pipe(socket);
      socket.on('close',()=>remote.destroy());remote.on('close',()=>socket.destroy());
      socket.on('error',()=>remote.destroy());remote.on('error',()=>socket.destroy());
    });
    upstream.on('response',response=>{response.resume();socket.end(`HTTP/1.1 ${response.statusCode} Rejected\r\nConnection: close\r\nContent-Length: 0\r\n\r\n`);});
    upstream.on('error',()=>socket.destroy());socket.on('close',()=>upstream.destroy());upstream.end();
  });
  server.on('connection',socket=>{sockets.add(socket);socket.on('close',()=>sockets.delete(socket));socket.on('error',()=>socket.destroy());});
  server.on('clientError',(_error,socket)=>socket.end('HTTP/1.1 400 Bad Request\r\nConnection: close\r\nContent-Length: 0\r\n\r\n'));
  server.headersTimeout=15000;server.requestTimeout=120000;server.keepAliveTimeout=5000;server.maxConnections=64;
  return{server,publicOrigin,username,
    listen:port=>new Promise((resolve,reject)=>{server.once('error',reject);server.listen(port,'127.0.0.1',()=>{server.removeListener('error',reject);resolve(server.address());});}),
    close:()=>new Promise(resolve=>{for(const request of upstreams)request.destroy();for(const socket of sockets)socket.destroy();server.close(resolve);})};
}
