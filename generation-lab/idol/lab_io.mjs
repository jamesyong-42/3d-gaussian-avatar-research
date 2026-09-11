// Same-origin bounded downloads; verify whole reassembled data before use.
export async function sha256(binary){return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',binary)),x=>x.toString(16).padStart(2,'0')).join('');}
export async function getJson(url){const r=await fetch(url);if(r.status!==200)throw Error('HTTP '+r.status+' '+url);return r.json();}
export async function getBinary(url,size,expectedHash){
  if(!Number.isSafeInteger(size)||size<=0||size>256*1024**2)throw Error('Invalid payload size');
  const result=new Uint8Array(size),chunk=4*1024**2;
  for(let start=0;start<size;start+=chunk){
    const end=Math.min(size,start+chunk)-1;
    const response=await fetch(url+'?rangeStart='+start,{headers:{Range:`bytes=${start}-${end}`}});
    const part=new Uint8Array(await response.arrayBuffer());
    if(response.status!==206||part.length!==end-start+1||response.headers.get('content-range')!==`bytes ${start}-${end}/${size}`)throw Error('Invalid binary range '+url+' '+start);
    result.set(part,start);
  }
  if(await sha256(result.buffer)!==expectedHash)throw Error('Payload SHA-256 mismatch: '+url);
  return result.buffer;
}
