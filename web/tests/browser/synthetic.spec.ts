import {test,expect} from '@playwright/test';

test('clean-clone viewer loads and animates without photos or native weights',async({page,request})=>{
  const health=await (await request.get('/api/health')).json();
  expect(health.ok).toBe(true);expect(health.generationEnabled).toBe(false);
  const response=await request.post('/api/jobs');expect(response.status()).toBe(409);
  const errors:string[]=[];page.on('pageerror',error=>errors.push(error.message));
  await page.goto('/?asset=synthetic-demo');
  await page.waitForFunction(()=>(window as any).__lab?.state.loaded);
  await expect(page.locator('#avatar-title')).toContainText('Synthetic demo');
  await expect(page.locator('#photo-status')).toContainText('no source photos');
  await expect(page.locator('#generate')).toBeDisabled();
  await expect(page.locator('#verify')).toBeDisabled();
  await page.waitForFunction(()=>(window as any).__lab.state.renderedFrames>5);
  // Inspect the composited canvas screenshot: Spark may leave an offscreen
  // framebuffer bound, so a raw WebGL readPixels probe is not visual evidence.
  const png=await page.locator('#viewport canvas').screenshot();
  const visible=await page.evaluate(async base64=>{
    const image=new Image();image.src='data:image/png;base64,'+base64;await image.decode();
    const canvas=document.createElement('canvas');canvas.width=image.width;canvas.height=image.height;
    const ctx=canvas.getContext('2d')!;ctx.drawImage(image,0,0);const pixels=ctx.getImageData(0,0,canvas.width,canvas.height).data;
    let count=0;for(let i=0;i<pixels.length;i+=4)if(Math.max(pixels[i],pixels[i+1],pixels[i+2])>120&&Math.max(pixels[i],pixels[i+1],pixels[i+2])-Math.min(pixels[i],pixels[i+1],pixels[i+2])>50)count++;
    return count;
  },png.toString('base64'));
  expect(visible).toBeGreaterThan(1000);
  await page.locator('[data-clip="wave"]').click();
  await page.waitForFunction(()=>Math.abs((window as any).__lab.state.pose.angles[50])>.1);
  expect((await page.evaluate(()=>(window as any).__lab.state)).clip).toBe('wave');
  expect(errors).toEqual([]);
});

test('synthetic recording and second-browser receiver use the resolved pose',async({page,context})=>{
  await page.goto('/?asset=synthetic-demo');
  await page.waitForFunction(()=>(window as any).__lab?.state.loaded);
  await page.locator('[data-clip="walk"]').click();
  await page.locator('#record').click();
  await page.waitForFunction(()=>(window as any).__lab.state.recorded>15);
  await page.locator('#record').click();await expect(page.locator('#replay')).toBeEnabled();
  await page.locator('#replay').click();
  await page.waitForFunction(()=>(window as any).__lab.state.replaying);
  const popup=context.waitForEvent('page');await page.locator('#share').click();
  const receiver=await popup;
  await receiver.waitForFunction(()=>(window as any).__lab?.state.loaded&&(window as any).__lab.state.received>3);
  const state=await receiver.evaluate(()=>(window as any).__lab.state);
  expect(state.assetId).toBe('synthetic-demo');expect(state.remote).toBe(true);
  expect(state.pose.rootPosition.every(Number.isFinite)).toBe(true);
  await receiver.close();
});
