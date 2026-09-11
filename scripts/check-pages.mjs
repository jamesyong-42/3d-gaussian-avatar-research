import assert from 'node:assert/strict';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {chromium} from '../web/node_modules/playwright-core/index.mjs';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const browser=await chromium.launch({executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE});
try{
  const page=await browser.newPage();const errors=[];page.on('pageerror',error=>errors.push(error.message));
  await page.goto(pathToFileURL(path.join(root,'docs/index.html')).href);
  for(const width of [320,390,768,1024,1440]){
    await page.setViewportSize({width,height:1000});
    const result=await page.evaluate(()=>({width:document.documentElement.clientWidth,scroll:document.documentElement.scrollWidth,chapters:document.querySelectorAll('section.chapter').length,papers:document.querySelectorAll('article.paper-card, tr[id^="paper-p"]').length,figures:document.querySelectorAll('.figure-top').length,svgs:document.querySelectorAll('svg').length}));
    assert.ok(result.scroll<=result.width+1,`Horizontal overflow at ${width}px`);
    assert.equal(result.chapters,11);assert.equal(result.papers,21);assert.equal(result.figures,9);assert.ok(result.svgs>=7);
  }
  await page.locator('#toc a[href="#generation"]').click();
  await page.waitForURL('**#generation');
  await page.emulateMedia({media:'print'});
  assert.equal(await page.locator('#goals').isVisible(),true);
  assert.deepEqual(errors,[]);
  console.log('Architecture: 11 chapters, inline diagrams, navigation, print visibility and 5 responsive widths passed.');
}finally{await browser.close();}
