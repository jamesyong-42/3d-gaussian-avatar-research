import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL,fileURLToPath} from 'node:url';
import {chromium} from '../../web/node_modules/playwright-core/index.mjs';
const report=path.resolve(process.argv[2]??'');
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../reports/idol');
if(!report.startsWith(root+path.sep)||path.basename(report)!=='index.html'||!fs.existsSync(report))throw Error('Supply an IDOL report index');
const out=path.dirname(report),evidence={date:new Date().toISOString(),report,viewports:[],errors:[]};
const browser=await chromium.launch({executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE});
try{
  for(const [name,width,height]of [['desktop',1440,1000],['mobile',390,844]]){
    const page=await browser.newPage({viewport:{width,height},deviceScaleFactor:1});
    page.on('pageerror',e=>evidence.errors.push(String(e)));
    await page.goto(pathToFileURL(report).href);
    await page.evaluate(()=>{for(const image of document.images)image.loading='eager';});
    await page.waitForFunction(()=>[...document.images].every(x=>x.complete));
    const result=await page.evaluate(()=>({images:document.images.length,missingImages:[...document.images].filter(x=>!x.naturalWidth).map(x=>x.src),
      overflow:document.documentElement.scrollWidth>innerWidth,links:[...document.querySelectorAll('a[href]')].map(x=>x.href),placeholders:document.body.textContent.includes('@@')}));
    const missingLocalLinks=result.links.filter(href=>href.startsWith('file:')&&!fs.existsSync(fileURLToPath(new URL(href))));
    const checks={name,width,height,images:result.images,missingImages:result.missingImages,overflow:result.overflow,missingLocalLinks,placeholders:result.placeholders};
    checks.pass=!checks.missingImages.length&&!checks.missingLocalLinks.length&&!checks.overflow&&!checks.placeholders;
    evidence.viewports.push(checks);
    await page.screenshot({path:path.join(out,'report-'+name+'.png')});
    if(name==='desktop'&&await page.locator('.pair').count())await page.locator('.pair').first().screenshot({path:path.join(out,'report-gallery.png')});
    await page.close();
  }
  evidence.pass=evidence.viewports.every(x=>x.pass)&&!evidence.errors.length;
}catch(error){evidence.pass=false;evidence.errors.push(String(error));}
finally{await browser.close();fs.writeFileSync(path.join(out,'report-check.json'),JSON.stringify(evidence,null,2));console.log(JSON.stringify(evidence,null,2));}
if(!evidence.pass)process.exitCode=1;
