const fs = require('fs');
const path = require('path');
const {pathToFileURL} = require('url');
const {chromium} = require('C:/Users/khchoi_exp/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert = require('assert/strict');
(async()=>{
  const out=path.join(__dirname,'qa');fs.mkdirSync(out,{recursive:true});
  const browser=await chromium.launch({headless:true,channel:'msedge'});
  const checks=[];
  try{
    for(const name of ['paper-vs-results','paper-study']){
      const page=await browser.newPage({viewport:{width:1280,height:900}});
      const errors=[];page.on('pageerror',e=>errors.push(e.message));
      await page.goto(pathToFileURL(path.join(__dirname,name+'.html')).href);
      await page.waitForFunction(()=>document.querySelectorAll('svg path').length>0);
      assert.equal(await page.locator('h1').count(),1);
      await page.locator('#mode').click();
      assert.equal(await page.locator('#mode').getAttribute('aria-pressed'),'true');
      assert.ok(await page.locator('.analogy').first().isVisible());
      await page.locator('#expand').click();
      assert.ok(await page.locator('details').evaluateAll(ds=>ds.every(d=>d.open)));
      await page.locator('#expand').click();
      if(name==='paper-study'){
        await page.locator('#phase-pi').click();
        assert.match(await page.locator('#interference-result').innerText(),/I = 0\.000/);
        await page.locator('#phase-zero').click();
        assert.match(await page.locator('#interference-result').innerText(),/I = 4\.000/);
        await page.locator('#depth-slider').fill('35');await page.locator('#depth-slider').dispatchEvent('input');
        assert.match(await page.locator('#mask-result').innerText(),/35 mm 평면/);
        await page.locator('#depth-slider').fill('12.5');await page.locator('#depth-slider').dispatchEvent('input');
        assert.match(await page.locator('#mask-result').innerText(),/10 mm 평면/);
        await page.locator('#architecture-select').selectOption('current');
        assert.match(await page.locator('#architecture-note').innerText(),/진폭의 곱셈 보정/);
        for(const box of await page.locator('.exercise').all()){
          const answer=await box.getAttribute('data-answer');
          await box.locator('button[data-choice="'+answer+'"]').click();
          assert.match(await box.locator('.feedback').innerText(),/^맞습니다/);
        }
      }else{
        assert.equal(await page.locator('#final-table tbody tr').count(),9);
        for(const id of ['0846','0898','0858','0890','0853','0849','0808','0876']){
          await page.locator('#target-select').selectOption(id);
          assert.match(await page.locator('#citl-detail').innerText(),new RegExp(id));
          await page.waitForFunction(()=>[...document.images].every(im=>im.complete && im.naturalWidth>0));
          assert.equal(await page.locator('#image-grid img').count(),4);
        }
        await page.locator('#target-select').selectOption('0846');
        await page.locator('#diagnosis-select').selectOption('0898');
        assert.equal(await page.locator('#diagnosis-curve path').count(),2);
      }
      const hrefs=await page.locator('a[href]').evaluateAll(as=>as.map(a=>a.getAttribute('href')));
      for(const href of hrefs){
        if(/^(https?:|chatgpt-conversation:)/.test(href))continue;
        if(href.startsWith('#'))assert.ok(await page.locator(href).count(),'anchor '+href);
        else assert.ok(fs.existsSync(path.resolve(__dirname,decodeURIComponent(href.split('#')[0]))),'file '+href);
      }
      const ids=await page.locator('[id]').evaluateAll(es=>es.map(e=>e.id));
      assert.equal(new Set(ids).size,ids.length,'duplicate ids');
      for(const width of [1280,768,360,320]){
        await page.setViewportSize({width,height:900});
        await page.waitForTimeout(100);
        const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>window.innerWidth+1);
        assert.equal(overflow,false,name+' width '+width+' overflow');
        checks.push({document:name,width,overflow:false});
        await page.evaluate(()=>window.scrollTo(0,0));
        if(width===1280||width===360)await page.screenshot({path:path.join(out,name+'-'+width+'.png')});
      }
      await page.setViewportSize({width:1280,height:900});
      const section=name==='paper-study'?'#field':'#explorer';
      await page.locator(section).scrollIntoViewIfNeeded();
      await page.locator(section).screenshot({path:path.join(out,name+'-interactive.png')});
      assert.deepEqual(errors,[]);
      await page.close();
    }
    fs.writeFileSync(path.join(out,'verification.json'),JSON.stringify({status:'passed',checks,verified:['offline file loading','local source links','HTML anchors','image decode for all 8 targets','quiz feedback','phase extremes','depth endpoint and tie','architecture switch','tables','responsive widths','no JavaScript page errors']},null,2));
    console.log(JSON.stringify({status:'passed',checks:checks.length}));
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1);});
