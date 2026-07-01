const puppeteer = require('puppeteer');

(async () => {
    const browser = await puppeteer.launch();
    const page = await browser.newPage();
    page.on('console', msg => console.log('BROWSER CONSOLE:', msg.text()));
    page.on('pageerror', err => console.log('BROWSER ERROR:', err.toString()));
    
    await page.goto('http://localhost:5000', { waitUntil: 'networkidle0' });
    
    // Check if the characters exist and are visible
    const charsVisible = await page.evaluate(() => {
        const chars = document.querySelectorAll('.pioneer-char');
        if (chars.length === 0) return 'No chars found';
        
        let visibleCount = 0;
        let opacitySum = 0;
        chars.forEach(c => {
            const style = window.getComputedStyle(c);
            opacitySum += parseFloat(style.opacity);
            if (parseFloat(style.opacity) > 0) visibleCount++;
        });
        
        return `Found ${chars.length} chars. ${visibleCount} are visible. Total opacity: ${opacitySum}`;
    });
    console.log('DOM check:', charsVisible);
    
    await browser.close();
})();
