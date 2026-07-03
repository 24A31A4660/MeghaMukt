const puppeteer = require('puppeteer');

(async () => {
    const browser = await puppeteer.launch();
    const page = await browser.newPage();
    
    let logs = [];
    page.on('console', msg => logs.push(msg.text()));
    
    await page.goto('http://127.0.0.1:8080/');
    
    // Wait for load
    await new Promise(r => setTimeout(r, 4000));

    const titleInfo = await page.evaluate(() => {
        const title = document.getElementById('hero-title');
        if (!title) return 'Not found';
        
        const chars = title.querySelectorAll('.pioneer-char');
        const computed = window.getComputedStyle(title);
        
        let charStyles = [];
        if (chars.length > 0) {
            charStyles = Array.from(chars).slice(0, 3).map(c => {
                const s = window.getComputedStyle(c);
                return {
                    text: c.textContent,
                    opacity: s.opacity,
                    transform: s.transform,
                    visibility: s.visibility,
                    className: c.className
                };
            });
        }

        return {
            dataset: title.dataset.split,
            opacity: computed.opacity,
            visibility: computed.visibility,
            height: computed.height,
            charCount: chars.length,
            firstChars: charStyles,
            innerHTML: title.innerHTML.substring(0, 200)
        };
    });

    console.log("Console Logs:", logs);
    console.log("Title Info:", JSON.stringify(titleInfo, null, 2));
    await browser.close();
})();
