/* ============================================
   text-reveal.js — Pioneer GIF Kinetic Character
   & Word Reveal Engine
   
   Technique: Each character/word is wrapped in
   an overflow:hidden mask clip. On reveal, characters
   rapidly stagger up into view with dynamic skew and
   kinetic precision matching cornrevolution.resn.global.
   ============================================ */

(function () {
    'use strict';

    function splitToChars(el) {
        if (!el) return [];
        if (el.dataset.split === 'done') {
            return Array.from(el.querySelectorAll('.pioneer-char'));
        }

        const rawNodes = Array.from(el.childNodes);
        el.innerHTML = '';
        const allChars = [];
        
        let currentLine = document.createElement('div');
        currentLine.className = 'pioneer-line';
        el.appendChild(currentLine);

        function processText(text, classNames) {
            const words = text.split(/(\s+)/); // Keep spaces to preserve whitespace
            
            words.forEach(word => {
                if (word.trim().length === 0) {
                    // It's whitespace
                    const space = document.createElement('span');
                    space.className = 'pioneer-space';
                    space.innerHTML = '&nbsp;';
                    currentLine.appendChild(space);
                } else {
                    const wordSpan = document.createElement('span');
                    wordSpan.className = 'pioneer-word-group';

                    for (let i = 0; i < word.length; i++) {
                        const char = word[i];
                        const maskClip = document.createElement('span');
                        maskClip.className = 'pioneer-char-mask';

                        const charSpan = document.createElement('span');
                        charSpan.className = 'pioneer-char';
                        if (classNames) charSpan.className += ' ' + classNames;
                        charSpan.textContent = char;

                        maskClip.appendChild(charSpan);
                        wordSpan.appendChild(maskClip);
                        allChars.push(charSpan);
                    }
                    currentLine.appendChild(wordSpan);
                }
            });
        }

        rawNodes.forEach(node => {
            if (node.nodeType === Node.TEXT_NODE) {
                processText(node.textContent, '');
            } else if (node.nodeType === Node.ELEMENT_NODE) {
                if (node.tagName.toLowerCase() === 'br') {
                    currentLine = document.createElement('div');
                    currentLine.className = 'pioneer-line';
                    el.appendChild(currentLine);
                } else {
                    processText(node.textContent, node.className);
                }
            }
        });

        el.dataset.split = 'done';
        return allChars;
    }

    /**
     * Split body paragraph text into individual word spans
     */
    function splitBodyToWords(el) {
        if (!el) return [];
        if (el.dataset.split === 'done') {
            return Array.from(el.querySelectorAll('.body-word'));
        }

        const text = el.textContent.trim();
        const words = text.split(/\s+/).filter(Boolean);

        el.innerHTML = '';
        const spans = words.map((word, i) => {
            const span = document.createElement('span');
            span.className = 'body-word';
            span.textContent = word;
            el.appendChild(span);
            if (i < words.length - 1) el.appendChild(document.createTextNode(' '));
            return span;
        });

        el.dataset.split = 'done';
        return spans;
    }

    /**
     * Animate character units with kinetic Pioneer slide-up
     */
    function revealChars(chars, options = {}) {
        if (!chars || chars.length === 0) return;
        const {
            delay = 0.08,
            stagger = 0.018, // Authentic kinetic character stagger like Resn Pioneer
            duration = 0.75,
            ease = 'expo.out'
        } = options;

        if (typeof gsap !== 'undefined') {
            gsap.killTweensOf(chars);
            gsap.fromTo(chars,
                {
                    yPercent: 120,
                    y: 0,
                    opacity: 0,
                    scale: 1.0,
                    filter: 'blur(0px)'
                },
                {
                    yPercent: 0,
                    y: 0,
                    opacity: 1,
                    scale: 1.0,
                    filter: 'blur(0px)',
                    duration,
                    ease,
                    stagger,
                    delay,
                    overwrite: 'auto',
                    onStart: () => console.log('revealChars tween started', chars.length),
                    onComplete: () => console.log('revealChars tween completed')
                }
            );
        }
    }

    function resetChars(chars) {
        if (!chars || chars.length === 0) return;
        if (typeof gsap !== 'undefined') {
            gsap.set(chars, { yPercent: 120, y: 0, opacity: 0, scale: 1.0, filter: 'blur(0px)' });
        }
    }

    function revealBodyWords(words, options = {}) {
        if (!words || words.length === 0) return;
        const {
            delay = 0.3,
            stagger = 0.015,
            duration = 0.5,
            ease = 'power2.out'
        } = options;

        if (typeof gsap !== 'undefined') {
            gsap.killTweensOf(words);
            gsap.fromTo(words,
                { opacity: 0, y: 12 },
                { opacity: 1, y: 0, duration, ease, stagger, delay, overwrite: 'auto' }
            );
        }
    }

    function resetBodyWords(words) {
        if (!words || words.length === 0) return;
        if (typeof gsap !== 'undefined') {
            gsap.set(words, { opacity: 0, y: 12 });
        }
    }

    window.TextReveal = {
        splitToWords: splitToChars, // Compatibility wrapper
        splitToChars,
        splitBodyToWords,
        revealWords: revealChars,
        revealChars,
        resetWords: resetChars,
        resetChars,
        revealBodyWords,
        resetBodyWords,
    };

})();
