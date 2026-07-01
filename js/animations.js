/* ============================================
   animations.js — Pioneer-Style Section Engine
   Uses TextReveal word-mask slide-up system
   Enhanced with blur transitions & premium animations
   ============================================ */

(function () {
    'use strict';

    const scrollContainer = document.getElementById('scroll-container');
    const sections = Array.from(document.querySelectorAll('.section'));
    let currentSection = -1;
    let isTransitioning = false;

    // ---- Per-section word caches ----
    const sectionData = [];

    function initSectionData() {
        const configs = [
            {
                id: 'hero',
                titleId: 'hero-title',
                bodyId: 'hero-body',
                extras: ['#hero-scroll'],
            },
            {
                id: 'foundation',
                titleId: 'foundation-title',
                bodyId: 'foundation-body',
                extras: ['#telemetry-s02'],
            },
            {
                id: 'observation',
                titleId: 'observation-title',
                bodyId: 'observation-body',
                extras: ['#telemetry-s03'],
            },
            {
                id: 'results',
                titleId: 'results-title',
                bodyId: 'results-body',
                extras: ['#results-cta', '#cta-results', '#success-icon'],
            },
        ];

        configs.forEach((cfg) => {
            const titleEl = document.getElementById(cfg.titleId);
            const bodyEl = document.getElementById(cfg.bodyId);

            const titleWords = window.TextReveal ? window.TextReveal.splitToChars(titleEl, 'cinematic-title') : [];
            const bodyWords = window.TextReveal ? window.TextReveal.splitBodyToWords(bodyEl, 'cinematic-body') : [];

            let extraEls = cfg.extras
                .map(sel => document.querySelector(sel))
                .filter(Boolean);

            if (window.TextReveal) {
                window.TextReveal.resetChars(titleWords);
                window.TextReveal.resetBodyWords(bodyWords);
            }
            if (typeof gsap !== 'undefined' && extraEls.length > 0) {
                gsap.set(extraEls, { opacity: 0, y: 20 });
            }

            sectionData.push({ titleWords, bodyWords, extraEls });
        });
    }

    // ---- Activate a Section ----
    function activateSection(index, force = false) {
        if (!force && index === currentSection) return;
        const prevIndex = currentSection;
        currentSection = index;

        sections.forEach((s, i) => {
            s.classList.toggle('is-active', i === index);
        });

        // ---- GSAP Pioneer Kinetic Character Animations & Custom Scene Tweens ----
        if (typeof gsap !== 'undefined') {
            const TR = window.TextReveal;

            // 1. TEXT REVEAL EXIT / ENTER
            if (TR) {
                if (prevIndex >= 0 && prevIndex < sectionData.length) {
                    const prev = sectionData[prevIndex];
                    if (prev) {
                        TR.resetChars(prev.titleWords);
                        TR.resetBodyWords(prev.bodyWords);
                        gsap.to(prev.extraEls, { opacity: 0, y: 15, duration: 0.2, overwrite: 'auto' });
                    }
                }

                // Handle fixed hero overlay visibility
                if (index === 0 && prevIndex !== 0) {
                    gsap.to('#hero-ui-overlay', { autoAlpha: 1, duration: 0.5, ease: 'power2.inOut' });
                } else if (index !== 0 && prevIndex === 0) {
                    gsap.to('#hero-ui-overlay', { autoAlpha: 0, duration: 0.5, ease: 'power2.out' });
                }

                const curr = sectionData[index];
                if (curr) {
                    let delay = prevIndex === -1 ? 0.15 : 0.1;
                    let textDur = 0.75;
                    let bodyDur = 0.55;
                    
                    if (index === 2) {
                        delay = 0.0;
                        textDur = 0.45;
                        bodyDur = 0.35;
                    } else if (index === 3) {
                        delay = 1.2; // Delay text reveal until Earth cleans
                        textDur = 1.2; // Slower cinematic text reveal
                        bodyDur = 0.8;
                    }
                    
                    TR.revealChars(curr.titleWords, { delay: delay, stagger: 0.02, duration: textDur, ease: 'expo.out' });
                    TR.revealBodyWords(curr.bodyWords, { delay: delay + (index === 2 ? 0.15 : 0.4), stagger: 0.015, duration: bodyDur, ease: 'power2.out' });
                    
                    let extraDelay = delay + (index === 2 ? 0.2 : 0.5);
                    if (index === 3) extraDelay = 3.6; // CTA appears last

                    gsap.fromTo(curr.extraEls,
                        { opacity: 0, y: 20 },
                        { opacity: 1, y: 0, duration: (index === 2 ? 0.4 : 0.8), ease: 'power3.out', delay: extraDelay, overwrite: 'auto' }
                    );
                }
            }

            // 2. MISSION SCENE: Tell EarthRenderer which section we're in
            if (window.EarthRenderer) {
                window.EarthRenderer.setSection(index);
            }

            // 3. CUSTOM SCENE ANIMATIONS (Only play once per section)
            const s = sections[index];
            if (s && !s.dataset.animPlayed) {
                s.dataset.animPlayed = 'true';
                
                if (index === 1) { 
                    // Section 02: Cloud Detection — floating telemetry counters
                    const progressBar = document.getElementById('hud-progress-bar');
                    const cloudPct   = document.getElementById('hud-cloud-pct');
                    const scanProg   = document.getElementById('hud-scan-prog');

                    gsap.timeline({ delay: 0.5 })
                        .to({ val: 0 }, {
                            val: 67, duration: 2.8, ease: 'power3.out',
                            onUpdate: function () {
                                const v = Math.round(this.targets()[0].val);
                                if (cloudPct)  cloudPct.textContent  = v + '%';
                            }
                        }, 0)
                        .to({ val: 0 }, {
                            val: 100, duration: 3.5, ease: 'power2.inOut',
                            onUpdate: function () {
                                const v = Math.round(this.targets()[0].val);
                                if (scanProg)    scanProg.textContent    = v + '%';
                                if (progressBar) progressBar.style.width = v + '%';
                            }
                        }, 0.2);

                    // Animate scanning beam
                    const scanBeam = document.querySelector('.scanning-beam');
                    if (scanBeam) {
                        scanBeam.classList.add('active');
                    }

                    // Animate timeline steps
                    const timelineSteps = document.querySelectorAll('.detection-timeline__step');
                    gsap.fromTo(timelineSteps,
                        { opacity: 0, x: -20 },
                        { opacity: 1, x: 0, duration: 0.6, ease: 'power3.out', stagger: 0.3, delay: 1.0 }
                    );

                    // Animate radial progress for cloud coverage
                    const radialFill = document.getElementById('cloud-radial-fill');
                    if (radialFill) {
                        gsap.to(radialFill, {
                            attr: { 'stroke-dashoffset': 163.36 * (1 - 0.67) },
                            duration: 2.8,
                            ease: 'power3.out',
                            delay: 0.5
                        });
                    }
                        
                } else if (index === 2) { 
                    // Section 03: AI Reconstruction — floating AI metrics
                    const aiConf     = document.getElementById('hud-ai-conf');
                    const inferTime  = document.getElementById('hud-infer-time');
                    const cloudPx    = document.getElementById('hud-cloud-px');

                    gsap.timeline({ delay: 0.05 })
                        .to({ val: 0 }, {
                            val: 99.4, duration: 1.5, ease: 'power3.out',
                            onUpdate: function () {
                                const v = this.targets()[0].val;
                                if (aiConf) aiConf.textContent = v.toFixed(1) + '%';
                            }
                        }, 0)
                        .to({ val: 0 }, {
                            val: 12, duration: 0.9, ease: 'power3.out',
                            onUpdate: function () {
                                const v = this.targets()[0].val;
                                if (inferTime) inferTime.textContent = Math.round(v) + 'ms';
                            }
                        }, 0.15)
                        .to({ val: 0 }, {
                            val: 1847293, duration: 1.8, ease: 'power3.out',
                            onUpdate: function () {
                                const v = Math.round(this.targets()[0].val);
                                if (cloudPx) cloudPx.textContent = v.toLocaleString() + 'px';
                            }
                        }, 0);

                    // Animate spectral bands
                    const spectralFills = document.querySelectorAll('.spectral-band__fill');
                    gsap.fromTo(spectralFills,
                        { width: '0%' },
                        { width: '80%', duration: 1.5, ease: 'power3.out', stagger: 0.2, delay: 0.3 }
                    );

                    // Animate holographic cards
                    const holoCards = document.querySelectorAll('.holo-card');
                    gsap.fromTo(holoCards,
                        { opacity: 0, y: 40, rotateX: -15 },
                        { opacity: 1, y: 0, rotateX: 0, duration: 0.8, ease: 'back.out(1.2)', stagger: 0.15, delay: 0.2 }
                    );

                    // Animate neural network lines
                    const neuralLines = document.querySelectorAll('.neural-line');
                    gsap.fromTo(neuralLines,
                        { attr: { 'stroke-dasharray': '20', 'stroke-dashoffset': '20' } },
                        { attr: { 'stroke-dashoffset': '0' }, duration: 1.2, ease: 'power2.inOut', stagger: 0.05, delay: 0.5 }
                    );
                    
                } else if (index === 3) { 
                    // Section 04: Mission Complete — metric pill counters
                    const psnrEl    = document.getElementById('metric-psnr');
                    const ssimEl    = document.getElementById('metric-ssim');
                    const rmseEl    = document.getElementById('metric-rmse');
                    const removalEl = document.getElementById('metric-removal');

                    // Sequence:
                    // 1. Earth cleans & hologram fades (earth.js)
                    // 2. Title appears (text reveal delays adjusted below)
                    // 3. Metrics animate in
                    // 4. CTA appears
                    
                    const pills = document.querySelectorAll('.metric-pill');
                    gsap.fromTo(pills,
                        { opacity: 0, y: 30, scale: 0.9 },
                        { opacity: 1, y: 0, scale: 1, duration: 1.0, ease: 'power3.out', stagger: 0.15, delay: 2.2, overwrite: 'auto' }
                    );

                    gsap.timeline({ delay: 2.4 })
                        .to({ val: 0 }, {
                            val: 34.5, duration: 2.8, ease: 'power3.out',
                            onUpdate: function () {
                                const v = this.targets()[0].val;
                                if (psnrEl) psnrEl.textContent = v.toFixed(1);
                            }
                        }, 0)
                        .to({ val: 0 }, {
                            val: 0.942, duration: 2.8, ease: 'power3.out',
                            onUpdate: function () {
                                const v = this.targets()[0].val;
                                if (ssimEl) ssimEl.textContent = v.toFixed(3);
                            }
                        }, 0)
                        .to({ val: 0.05 }, {
                            val: 0.028, duration: 2.8, ease: 'power3.out',
                            onUpdate: function () {
                                const v = this.targets()[0].val;
                                if (rmseEl) rmseEl.textContent = v.toFixed(3);
                            }
                        }, 0)
                        .to({ val: 0 }, {
                            val: 99.8, duration: 2.8, ease: 'power3.out',
                            onUpdate: function () {
                                const v = this.targets()[0].val;
                                if (removalEl) removalEl.textContent = v.toFixed(1) + '%';
                            }
                        }, 0);
                        
                    // Make sure extraEls (like buttons) appear very last
                }
            }
        }
    }

    // ---- Smooth Snap Scroll ----
    function scrollToSection(index) {
        if (index < 0 || index >= sections.length) return;
        if (isTransitioning) return;
        isTransitioning = true;

        const targetScrollTop = index * window.innerHeight;
        const scrollObj = { y: scrollContainer.scrollTop };

        // Slow down transition to Section 04 for cinematic reveal
        let scrollDur = 1.8;
        if (currentSection === 2 && index === 3) scrollDur = 2.6;

        // Blur transition effect at midpoint
        gsap.to(document.body, {
            filter: 'blur(3px)',
            duration: scrollDur * 0.3,
            ease: 'power2.in',
            onComplete: () => {
                gsap.to(document.body, {
                    filter: 'blur(0px)',
                    duration: scrollDur * 0.4,
                    ease: 'power2.out'
                });
            }
        });

        // Snappy scroll transition matching modern web standard, but smoother and slower for cinematic feel
        gsap.to(scrollObj, {
            y: targetScrollTop,
            duration: scrollDur,
            ease: 'power3.inOut',
            onStart: () => {
                activateSection(index);
            },
            onUpdate: () => {
                scrollContainer.scrollTop = scrollObj.y;
                triggerScrollCalculations();
            },
            onComplete: () => {
                isTransitioning = false;
            }
        });
    }

    function triggerScrollCalculations() {
        const scrollY = scrollContainer.scrollTop;
        const maxScroll = scrollContainer.scrollHeight - window.innerHeight;
        const progress = maxScroll > 0 ? scrollY / maxScroll : 0;

        if (window.EarthRenderer) window.EarthRenderer.setScrollProgress(progress);

        const diagram = document.querySelector('.central-diagram__svg');
        if (diagram) {
            diagram.style.transform = `translateY(${progress * -180}px)`;
        }

        sections.forEach(section => {
            const rect = section.getBoundingClientRect();
            const centerOffset = rect.top + rect.height / 2 - window.innerHeight / 2;
            
            // Background Parallax
            const img = section.querySelector('.section__bg-img');
            if (img) img.style.transform = `translateY(${centerOffset * 0.08}px)`;

            // Cinematic fade and translate for content (excluding Hero/Section 01)
            if (section.id !== 'hero') {
                const content = section.querySelector('.section__content');
                if (content) {
                    const ratio = Math.min(1, Math.abs(centerOffset) / (window.innerHeight * 0.7));
                    const opacity = 1 - Math.pow(ratio, 2.5);
                    const translateY = centerOffset * 0.12;
                    content.style.opacity = opacity;
                    content.style.transform = `translateY(${translateY}px)`;
                }
            }
        });
    }

    scrollContainer.addEventListener('scroll', triggerScrollCalculations);

    // ---- Nav Dots ----
    document.querySelectorAll('.nav-dots__dot').forEach(dot => {
        dot.addEventListener('click', () => scrollToSection(parseInt(dot.dataset.section, 10)));
    });

    // ---- Menu Links ----
    document.querySelectorAll('.menu-overlay__link').forEach(link => {
        link.addEventListener('click', (e) => {
            const idx = parseInt(link.dataset.section, 10);
            if (!isNaN(idx)) {
                e.preventDefault();
                scrollToSection(idx);
                document.getElementById('menu-overlay').classList.remove('is-open');
                document.getElementById('nav-hamburger').classList.remove('is-open');
            }
        });
    });

    // ---- Mouse Wheel ----
    window.addEventListener('wheel', (e) => {
        if (isTransitioning) return;
        if (Math.abs(e.deltaY) > 8) {
            scrollToSection(e.deltaY > 0 ? currentSection + 1 : currentSection - 1);
        }
    }, { passive: true });

    // ---- Keyboard ----
    document.addEventListener('keydown', (e) => {
        if (isTransitioning) return;
        if (e.key === 'ArrowDown' || e.key === 'PageDown') scrollToSection(currentSection + 1);
        else if (e.key === 'ArrowUp' || e.key === 'PageUp') scrollToSection(currentSection - 1);
    });

    // ---- Touch Swipe ----
    let touchStartY = 0;
    window.addEventListener('touchstart', e => { touchStartY = e.touches[0].clientY; }, { passive: true });
    window.addEventListener('touchend', e => {
        if (isTransitioning) return;
        const diff = touchStartY - e.changedTouches[0].clientY;
        if (Math.abs(diff) > 50) scrollToSection(diff > 0 ? currentSection + 1 : currentSection - 1);
    }, { passive: true });

    // ---- Hamburger Menu ----
    const hamburger = document.getElementById('nav-hamburger');
    const menuOverlay = document.getElementById('menu-overlay');
    hamburger.addEventListener('click', () => {
        hamburger.classList.toggle('is-open');
        menuOverlay.classList.toggle('is-open');
    });
    menuOverlay.addEventListener('click', (e) => {
        if (e.target === menuOverlay || e.target.classList.contains('menu-overlay__bg')) {
            hamburger.classList.remove('is-open');
            menuOverlay.classList.remove('is-open');
        }
    });
    document.addEventListener('keydown', e => {
        if (e.key === 'Escape' && menuOverlay.classList.contains('is-open')) {
            hamburger.classList.remove('is-open');
            menuOverlay.classList.remove('is-open');
        }
    });

    // ---- Initialize ----
    let isInitialized = false;
    function initApp() {
        if (isInitialized) return;
        isInitialized = true;
        initSectionData();
        initObservers();
        // Do NOT activateSection(0) here. Wait for main.js loading screen to finish.
    }
    function initObservers() {
        // 1. Auth Buttons (Fade out when leaving Section 01)
        const authGroup = document.querySelector('.nav__auth-group');
        if (authGroup && sections.length > 0) {
            const heroObserver = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        gsap.to(authGroup, { opacity: 1, y: 0, duration: 0.4, ease: 'power2.out', overwrite: 'auto' });
                        authGroup.style.pointerEvents = 'auto';
                    } else {
                        gsap.to(authGroup, { opacity: 0, y: -20, duration: 0.4, ease: 'power2.out', overwrite: 'auto' });
                        authGroup.style.pointerEvents = 'none';
                    }
                });
            }, { threshold: 0.1 });
            heroObserver.observe(sections[0]);
        }

        // 2. Observer for Left Nav and Right Dots
        const visibleSections = new Map();
        const sectionObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                visibleSections.set(entry.target, entry.intersectionRatio);
            });
            
            let maxRatio = 0;
            let bestIndex = -1;
            sections.forEach((sec, i) => {
                let ratio = visibleSections.get(sec) || 0;
                
                // Artificially boost Section 03 so it activates immediately on scroll
                if (i === 2 && ratio > 0.15) {
                    ratio += 1.0;
                }

                // Penalize Section 04 so the user has to scroll significantly more before it triggers
                if (i === 3 && ratio < 0.6) {
                    ratio -= 0.5;
                }
                
                if (ratio > maxRatio) {
                    maxRatio = ratio;
                    bestIndex = i;
                }
            });
            
            if (bestIndex !== -1 && maxRatio > 0.1) {
                syncNavState(bestIndex);
            }
        }, { 
            threshold: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] 
        });

        sections.forEach(sec => sectionObserver.observe(sec));
    }

    function syncNavState(index) {
        // Left Nav
        document.querySelectorAll('.side-nav-left__item').forEach((item, i) => {
            item.classList.toggle('active', i === index);
        });
        const progressEl = document.querySelector('.side-nav-left__progress');
        if (progressEl && sections.length > 1) {
            progressEl.style.height = `${(index / (sections.length - 1)) * 100}%`;
        }

        // Right Dots
        document.querySelectorAll('.nav-dots__dot').forEach((dot, i) => {
            dot.classList.toggle('active', i === index);
        });

        // Tell EarthRenderer which section is now active
        if (window.EarthRenderer) {
            window.EarthRenderer.setSection(index);
        }

        // Trigger GSAP cinematic reveals and scene animations for active section
        activateSection(index);
    }

    if (document.readyState === 'complete' || document.readyState === 'interactive') {
        initApp();
    } else {
        window.addEventListener('DOMContentLoaded', initApp);
        window.addEventListener('load', initApp);
    }

    // ---- Expose API ----
    window.AnimationEngine = {
        activateSection,
        scrollToSection,
        getCurrentSection: () => currentSection,
    };

})();
