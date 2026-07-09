/* ============================================
   main.js — Core Application Entry Point
   Loading screen, init, keyboard navigation,
   custom cursor with trail, scroll wheel control
   ============================================ */

(function () {
    'use strict';

    // ---- Loading Screen ----
    function createLoadingScreen() {
        const loader = document.createElement('div');
        loader.className = 'loading-screen';
        loader.id = 'loading-screen';
        loader.innerHTML = `
            <div class="loading-screen__text">MEGHĀMUKT</div>
            <div class="loading-screen__bar">
                <div class="loading-screen__progress" id="load-progress"></div>
            </div>
        `;
        document.body.prepend(loader);
        return loader;
    }

    function runLoadingSequence(loader) {
        const bar = document.getElementById('load-progress');
        let progress = 0;
        let isFinished = false;
        let fakeInterval = null;

        function finish() {
            if (isFinished) return;
            isFinished = true;
            if (fakeInterval) clearInterval(fakeInterval);
            if (bar) bar.style.width = '100%';
            
            if (loader) {
                loader.classList.add('is-hidden');
                loader.style.opacity = '0';
                loader.style.pointerEvents = 'none';
                setTimeout(() => {
                    if (loader.parentNode) loader.remove();
                    // Trigger hero text reveal or preserve hashed section
                    if (window.AnimationEngine && typeof window.AnimationEngine.activateSection === 'function') {
                        let sec = 0;
                        if (typeof window.AnimationEngine.getCurrentSection === 'function') {
                            const curr = window.AnimationEngine.getCurrentSection();
                            if (curr !== -1) sec = curr;
                        }
                        window.AnimationEngine.activateSection(sec, true);
                    }
                }, 300);
            }
        }

        // Count images to load
        const images = Array.from(document.querySelectorAll('img'));
        let loaded = 0;
        const total = images.length;

        function onImageLoad() {
            loaded++;
            const pct = Math.round((loaded / Math.max(total, 1)) * 100);
            if (bar) bar.style.width = pct + '%';
            if (loaded >= total) finish();
        }

        if (total === 0) {
            finish();
        } else {
            images.forEach(img => {
                if (img.complete) {
                    onImageLoad();
                } else {
                    img.addEventListener('load', onImageLoad);
                    img.addEventListener('error', onImageLoad);
                }
            });
        }

        // Animate progress bar for smooth feel
        fakeInterval = setInterval(() => {
            if (isFinished) return;
            progress += Math.random() * 20 + 10;
            if (progress >= 100) {
                finish();
                return;
            }
            if (bar) bar.style.width = progress + '%';
        }, 80);

        // Fallback safety — hide after 600ms max so user is never stuck
        setTimeout(finish, 600);
    }

    // ---- Premium Custom Cursor with Trail ----
    function initCursor() {
        const cursor = document.getElementById('custom-cursor');
        const cursorDot = document.getElementById('custom-cursor-dot');
        if (!cursor || !cursorDot) return;

        let mouseX = window.innerWidth / 2;
        let mouseY = window.innerHeight / 2;
        let lastX = mouseX, lastY = mouseY;
        let curX = mouseX, curY = mouseY;
        let speed = 0, angle = 0;

        // Trail system
        const TRAIL_LENGTH = 8;
        const trailPositions = [];
        const trailDots = [];
        for (let i = 0; i < TRAIL_LENGTH; i++) {
            const dot = document.createElement('div');
            dot.className = 'cursor-trail-dot';
            dot.style.cssText = `
                position: fixed; top: 0; left: 0;
                width: ${6 - i * 0.5}px; height: ${6 - i * 0.5}px;
                background: rgba(0, 212, 255, ${0.5 - i * 0.05});
                border-radius: 50%; pointer-events: none;
                z-index: 9998; transform: translate(-50%, -50%);
                transition: none; will-change: transform;
            `;
            document.body.appendChild(dot);
            trailDots.push(dot);
            trailPositions.push({ x: mouseX, y: mouseY });
        }

        document.addEventListener('mousemove', e => {
            mouseX = e.clientX;
            mouseY = e.clientY;
            
            // Interactive 3D Liquid Tilt for hero heading (Corn Revolutionized effect)
            const heroTitle = document.querySelector('.hero-title-liquid');
            if (heroTitle) {
                const rect = heroTitle.getBoundingClientRect();
                if (e.clientX >= rect.left - 100 && e.clientX <= rect.right + 100 &&
                    e.clientY >= rect.top - 100 && e.clientY <= rect.bottom + 100) {
                    const centerX = rect.left + rect.width / 2;
                    const centerY = rect.top + rect.height / 2;
                    const tiltX = (e.clientY - centerY) / (rect.height / 2) * -12;
                    const tiltY = (e.clientX - centerX) / (rect.width / 2) * 12;
                    heroTitle.style.transform = `perspective(1000px) rotateX(${tiltX}deg) rotateY(${tiltY}deg)`;
                } else {
                    heroTitle.style.transform = 'perspective(1000px) rotateX(0deg) rotateY(0deg)';
                }
            }
            
            // Render the spotlight mask on any active reveals
            document.querySelectorAll('.title-reveal-wrapper').forEach(wrapper => {
                const rect = wrapper.getBoundingClientRect();
                const rx = e.clientX - rect.left;
                const ry = e.clientY - rect.top;
                const topText = wrapper.querySelector('.title-reveal-top');
                if (topText) {
                    topText.style.clipPath = `circle(90px at ${rx}px ${ry}px)`;
                }
            });
        });

        // Hide spotlight mask when cursor leaves window or titles
        document.addEventListener('mouseleave', () => {
            cursor.style.opacity = '0';
            cursorDot.style.opacity = '0';
            trailDots.forEach(d => d.style.opacity = '0');
        });
        document.addEventListener('mouseenter', () => {
            cursor.style.opacity = '1';
            cursorDot.style.opacity = '1';
            trailDots.forEach(d => d.style.opacity = '1');
        });

        function animateCursor() {
            // Lerp outer cursor (smooth follow, factor 0.1)
            curX += (mouseX - curX) * 0.1;
            curY += (mouseY - curY) * 0.1;

            // Calculate speed for stretch
            const dx = mouseX - lastX;
            const dy = mouseY - lastY;
            speed = Math.sqrt(dx * dx + dy * dy);
            
            if (speed > 1) {
                angle = Math.atan2(dy, dx) * (180 / Math.PI);
            }

            // Apply transformations — outer ring
            const speedScale = Math.min(1 + speed * 0.02, 1.5);
            cursor.style.transform = `translate3d(${curX}px, ${curY}px, 0) rotate(${angle}deg) scale(${speedScale}, ${1 / (speedScale * 0.6 + 0.4)})`;
            // Dot — instant position
            cursorDot.style.transform = `translate3d(${mouseX}px, ${mouseY}px, 0)`;

            // Update trail — shift positions
            for (let i = TRAIL_LENGTH - 1; i > 0; i--) {
                trailPositions[i].x = trailPositions[i - 1].x;
                trailPositions[i].y = trailPositions[i - 1].y;
            }
            trailPositions[0].x = mouseX;
            trailPositions[0].y = mouseY;

            // Render trail dots
            trailDots.forEach((dot, i) => {
                dot.style.transform = `translate3d(${trailPositions[i].x}px, ${trailPositions[i].y}px, 0)`;
            });

            lastX = mouseX;
            lastY = mouseY;
            requestAnimationFrame(animateCursor);
        }
        animateCursor();

        // Magnetic hover snaps
        document.addEventListener('mouseover', e => {
            if (e.target.closest('a, button, .cta-circle, .nav-dots__dot, .side-nav-left__item, .feature-pill, .achievement-card, .telemetry-card')) {
                cursor.classList.add('hovering');
            }
        });
        document.addEventListener('mouseout', e => {
            if (e.target.closest('a, button, .cta-circle, .nav-dots__dot, .side-nav-left__item, .feature-pill, .achievement-card, .telemetry-card')) {
                cursor.classList.remove('hovering');
            }
        });

        // Click reaction
        document.addEventListener('mousedown', () => cursor.classList.add('clicking'));
        document.addEventListener('mouseup', () => cursor.classList.remove('clicking'));
    }

    // ---- Navbar Shrink on Scroll ----
    function initNavbarScroll() {
        const nav = document.getElementById('main-nav');
        if (!nav) return;

        const hero = document.getElementById('hero');
        if (!hero) return;

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    nav.classList.remove('nav--scrolled');
                } else {
                    nav.classList.add('nav--scrolled');
                }
            });
        }, { threshold: 0 });

        observer.observe(hero);
    }

    // ---- Back to Top Button ----
    function initBackToTop() {
        const btn = document.getElementById('back-to-top');
        if (!btn) return;

        // Show/hide based on current section
        const observer = new MutationObserver(() => {
            const current = window.AnimationEngine ? window.AnimationEngine.getCurrentSection() : 0;
            if (current > 0) {
                btn.classList.add('visible');
            } else {
                btn.classList.remove('visible');
            }
        });

        // Observe section class changes
        document.querySelectorAll('.section').forEach(s => {
            observer.observe(s, { attributes: true, attributeFilter: ['class'] });
        });

        btn.addEventListener('click', () => {
            if (window.AnimationEngine) {
                window.AnimationEngine.scrollToSection(0);
            }
        });
    }

    // ---- Nav Center Links ----
    function initNavCenterLinks() {
        document.querySelectorAll('.nav__section-link').forEach(link => {
            link.addEventListener('click', () => {
                const idx = parseInt(link.dataset.section, 10);
                if (window.AnimationEngine && !isNaN(idx)) {
                    window.AnimationEngine.scrollToSection(idx);
                }
            });
        });
    }

    // ---- Keyboard Navigation ----
    function initKeyboardNav() {
        const sections = Array.from(document.querySelectorAll('.section'));

        document.addEventListener('keydown', e => {
            const menu = document.getElementById('menu-overlay');
            if (menu && menu.classList.contains('is-open')) return;

            const current = window.AnimationEngine ? window.AnimationEngine.getCurrentSection() : 0;

            if (e.key === 'ArrowDown' || e.key === 'PageDown') {
                e.preventDefault();
                const next = Math.min(current + 1, sections.length - 1);
                sections[next].scrollIntoView({ behavior: 'smooth' });
            }
            if (e.key === 'ArrowUp' || e.key === 'PageUp') {
                e.preventDefault();
                const prev = Math.max(current - 1, 0);
                sections[prev].scrollIntoView({ behavior: 'smooth' });
            }
        });
    }

    // ---- Touch/Swipe Navigation ----
    function initTouchNav() {
        let startY = 0;
        let isDragging = false;
        const container = document.getElementById('scroll-container');
        const sections = Array.from(document.querySelectorAll('.section'));

        container.addEventListener('touchstart', e => {
            startY = e.touches[0].clientY;
            isDragging = true;
        }, { passive: true });

        container.addEventListener('touchend', e => {
            if (!isDragging) return;
            const endY = e.changedTouches[0].clientY;
            const diff = startY - endY;
            if (Math.abs(diff) < 30) return; // Too small, ignore

            const current = window.AnimationEngine ? window.AnimationEngine.getCurrentSection() : 0;
            if (diff > 0) {
                const next = Math.min(current + 1, sections.length - 1);
                sections[next].scrollIntoView({ behavior: 'smooth' });
            } else {
                const prev = Math.max(current - 1, 0);
                sections[prev].scrollIntoView({ behavior: 'smooth' });
            }
            isDragging = false;
        }, { passive: true });
    }

    // ---- Stagger Section CTA entrances ----
    function initCtaAnimations() {
        document.querySelectorAll('.cta-circle').forEach(cta => {
            // Rotating outer ring on hover
            let angle = 0;
            let animId = null;

            cta.addEventListener('mouseenter', () => {
                const outerRing = cta.querySelector('.cta-circle__ring--outer');
                function rotateBorder() {
                    angle += 0.5;
                    if (outerRing) {
                        outerRing.style.transform = `rotate(${angle}deg)`;
                    }
                    animId = requestAnimationFrame(rotateBorder);
                }
                rotateBorder();
            });

            cta.addEventListener('mouseleave', () => {
                if (animId) cancelAnimationFrame(animId);
                const outerRing = cta.querySelector('.cta-circle__ring--outer');
                if (outerRing) outerRing.style.transform = '';
            });
        });
    }

    // ---- Ambient background color morph based on scroll ----
    function initBackgroundMorph() {
        const container = document.getElementById('scroll-container');
        const colors = [
            [2, 9, 18],     // Hero - deep space
            [5, 12, 24],    // Foundation - dark navy
            [4, 10, 20],    // Observation - dark teal
            [6, 15, 8],     // Results - dark green tint
        ];

        container.addEventListener('scroll', () => {
            const scrollY = container.scrollTop;
            const vh = window.innerHeight;
            const sectionIndex = scrollY / vh;
            const i = Math.floor(sectionIndex);
            const t = sectionIndex - i;

            const ci = Math.min(i, colors.length - 1);
            const cn = Math.min(i + 1, colors.length - 1);

            const r = Math.round(colors[ci][0] + (colors[cn][0] - colors[ci][0]) * t);
            const g = Math.round(colors[ci][1] + (colors[cn][1] - colors[ci][1]) * t);
            const b = Math.round(colors[ci][2] + (colors[cn][2] - colors[ci][2]) * t);

            document.body.style.backgroundColor = `rgb(${r},${g},${b})`;
        });
    }

    // ---- INIT ----
    function init() {
        const loader = createLoadingScreen();
        runLoadingSequence(loader);
        initCursor();
        initKeyboardNav();
        initTouchNav();
        initCtaAnimations();
        initBackgroundMorph();
        initNavbarScroll();
        initBackToTop();
        initNavCenterLinks();
    }

    // Run on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();
