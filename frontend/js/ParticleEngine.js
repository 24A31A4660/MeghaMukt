/* ============================================
   particles.js — Premium Star Field & Constellation
   Features: 400 stars with 3 depth layers, nebula blobs,
   shooting stars, satellite trajectory lines,
   grid overlay, section-aware color shifting
   ============================================ */

(function () {
    'use strict';

    const canvas = document.getElementById('particle-canvas');
    const ctx = canvas.getContext('2d');

    let W, H, dpr;
    let particles = [];
    let shootingStars = [];
    let nebulaBlobs = [];
    let mouse = { x: -9999, y: -9999, active: false };

    // Config
    const PARTICLE_COUNT = 800; // Increased for space dust
    const CONNECTION_DIST = 0; // Disabled connections for realism
    const MOUSE_RADIUS = 160;
    const SHOOTING_STAR_CHANCE = 0.0005; // Very rare
    const NEBULA_COUNT = 3; // Subtle nebula
    const SAT_LINE_COUNT = 0; // Removed for cleaner look

    // Section-aware color palette (deeper blues/purples)
    const sectionColors = [
        { r: 0, g: 212, b: 255, name: 'cyan' },       // Hero
        { r: 80, g: 160, b: 255, name: 'blue' },       // Foundation
        { r: 57, g: 255, b: 20, name: 'green' },       // Observation
        { r: 124, g: 58, b: 237, name: 'violet' },     // Results
    ];
    let currentColorIdx = 0;
    let targetColor = sectionColors[0];
    let activeColor = { r: 0, g: 212, b: 255 };

    function resize() {
        dpr = Math.min(window.devicePixelRatio, 2);
        W = window.innerWidth;
        H = window.innerHeight;
        canvas.width  = W * dpr;
        canvas.height = H * dpr;
        canvas.style.width  = W + 'px';
        canvas.style.height = H + 'px';
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function random(min, max) { return Math.random() * (max - min) + min; }
    function lerp(a, b, t) { return a + (b - a) * t; }

    function createParticle() {
        const depth = Math.pow(Math.random(), 3);  // Heavily skew towards 0 (distant space dust)
        const speedMult = 0.005 + depth * 0.015; // Slow movement
        
        // 3 depth layers
        let type;
        if (depth > 0.85) type = 'foreground'; // glowing bright
        else if (depth > 0.45) type = 'midfield';
        else type = 'distant'; // micro-stars

        return {
            x:  random(0, W),
            y:  random(0, H),
            vx: random(-speedMult, speedMult),
            vy: random(-speedMult, speedMult),
            r:  type === 'distant' ? random(0.1, 0.4) : (type === 'midfield' ? random(0.4, 1.0) : random(1.0, 1.8) + depth * 0.5),
            depth: depth,
            baseAlpha: type === 'distant' ? random(0.02, 0.08) : (type === 'midfield' ? random(0.05, 0.15) : random(0.1, 0.25)), // Reduced opacity by ~70%
            alpha: 0,
            twinkleSpeed: random(0.003, 0.015),
            twinkleOffset: random(0, Math.PI * 2),
            type: type,
            pulsePhase: Math.random() * Math.PI * 2,
        };
    }

    function createShootingStar() {
        const angle = random(-0.3, -0.8);
        const speed = random(10, 22);
        return {
            x: random(W * 0.2, W * 1.1),
            y: random(-30, H * 0.25),
            vx: Math.cos(angle) * speed,
            vy: -Math.sin(angle) * speed,
            length: random(80, 180),
            alpha: 1,
            life: 1,
            decay: random(0.008, 0.02),
        };
    }

    function createNebulaBlob() {
        return {
            x: random(0, W),
            y: random(0, H),
            vx: random(-0.08, 0.08),
            vy: random(-0.06, 0.06),
            rx: random(150, 400),
            ry: random(100, 300),
            alpha: random(0.008, 0.025),
            hue: random(200, 280), // blues/purples
            rotation: random(0, Math.PI * 2),
            rotSpeed: random(-0.0003, 0.0003),
        };
    }

    // Satellite trajectory lines
    const satLines = [];
    function createSatLine() {
        const isHorizontal = Math.random() > 0.5;
        return {
            x: isHorizontal ? -50 : random(0, W),
            y: isHorizontal ? random(H * 0.1, H * 0.9) : -50,
            vx: isHorizontal ? random(0.3, 0.8) : random(-0.1, 0.1),
            vy: isHorizontal ? random(-0.05, 0.05) : random(0.3, 0.8),
            length: random(80, 200),
            alpha: random(0.02, 0.06),
        };
    }

    function initParticles() {
        particles = [];
        for (let i = 0; i < PARTICLE_COUNT; i++) particles.push(createParticle());
        
        nebulaBlobs = [];
        for (let i = 0; i < NEBULA_COUNT; i++) nebulaBlobs.push(createNebulaBlob());

        satLines.length = 0;
        for (let i = 0; i < SAT_LINE_COUNT; i++) satLines.push(createSatLine());
    }

    function updateParticle(p, t) {
        const twinkle = 0.5 + 0.5 * Math.sin(t * p.twinkleSpeed + p.twinkleOffset);
        const pulse = 0.85 + 0.15 * Math.sin(t * 0.002 + p.pulsePhase);
        p.alpha = p.baseAlpha * twinkle * pulse;

        if (mouse.active) {
            const dx = p.x - mouse.x;
            const dy = p.y - mouse.y;
            const dist = Math.hypot(dx, dy);
            const effectiveRadius = MOUSE_RADIUS * (0.5 + p.depth * 0.8);

            if (dist < effectiveRadius && dist > 0) {
                const force = (effectiveRadius - dist) / effectiveRadius;
                const strength = force * force * 0.15 * p.depth;

                if (p.depth > 0.6) {
                    p.vx -= (dx / dist) * strength * 0.5;
                    p.vy -= (dy / dist) * strength * 0.5;
                } else {
                    p.vx += (dx / dist) * strength;
                    p.vy += (dy / dist) * strength;
                }
                p.alpha = Math.min(1, p.alpha + force * 0.3);
            }
        }

        p.vx *= 0.992;
        p.vy *= 0.992;

        const speed = Math.hypot(p.vx, p.vy);
        if (speed < 0.005) {
            const speedMult = 0.005 + p.depth * 0.015;
            p.vx = random(-speedMult, speedMult);
            p.vy = random(-speedMult, speedMult);
        }

        p.x += p.vx;
        p.y += p.vy;

        const margin = 20;
        if (p.x < -margin) p.x = W + margin;
        if (p.x > W + margin) p.x = -margin;
        if (p.y < -margin) p.y = H + margin;
        if (p.y > H + margin) p.y = -margin;
    }

    function drawParticle(p) {
        const ar = Math.round(lerp(activeColor.r, 255, p.depth * 0.3));
        const ag = Math.round(lerp(activeColor.g, 255, p.depth * 0.2));
        const ab = Math.round(lerp(activeColor.b, 255, p.depth * 0.1));

        if (p.type === 'foreground') {
            // Big glow halo
            const grd = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.r * 4);
            grd.addColorStop(0, `rgba(${ar}, ${ag}, ${ab}, ${p.alpha})`);
            grd.addColorStop(0.3, `rgba(${ar}, ${ag}, ${ab}, ${p.alpha * 0.4})`);
            grd.addColorStop(1, `rgba(${ar}, ${ag}, ${ab}, 0)`);
            ctx.fillStyle = grd;
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r * 4, 0, Math.PI * 2);
            ctx.fill();

            // Core
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(232, 244, 255, ${p.alpha})`;
            ctx.fill();
        } else if (p.type === 'midfield') {
            const grd = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.r * 2.5);
            grd.addColorStop(0, `rgba(${ar}, ${ag}, ${ab}, ${p.alpha * 0.8})`);
            grd.addColorStop(1, `rgba(${ar}, ${ag}, ${ab}, 0)`);
            ctx.fillStyle = grd;
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r * 2.5, 0, Math.PI * 2);
            ctx.fill();
        } else {
            // Distant micro-stars — simple dots
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(200, 220, 255, ${p.alpha * 0.6})`;
            ctx.fill();
        }
    }

    // Spawn a burst of tiny interactive star particles on text touch
    const burstParticles = [];
    function spawnBurst(x, y, count) {
        for (let i = 0; i < count; i++) {
            const angle = Math.random() * Math.PI * 2;
            const speed = random(1.5, 3.5);
            burstParticles.push({
                x: x, y: y,
                vx: Math.cos(angle) * speed,
                vy: Math.sin(angle) * speed,
                r: random(0.5, 1.5),
                alpha: 1.0,
                decay: random(0.02, 0.04),
                color: `rgba(0, 212, 255, ${random(0.7, 1)})`
            });
        }
    }

    // ---- Animation Loop ----
    let t = 0;

    function loop() {
        ctx.clearRect(0, 0, W, H);
        t++;

        // Smooth color transition
        activeColor.r = lerp(activeColor.r, targetColor.r, 0.02);
        activeColor.g = lerp(activeColor.g, targetColor.g, 0.02);
        activeColor.b = lerp(activeColor.b, targetColor.b, 0.02);

        // ---- Draw Nebula Cloud Blobs ----
        nebulaBlobs.forEach(nb => {
            nb.x += nb.vx;
            nb.y += nb.vy;
            nb.rotation += nb.rotSpeed;

            // Wrap
            if (nb.x < -nb.rx) nb.x = W + nb.rx;
            if (nb.x > W + nb.rx) nb.x = -nb.rx;
            if (nb.y < -nb.ry) nb.y = H + nb.ry;
            if (nb.y > H + nb.ry) nb.y = -nb.ry;

            ctx.save();
            ctx.translate(nb.x, nb.y);
            ctx.rotate(nb.rotation);
            const grd = ctx.createRadialGradient(0, 0, 0, 0, 0, nb.rx);
            grd.addColorStop(0, `hsla(${nb.hue}, 60%, 30%, ${nb.alpha})`);
            grd.addColorStop(0.5, `hsla(${nb.hue}, 50%, 20%, ${nb.alpha * 0.4})`);
            grd.addColorStop(1, `hsla(${nb.hue}, 40%, 10%, 0)`);
            ctx.fillStyle = grd;
            ctx.beginPath();
            ctx.ellipse(0, 0, nb.rx, nb.ry, 0, 0, Math.PI * 2);
            ctx.fill();
            ctx.restore();
        });

        // ---- Draw Faint Grid Overlay ----
        // Disabled for cinematic realism

        // ---- Draw Satellite Trajectory Lines ----
        satLines.forEach(sl => {
            sl.x += sl.vx;
            sl.y += sl.vy;

            // Reset when off screen
            if (sl.x > W + 200 || sl.y > H + 200 || sl.x < -200 || sl.y < -200) {
                Object.assign(sl, createSatLine());
            }

            ctx.beginPath();
            ctx.moveTo(sl.x, sl.y);
            ctx.lineTo(sl.x - sl.vx * sl.length, sl.y - sl.vy * sl.length);
            ctx.strokeStyle = `rgba(0, 200, 255, ${sl.alpha})`;
            ctx.lineWidth = 0.5;
            ctx.stroke();
        });

        // Dynamically scale connection distance when touching text
        const currentConnectionDist = mouse.hoveringText ? 240 : CONNECTION_DIST;

        // Draw connections
        for (let i = 0; i < particles.length; i++) {
            if (particles[i].depth < 0.3) continue;
            for (let j = i + 1; j < particles.length; j++) {
                if (particles[j].depth < 0.3) continue;
                const a = particles[i], b = particles[j];
                const dx = a.x - b.x;
                const dy = a.y - b.y;
                const dist = Math.hypot(dx, dy);

                const effectiveDist = currentConnectionDist * ((a.depth + b.depth) * 0.5);
                if (dist < effectiveDist) {
                    const alpha = (1 - dist / effectiveDist) * (mouse.hoveringText ? 0.45 : 0.2) * ((a.depth + b.depth) * 0.5);

                    const grad = ctx.createLinearGradient(a.x, a.y, b.x, b.y);
                    grad.addColorStop(0, `rgba(${activeColor.r}, ${activeColor.g}, ${activeColor.b}, ${alpha * a.alpha})`);
                    grad.addColorStop(1, `rgba(${activeColor.r}, ${activeColor.g}, ${activeColor.b}, ${alpha * b.alpha})`);

                    ctx.beginPath();
                    ctx.moveTo(a.x, a.y);
                    ctx.lineTo(b.x, b.y);
                    ctx.strokeStyle = grad;
                    ctx.lineWidth = mouse.hoveringText ? 0.75 : 0.5;
                    ctx.stroke();
                }
            }
        }

        // Draw main particles
        particles.forEach(p => {
            updateParticle(p, t);
            drawParticle(p);
        });

        // Update and draw burst particles
        for (let i = burstParticles.length - 1; i >= 0; i--) {
            const bp = burstParticles[i];
            bp.x += bp.vx;
            bp.y += bp.vy;
            bp.vx *= 0.95;
            bp.vy *= 0.95;
            bp.alpha -= bp.decay;

            if (bp.alpha <= 0) {
                burstParticles.splice(i, 1);
            } else {
                ctx.beginPath();
                ctx.arc(bp.x, bp.y, bp.r, 0, Math.PI * 2);
                ctx.fillStyle = bp.color.replace(/[\d\.]+\)$/, `${bp.alpha})`);
                ctx.fill();
            }
        }

        // ---- Shooting Stars ----
        if (Math.random() < SHOOTING_STAR_CHANCE) {
            shootingStars.push(createShootingStar());
        }

        for (let i = shootingStars.length - 1; i >= 0; i--) {
            const ss = shootingStars[i];
            ss.x += ss.vx;
            ss.y += ss.vy;
            ss.life -= ss.decay;

            if (ss.life <= 0) {
                shootingStars.splice(i, 1);
                continue;
            }

            // Draw trail
            const tailX = ss.x - (ss.vx / Math.hypot(ss.vx, ss.vy)) * ss.length;
            const tailY = ss.y - (ss.vy / Math.hypot(ss.vx, ss.vy)) * ss.length;

            const grad = ctx.createLinearGradient(ss.x, ss.y, tailX, tailY);
            grad.addColorStop(0, `rgba(255, 255, 255, ${ss.life * 0.9})`);
            grad.addColorStop(0.3, `rgba(0, 212, 255, ${ss.life * 0.5})`);
            grad.addColorStop(1, `rgba(0, 212, 255, 0)`);

            ctx.beginPath();
            ctx.moveTo(ss.x, ss.y);
            ctx.lineTo(tailX, tailY);
            ctx.strokeStyle = grad;
            ctx.lineWidth = 1.5;
            ctx.stroke();

            // Head glow
            ctx.beginPath();
            ctx.arc(ss.x, ss.y, 2, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(255, 255, 255, ${ss.life})`;
            ctx.fill();
        }

        requestAnimationFrame(loop);
    }

    // ---- Mouse Tracking ----
    window.addEventListener('mousemove', e => {
        mouse.x = e.clientX;
        mouse.y = e.clientY;
        mouse.active = true;

        const element = document.elementFromPoint(e.clientX, e.clientY);
        if (element && element.classList.contains('char-unit')) {
            mouse.hoveringText = true;
            spawnBurst(e.clientX, e.clientY, 3);
        } else {
            mouse.hoveringText = false;
        }
    });
    window.addEventListener('mouseleave', () => {
        mouse.active = false;
    });

    // ---- Section Color Updates ----
    function setSectionColor(index) {
        if (index >= 0 && index < sectionColors.length) {
            currentColorIdx = index;
            targetColor = sectionColors[index];
        }
    }

    // Watch for section changes
    const sectionObserver = new MutationObserver(() => {
        const sections = document.querySelectorAll('.section');
        sections.forEach((s, i) => {
            if (s.classList.contains('is-active')) {
                setSectionColor(i);
            }
        });
    });

    document.querySelectorAll('.section').forEach(s => {
        sectionObserver.observe(s, { attributes: true, attributeFilter: ['class'] });
    });

    // ---- Init ----
    window.addEventListener('resize', () => {
        resize();
        initParticles();
    });

    resize();
    initParticles();
    loop();

    // Expose for external use
    window.ParticleSystem = { setSectionColor };

})();
