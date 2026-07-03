/* ============================================
   earth.js — Cinematic Mission WebGL Globe V2
   One continuous camera journey. Earth is always
   visible. The story unfolds through the 3D scene.
   ============================================ */

(function () {
    'use strict';

    const container = document.getElementById('earth-canvas');
    if (!container) return;

    function waitForThree(cb) {
        if (typeof THREE !== 'undefined') return cb();
        const id = setInterval(() => {
            if (typeof THREE !== 'undefined') { clearInterval(id); cb(); }
        }, 50);
    }

    waitForThree(() => {

        /* ---------- Scene Setup ---------- */
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(42, window.innerWidth / window.innerHeight, 0.1, 1000);

        const renderer = new THREE.WebGLRenderer({
            canvas: container,
            alpha: true,
            antialias: true,
            powerPreference: 'high-performance',
        });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 1.1; // Reduced from 1.6 to remove excessive bloom

        /* ---------- Cinematic Camera Journey ---------- */
        // Each config is the camera state for each section
        const camJourney = [
            { z: 2.9, xOff: -1.28,  yOff: 0.0,  name: 'hero' },      // S01 (xOff reset, using CSS translateX instead)
            { z: 1.8, xOff: 0.0,    yOff: 0.0,  name: 'challenge' }, // S02 (Zoomed in, centered)
            { z: 1.6, xOff: -1.75,  yOff: 0.25, name: 'ai' },        // S03 (Significantly zoomed in on the right side)
            { z: 2.4, xOff: -0.30,  yOff: -0.1, name: 'complete' },  // S04 (Moved DOWN and RIGHT to perfectly center behind text)
        ];

        let targetCam = { z: 2.9, xOff: -1.28, yOff: 0.0 };
        let curCam    = { z: 2.9, xOff: -1.28, yOff: 0.0 };
        let currentSectionIdx = 0;
        let mouseX = 0, mouseY = 0;
        let scrollProgress = 0;

        document.addEventListener('mousemove', e => {
            mouseX = (e.clientX / window.innerWidth - 0.5) * 2;
            mouseY = (e.clientY / window.innerHeight - 0.5) * 2;
        });

        /* ---------- Lights ---------- */
        const ambientLight = new THREE.AmbientLight(0x1a2e4c, 0.35); // Toned down from 0.55
        scene.add(ambientLight);
        const sunLight = new THREE.DirectionalLight(0xffffff, 2.2); // Toned down from 2.8
        sunLight.position.set(-10, 3, 8);
        scene.add(sunLight);
        const earthShine = new THREE.DirectionalLight(0xdbeafe, 0.35); // Toned down from 0.55
        earthShine.position.set(10, -3, -8);
        scene.add(earthShine);
        // Warm fill light (opposite sun)
        const warmFill = new THREE.DirectionalLight(0xffd4a0, 0.25);
        warmFill.position.set(8, -2, -6);
        scene.add(warmFill);

        /* ---------- Texture Loading ---------- */
        const loader = new THREE.TextureLoader();
        loader.crossOrigin = 'anonymous';
        const dayTex  = loader.load('assets/earth-blue-marble.jpg');
        const nightTex = loader.load('assets/earth-night.jpg');
        const bumpTex  = loader.load('assets/earth-topology.png');
        const specTex  = loader.load('assets/earth-water.png');
        [dayTex, nightTex, bumpTex, specTex].forEach(t => {
            t.minFilter = THREE.LinearFilter;
            t.magFilter = THREE.LinearFilter;
        });

        /* ---------- Procedural Clouds (canvas-based) ---------- */
        function makeCloudCanvas(density = 450, maxAlpha = 0.14) {
            const W = 4096, H = 2048;
            const c = document.createElement('canvas');
            c.width = W; c.height = H;
            const g = c.getContext('2d');
            g.clearRect(0, 0, W, H);
            for (let i = 0; i < density; i++) {
                const cx = Math.random() * W;
                const lat = H * 0.1 + Math.random() * H * 0.8;
                const r = 20 + Math.random() * 90;
                const alpha = 0.03 + Math.random() * maxAlpha;
                const grad = g.createRadialGradient(cx, lat, 0, cx, lat, r);
                grad.addColorStop(0, `rgba(255,255,255,${alpha})`);
                grad.addColorStop(0.5, `rgba(255,255,255,${alpha * 0.3})`);
                grad.addColorStop(1, 'rgba(255,255,255,0)');
                g.fillStyle = grad;
                g.beginPath(); g.arc(cx, lat, r, 0, Math.PI * 2); g.fill();
            }
            return new THREE.CanvasTexture(c);
        }

        /* ---------- Earth ---------- */
        const earthGeo = new THREE.SphereGeometry(0.85, 96, 96);
        const earthMat = new THREE.ShaderMaterial({
            uniforms: {
                uDay:   { value: dayTex },
                uNight: { value: nightTex },
                uBump:  { value: bumpTex },
                uSpec:  { value: specTex },
                uSun:   { value: sunLight.position.clone().normalize() },
                uTime:  { value: 0 },
                uClearProgress: { value: 0.0 }, // 0=cloudy, 1=clear (S04)
            },
            vertexShader: `
                varying vec2 vUv;
                varying vec3 vNormal;
                varying vec3 vPos;
                varying vec3 vTangent;
                varying vec3 vBitangent;
                void main() {
                    vUv = uv;
                    vNormal = normalize(normalMatrix * normal);
                    vPos = (modelViewMatrix * vec4(position, 1.0)).xyz;
                    vec3 c1 = cross(normal, vec3(0.0, 0.0, 1.0));
                    vec3 c2 = cross(normal, vec3(0.0, 1.0, 0.0));
                    vTangent = normalize(normalMatrix * (length(c1) > length(c2) ? c1 : c2));
                    vBitangent = normalize(cross(vNormal, vTangent));
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform sampler2D uDay;
                uniform sampler2D uNight;
                uniform sampler2D uBump;
                uniform sampler2D uSpec;
                uniform vec3 uSun;
                uniform float uTime;
                uniform float uClearProgress;
                varying vec2 vUv;
                varying vec3 vNormal;
                varying vec3 vPos;
                varying vec3 vTangent;
                varying vec3 vBitangent;
                void main() {
                    vec3 normal = normalize(vNormal);
                    float centerH = texture2D(uBump, vUv).r;
                    float rightH  = texture2D(uBump, vUv + vec2(0.001, 0.0)).r;
                    float topH    = texture2D(uBump, vUv + vec2(0.0, 0.001)).r;
                    float bumpScale = 0.025;
                    vec3 bumpN = normal + bumpScale * ((centerH - rightH) * vTangent + (centerH - topH) * vBitangent);
                    bumpN = normalize(bumpN);
                    float sunDot = dot(bumpN, uSun);
                    float dayStrength = smoothstep(-0.15, 0.28, sunDot);
                    vec4 dayColor = texture2D(uDay, vUv);
                    vec4 nightColor = texture2D(uNight, vUv);
                    vec4 specMask = texture2D(uSpec, vUv);
                    float diffuse = max(0.0, sunDot) * 0.85 + 0.15;
                    dayColor.rgb *= diffuse;
                    // Brighter when cleared (S04)
                    dayColor.rgb *= (1.0 + uClearProgress * 0.35);
                    vec3 finalColor = mix(nightColor.rgb * 3.2, dayColor.rgb, dayStrength);
                    vec3 viewDir = normalize(-vPos);
                    vec3 reflectDir = reflect(-uSun, bumpN);
                    float spec = pow(max(0.0, dot(reflectDir, viewDir)), 48.0);
                    finalColor += vec3(0.55, 0.72, 1.0) * spec * (specMask.r * 0.55);
                    float fresnel = pow(1.0 - max(dot(viewDir, normal), 0.0), 3.5);
                    // Soft, premium atmosphere glow
                    finalColor += vec3(0.35, 0.7, 0.95) * fresnel * (0.8 + uClearProgress * 0.2);
                    gl_FragColor = vec4(finalColor, 1.0);
                }
            `,
        });
        const earth = new THREE.Mesh(earthGeo, earthMat);
        scene.add(earth);

        /* ---------- Atmosphere ---------- */
        const atmoGeo = new THREE.SphereGeometry(0.96, 64, 64);
        const atmoMat = new THREE.ShaderMaterial({
            uniforms: {
                uSun: { value: earthMat.uniforms.uSun.value },
                uAtmoIntensity: { value: 1.0 },
            },
            vertexShader: `
                varying vec3 vN; varying vec3 vP;
                void main() {
                    vN = normalize(normalMatrix * normal);
                    vP = (modelViewMatrix * vec4(position, 1.0)).xyz;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform vec3 uSun; uniform float uAtmoIntensity;
                varying vec3 vN; varying vec3 vP;
                void main() {
                    vec3 n = normalize(vN); vec3 V = normalize(-vP);
                    float f = pow(1.0 - max(dot(V, n), 0.0), 3.2);
                    float sun = max(dot(n, uSun), 0.0) * 0.3 + 0.7;
                    vec3 col = mix(vec3(0.02, 0.12, 0.32), vec3(0.4, 0.7, 1.0), f) * sun;
                    gl_FragColor = vec4(col, f * 0.18 * uAtmoIntensity);
                }
            `,
            transparent: true, side: THREE.BackSide, depthWrite: false,
        });
        const atmo = new THREE.Mesh(atmoGeo, atmoMat);
        scene.add(atmo);

        /* ---------- Outer Glow Ring (Mission Complete) ---------- */
        const glowRingGeo = new THREE.SphereGeometry(0.98, 64, 64);
        const glowRingMat = new THREE.ShaderMaterial({
            uniforms: { uIntensity: { value: 0.0 }, uTime: { value: 0.0 } },
            vertexShader: `
                varying vec3 vN; varying vec3 vP;
                void main() {
                    vN = normalize(normalMatrix * normal);
                    vP = (modelViewMatrix * vec4(position, 1.0)).xyz;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform float uIntensity; uniform float uTime;
                varying vec3 vN; varying vec3 vP;
                void main() {
                    vec3 V = normalize(-vP);
                    float f = pow(1.0 - max(dot(V, normalize(vN)), 0.0), 2.5);
                    float pulse = 0.8 + 0.2 * sin(uTime * 2.0);
                    vec3 aurora = mix(vec3(0.0, 0.9, 1.0), vec3(0.22, 1.0, 0.08), f * 0.5);
                    gl_FragColor = vec4(aurora, f * uIntensity * pulse * 0.3);
                }
            `,
            transparent: true, side: THREE.BackSide, depthWrite: false,
            blending: THREE.AdditiveBlending,
        });
        const glowRing = new THREE.Mesh(glowRingGeo, glowRingMat);
        scene.add(glowRing);

        /* ---------- Cloud Layer ---------- */
        const cloudGeo = new THREE.SphereGeometry(0.86, 64, 64);
        const cloudMat = new THREE.MeshBasicMaterial({
            map: makeCloudCanvas(450, 0.14),
            transparent: true, opacity: 0.35,
            depthWrite: false, blending: THREE.AdditiveBlending,
        });
        const clouds = new THREE.Mesh(cloudGeo, cloudMat);
        scene.add(clouds);

        /* ---------- Thick Cloud Pulse Layer (S02 - Challenge) ---------- */
        const cloudPulseGeo = new THREE.SphereGeometry(0.868, 48, 48);
        const cloudPulseMat = new THREE.MeshBasicMaterial({
            map: makeCloudCanvas(300, 0.22),
            transparent: true, opacity: 0.0,
            depthWrite: false, blending: THREE.AdditiveBlending,
        });
        const cloudPulse = new THREE.Mesh(cloudPulseGeo, cloudPulseMat);
        scene.add(cloudPulse);

        /* ---------- AI Wireframe Grid (S03) ---------- */
        const gridGeo = new THREE.SphereGeometry(1.005, 24, 24); // Thinner line density
        const gridMat = new THREE.MeshBasicMaterial({
            color: 0x00aaff, wireframe: true, // Softer cyan
            transparent: true, opacity: 0.0,
            blending: THREE.AdditiveBlending, depthWrite: false,
        });
        const gridMesh = new THREE.Mesh(gridGeo, gridMat);
        scene.add(gridMesh);

        /* ---------- Hexagonal AI Node System (S03) ---------- */
        function createHexNode(color, size = 0.015) {
            const geo = new THREE.OctahedronGeometry(size);
            const mat = new THREE.MeshBasicMaterial({
                color, transparent: true, opacity: 0.0,
                blending: THREE.AdditiveBlending, depthWrite: false,
            });
            return new THREE.Mesh(geo, mat);
        }

        const aiNodes = [];
        const aiNodePositions = [];
        for (let i = 0; i < 12; i++) {
            const phi = Math.acos(-1 + (2 * i) / 12);
            const theta = Math.sqrt(12 * Math.PI) * phi;
            const r = 1.08;
            const x = r * Math.sin(phi) * Math.cos(theta);
            const y = r * Math.sin(phi) * Math.sin(theta);
            const z = r * Math.cos(phi);
            const node = createHexNode(i % 2 === 0 ? 0x00e5ff : 0x39ff14, 0.018);
            node.position.set(x, y, z);
            scene.add(node);
            aiNodes.push(node);
            aiNodePositions.push(new THREE.Vector3(x, y, z));
        }

        /* ---------- Data Flow Particles (S03 — satellite → Earth) ---------- */
        const dataParticleCount = 60;
        const dataParticlePositions = new Float32Array(dataParticleCount * 3);
        const dataParticleLife = [];
        const dataParticleGeo = new THREE.BufferGeometry();
        dataParticleGeo.setAttribute('position', new THREE.BufferAttribute(dataParticlePositions, 3));
        const dataParticleMat = new THREE.PointsMaterial({
            color: 0x00e5ff, size: 0.012,
            transparent: true, opacity: 0.0,
            blending: THREE.AdditiveBlending, depthWrite: false,
        });
        const dataParticles = new THREE.Points(dataParticleGeo, dataParticleMat);
        scene.add(dataParticles);

        for (let i = 0; i < dataParticleCount; i++) {
            dataParticleLife.push({ t: Math.random(), speed: 0.003 + Math.random() * 0.005, orbitIdx: Math.floor(Math.random() * 3) });
        }

        /* ---------- Aurora Victory Particles (S04) ---------- */
        const auroraCount = 200;
        const auroraPositions = new Float32Array(auroraCount * 3);
        const auroraGeo = new THREE.BufferGeometry();
        auroraGeo.setAttribute('position', new THREE.BufferAttribute(auroraPositions, 3));
        const auroraMat = new THREE.PointsMaterial({
            color: 0x39ff14, size: 0.01,
            transparent: true, opacity: 0.0,
            blending: THREE.AdditiveBlending, depthWrite: false,
        });
        const auroraParticles = new THREE.Points(auroraGeo, auroraMat);
        scene.add(auroraParticles);

        for (let i = 0; i < auroraCount; i++) {
            const theta = Math.random() * Math.PI * 2;
            const phi = (Math.random() - 0.5) * 0.6; // near equator
            const r = 1.15 + Math.random() * 0.5;
            auroraPositions[i * 3]     = r * Math.cos(theta) * Math.cos(phi);
            auroraPositions[i * 3 + 1] = r * Math.sin(phi);
            auroraPositions[i * 3 + 2] = r * Math.sin(theta) * Math.cos(phi);
        }
        auroraGeo.attributes.position.needsUpdate = true;

        /* ---------- Space Dust ---------- */
        const particleCount = 200;
        const particleGeo = new THREE.BufferGeometry();
        const positions = new Float32Array(particleCount * 3);
        for (let i = 0; i < particleCount; i++) {
            const r = 1.25 + Math.random() * 0.85;
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos(Math.random() * 2 - 1);
            positions[i * 3]     = r * Math.sin(phi) * Math.cos(theta);
            positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
            positions[i * 3 + 2] = r * Math.cos(phi);
        }
        particleGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        const particleMat = new THREE.PointsMaterial({
            color: 0x00e5ff, size: 0.008,
            transparent: true, opacity: 0.45,
            blending: THREE.AdditiveBlending, depthWrite: false,
        });
        const spaceDust = new THREE.Points(particleGeo, particleMat);
        scene.add(spaceDust);

        /* ---------- Radar Scan Ring (S02) ---------- */
        const scanRingGeo = new THREE.TorusGeometry(1.05, 0.0015, 8, 100);
        const scanRingMat = new THREE.MeshBasicMaterial({
            color: 0x00e5ff, transparent: true, opacity: 0.0,
            blending: THREE.AdditiveBlending, depthWrite: false,
        });
        const scanRing = new THREE.Mesh(scanRingGeo, scanRingMat);
        scene.add(scanRing);

        const scanRing2 = scanRing.clone();
        scanRing2.scale.setScalar(0.8);
        scanRing2.rotation.x = Math.PI / 3;
        scene.add(scanRing2);

        /* ---------- 3D Satellite Builder ---------- */
        function create3DSatellite(colorHex) {
            const satGroup = new THREE.Group();
            const body = new THREE.Mesh(
                new THREE.BoxGeometry(0.038, 0.038, 0.038),
                new THREE.MeshStandardMaterial({ color: 0xdfb76c, metalness: 0.95, roughness: 0.15 })
            );
            satGroup.add(body);
            const wingMat = new THREE.MeshStandardMaterial({ color: 0x1d4ed8, metalness: 0.85, roughness: 0.08 });
            const leftWing = new THREE.Mesh(new THREE.BoxGeometry(0.09, 0.024, 0.004), wingMat);
            leftWing.position.x = -0.07; satGroup.add(leftWing);
            const rightWing = leftWing.clone(); rightWing.position.x = 0.07; satGroup.add(rightWing);
            const dish = new THREE.Mesh(
                new THREE.ConeGeometry(0.015, 0.015, 12),
                new THREE.MeshStandardMaterial({ color: 0xdddddd, metalness: 0.9, roughness: 0.1 })
            );
            dish.position.y = -0.025; dish.rotation.x = Math.PI; satGroup.add(dish);
            // Scan beam
            const beamGeo = new THREE.CylinderGeometry(0.001, 0.09, 0.45, 16, 1, true);
            const beamMat = new THREE.MeshBasicMaterial({
                color: colorHex, transparent: true, opacity: 0.0,
                blending: THREE.AdditiveBlending, side: THREE.DoubleSide, depthWrite: false,
            });
            const beam = new THREE.Mesh(beamGeo, beamMat);
            beam.position.y = -0.25;
            satGroup.add(beam);
            // Blinking navigation light
            const navLightMat = new THREE.MeshBasicMaterial({ color: 0xff0033, transparent: true, opacity: 1.0 });
            const navLight = new THREE.Mesh(new THREE.SphereGeometry(0.007, 8, 8), navLightMat);
            navLight.position.set(0.08, 0.012, 0);
            satGroup.add(navLight);
            satGroup.userData = { beam, navLight, lightPhase: Math.random() * Math.PI * 2 };
            return satGroup;
        }

        /* ---------- Flowing Orbit System ---------- */
        const orbits = [];
        const orbitConfigs = [
            { r: 1.42, tilt: 22,  speed: 0.002,  n: 3, col: 0x00e5ff, density: 160 },
            { r: 1.72, tilt: -32, speed: -0.0015, n: 2, col: 0x39ff14, density: 200 },
            { r: 2.08, tilt: 48,  speed: 0.0025,  n: 1, col: 0xff6b35, density: 240 },
        ];

        orbitConfigs.forEach(cfg => {
            const grp = new THREE.Group();
            grp.rotation.x = cfg.tilt * Math.PI / 180;
            const dustCount = cfg.density;
            const dustPositions = new Float32Array(dustCount * 3);
            const dustAngles = [];
            for (let i = 0; i < dustCount; i++) {
                const angle = (i / dustCount) * Math.PI * 2;
                dustAngles.push(angle);
                const rn = cfg.r + (Math.random() - 0.5) * 0.012;
                dustPositions[i * 3]     = rn * Math.cos(angle);
                dustPositions[i * 3 + 1] = (Math.random() - 0.5) * 0.01;
                dustPositions[i * 3 + 2] = rn * Math.sin(angle);
            }
            const dustGeo = new THREE.BufferGeometry();
            dustGeo.setAttribute('position', new THREE.BufferAttribute(dustPositions, 3));
            const dustMat = new THREE.PointsMaterial({
                color: cfg.col, size: 0.008, transparent: true, opacity: 0.35,
                blending: THREE.AdditiveBlending, depthWrite: false,
            });
            const dustPoints = new THREE.Points(dustGeo, dustMat);
            grp.add(dustPoints);

            const sats = [];
            for (let i = 0; i < cfg.n; i++) {
                const ang = (i / cfg.n) * Math.PI * 2;
                const sg = create3DSatellite(cfg.col);
                const trailCount = 35;
                const trailPositions = new Float32Array(trailCount * 3);
                const trailColArray = new Float32Array(trailCount * 3);
                const baseColor = new THREE.Color(cfg.col);
                for (let k = 0; k < trailCount; k++) {
                    trailPositions[k * 3] = cfg.r * Math.cos(ang);
                    trailPositions[k * 3 + 1] = 0;
                    trailPositions[k * 3 + 2] = cfg.r * Math.sin(ang);
                    const ratio = 1 - k / trailCount;
                    trailColArray[k * 3]     = baseColor.r * ratio;
                    trailColArray[k * 3 + 1] = baseColor.g * ratio;
                    trailColArray[k * 3 + 2] = baseColor.b * ratio;
                }
                const trailGeo = new THREE.BufferGeometry();
                trailGeo.setAttribute('position', new THREE.BufferAttribute(trailPositions, 3));
                trailGeo.setAttribute('color', new THREE.BufferAttribute(trailColArray, 3));
                const trailMat = new THREE.PointsMaterial({
                    size: 0.007, vertexColors: true,
                    transparent: true, opacity: 0.3,
                    blending: THREE.AdditiveBlending, depthWrite: false,
                });
                const trail = new THREE.Points(trailGeo, trailMat);
                grp.add(trail);
                sg.position.set(cfg.r * Math.cos(ang), 0, cfg.r * Math.sin(ang));
                sats.push({ group: sg, angle: ang, speed: cfg.speed, trail, trailPositions, trailIdx: 0, orbitRadius: cfg.r });
                grp.add(sg);
            }
            orbits.push({ group: grp, sats, radius: cfg.r, dustPoints, dustAngles, color: cfg.col });
            scene.add(grp);
        });

        /* ---------- Resize Handler ---------- */
        function onResize() {
            const w = window.innerWidth, h = window.innerHeight;
            camera.aspect = w / h;
            camera.updateProjectionMatrix();
            renderer.setSize(w, h);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        }
        window.addEventListener('resize', onResize);

        /* ---------- State per section ---------- */
        let targetCloudOpacity = 0.35;
        let targetCloudPulseOpacity = 0.0;
        let targetGridOpacity = 0.0;
        let targetBeamOpacity = 0.0;
        let targetDataOpacity = 0.0;
        let targetAuroraOpacity = 0.0;
        let targetGlowIntensity = 0.0;
        let targetScanOpacity = 0.0;
        let targetAtmoIntensity = 1.0;
        let targetClearProgress = 0.0;

        /* ---------- Section → Scene State Map ---------- */
        function setSceneForSection(idx) {
            currentSectionIdx = idx;
            const cfg = camJourney[Math.min(idx, camJourney.length - 1)];
            targetCam.z    = cfg.z;
            targetCam.xOff = cfg.xOff;
            targetCam.yOff = cfg.yOff;
            
            let camDur = 2.5;
            if (idx === 2) camDur = 1.5; // ~40% faster for Section 03
            if (idx === 3) camDur = 4.0; // Slower cinematic reveal for Section 04

            if (typeof gsap !== 'undefined') {
                gsap.to(curCam, {
                    z: cfg.z,
                    xOff: cfg.xOff,
                    yOff: cfg.yOff,
                    duration: camDur,
                    ease: 'power3.inOut',
                    overwrite: 'auto'
                });
            }

            const canvasEl = document.getElementById('earth-canvas');
            if (canvasEl) {
                canvasEl.style.transition = 'transform ' + camDur + 's cubic-bezier(0.65, 0, 0.35, 1)';
                if (idx === 0) {
                    canvasEl.style.transform = 'translateX(180px)';
                } else {
                    canvasEl.style.transform = 'translateX(0px)';
                }
            }

            switch (idx) {
                case 0: // HERO — full orbit, normal clouds
                    targetCloudOpacity      = 0.35;
                    targetCloudPulseOpacity = 0.0;
                    targetGridOpacity       = 0.0;
                    targetBeamOpacity       = 0.0;
                    targetDataOpacity       = 0.0;
                    targetAuroraOpacity     = 0.0;
                    targetGlowIntensity     = 0.0;
                    targetScanOpacity       = 0.0;
                    targetAtmoIntensity     = 1.0;
                    targetClearProgress     = 0.0;
                    break;

                case 1: // CHALLENGE — thick clouds, scan beams active, radar
                    targetCloudOpacity      = 0.55;
                    targetCloudPulseOpacity = 0.25;
                    targetGridOpacity       = 0.08;
                    targetBeamOpacity       = 0.2;
                    targetDataOpacity       = 0.0;
                    targetAuroraOpacity     = 0.0;
                    targetGlowIntensity     = 0.0;
                    targetScanOpacity       = 0.25;
                    targetAtmoIntensity     = 1.2;
                    targetClearProgress     = 0.0;
                    break;

                case 2: // AI — wireframe grid, data flow, clouds dissolving
                    targetCloudOpacity      = 0.22;
                    targetCloudPulseOpacity = 0.0;
                    targetGridOpacity       = 0.06; // reduced ~70%
                    targetBeamOpacity       = 0.05; // reduced ~70%
                    targetDataOpacity       = 0.2;  // reduced ~70%
                    targetAuroraOpacity     = 0.0;
                    targetGlowIntensity     = 0.0;
                    targetScanOpacity       = 0.03; // reduced ~70%
                    targetAtmoIntensity     = 1.0;
                    targetClearProgress     = 0.4;
                    break;

                case 3: // COMPLETE — aurora, clear Earth, victory
                    targetCloudOpacity      = 0.05;
                    targetCloudPulseOpacity = 0.0;
                    targetGridOpacity       = 0.0;
                    targetBeamOpacity       = 0.0;
                    targetDataOpacity       = 0.0;
                    targetAuroraOpacity     = 0.45;
                    targetGlowIntensity     = 0.6;
                    targetScanOpacity       = 0.0;
                    targetAtmoIntensity     = 1.1;
                    targetClearProgress     = 1.0;
                    break;
            }
        }

        /* ---------- Render Loop ---------- */
        const clock = new THREE.Clock();

        function animate() {
            requestAnimationFrame(animate);
            const t = clock.getElapsedTime();

            // Camera is now animated smoothly via GSAP in setSceneForSection
            // Camera perfectly static, no breathing or mouse movement
            camera.position.x = curCam.xOff;
            camera.position.y = curCam.yOff;
            camera.position.z = curCam.z;
            camera.lookAt(curCam.xOff * 0.25, curCam.yOff * 0.25, 0);

            // Earth rotation — constant automatic rotation at 0.16
            const rotSpeed = 0.16;
            earth.rotation.y = t * rotSpeed;
            earth.rotation.x = 0;
            clouds.rotation.y = t * rotSpeed * 0.65;
            clouds.rotation.x = 0;
            cloudPulse.rotation.y = t * rotSpeed * 0.75;
            cloudPulse.rotation.x = 0;

            // Scale based on scroll progress (gentle)
            const s = 1.0 - scrollProgress * 0.15;
            earth.scale.setScalar(s);
            atmo.scale.setScalar(s * 1.09);
            clouds.scale.setScalar(s * 1.012);
            cloudPulse.scale.setScalar(s * 1.022);
            glowRing.scale.setScalar(s * 1.18);

            // Smooth opacity transitions - highly accelerated for Section 03 responsiveness
            let oEase = 0.04;
            if (currentSectionIdx === 2) oEase = 0.12;
            if (currentSectionIdx === 3) oEase = 0.015; // Extremely slow fade for cinematic reveal
            clouds.material.opacity      += (targetCloudOpacity - clouds.material.opacity) * oEase;
            cloudPulse.material.opacity  += (targetCloudPulseOpacity - cloudPulse.material.opacity) * oEase;
            
            // Cloud pulse animation (S02 scanner effect)
            if (cloudPulse.material.opacity > 0.01) {
                cloudPulse.material.opacity = targetCloudPulseOpacity * (0.7 + 0.3 * Math.sin(t * 1.5));
            }

            // AI grid
            let gridTarget = targetGridOpacity * (0.55 + 0.45 * Math.sin(t * 6 + 1.0));
            gridMesh.material.opacity += (gridTarget - gridMesh.material.opacity) * oEase;
            gridMesh.scale.setScalar(s * 1.005);
            gridMesh.rotation.y = t * -0.05;
            gridMesh.rotation.x = t * 0.015;

            // Scan rings (S02)
            scanRing.material.opacity  += (targetScanOpacity * (0.5 + 0.5 * Math.sin(t * 2.5)) - scanRing.material.opacity) * oEase;
            scanRing2.material.opacity += (targetScanOpacity * (0.5 + 0.5 * Math.sin(t * 2.5 + 1)) - scanRing2.material.opacity) * oEase;
            scanRing.rotation.y  = t * 0.3;
            scanRing.rotation.z  = t * 0.15;
            scanRing2.rotation.y = t * -0.25;
            scanRing2.rotation.x = t * 0.1;

            // AI nodes (S03)
            aiNodes.forEach((node, i) => {
                node.material.opacity += (targetGridOpacity * 0.9 - node.material.opacity) * (oEase * 0.5);
                node.rotation.x = t * (0.5 + i * 0.1);
                node.rotation.y = t * (0.3 + i * 0.07);
                const pulseFactor = 1.0 + 0.3 * Math.sin(t * 3 + i * 0.8);
                node.scale.setScalar(pulseFactor * s);
            });

            // Data flow particles (S03)
            dataParticleMat.opacity += (targetDataOpacity - dataParticleMat.opacity) * oEase;
            if (dataParticleMat.opacity > 0.01) {
                const dPos = dataParticleGeo.attributes.position.array;
                dataParticleLife.forEach((p, i) => {
                    p.t += p.speed;
                    if (p.t > 1.0) p.t = 0.0;
                    const orb = orbits[p.orbitIdx % orbits.length];
                    const sat = orb.sats[0];
                    const satPos = sat.group.position;
                    // Lerp from satellite toward Earth
                    const lerpT = p.t;
                    dPos[i * 3]     = satPos.x * (1 - lerpT) * s;
                    dPos[i * 3 + 1] = satPos.y * (1 - lerpT) * s;
                    dPos[i * 3 + 2] = satPos.z * (1 - lerpT) * s;
                });
                dataParticleGeo.attributes.position.needsUpdate = true;
            }

            // Aurora victory particles (S04)
            auroraMat.opacity += (targetAuroraOpacity - auroraMat.opacity) * (oEase * 0.5);
            if (auroraMat.opacity > 0.01) {
                const aPos = auroraGeo.attributes.position.array;
                for (let i = 0; i < auroraCount; i++) {
                    const theta = (i / auroraCount) * Math.PI * 2 + t * 0.08;
                    const phi = Math.sin(t * 0.5 + i * 0.3) * 0.4;
                    const r = 1.2 + Math.sin(t * 1.5 + i * 0.5) * 0.15;
                    aPos[i * 3]     = r * Math.cos(theta) * Math.cos(phi) * s;
                    aPos[i * 3 + 1] = r * Math.sin(phi) * s;
                    aPos[i * 3 + 2] = r * Math.sin(theta) * Math.cos(phi) * s;
                }
                auroraGeo.attributes.position.needsUpdate = true;
            }

            // Outer glow ring (S04)
            glowRingMat.uniforms.uIntensity.value += (targetGlowIntensity - glowRingMat.uniforms.uIntensity.value) * (oEase * 0.5);
            glowRingMat.uniforms.uTime.value = t;

            // Atmosphere intensity
            atmoMat.uniforms.uAtmoIntensity.value += (targetAtmoIntensity - atmoMat.uniforms.uAtmoIntensity.value) * oEase;

            // Earth clear progress
            earthMat.uniforms.uClearProgress.value += (targetClearProgress - earthMat.uniforms.uClearProgress.value) * (oEase * 0.3);
            earthMat.uniforms.uTime.value = t;

            // Satellite beams
            orbits.forEach(o => {
                const pos = o.dustPoints.geometry.attributes.position.array;
                for (let i = 0; i < o.dustAngles.length; i++) {
                    o.dustAngles[i] += o.sats[0].speed * 0.2;
                    const ang = o.dustAngles[i];
                    pos[i * 3]     = o.radius * Math.cos(ang) * s;
                    pos[i * 3 + 2] = o.radius * Math.sin(ang) * s;
                }
                o.dustPoints.geometry.attributes.position.needsUpdate = true;

                o.sats.forEach(sat => {
                    sat.angle += sat.speed;
                    const x = o.radius * Math.cos(sat.angle) * s;
                    const z = o.radius * Math.sin(sat.angle) * s;
                    sat.group.position.set(x, 0, z);
                    sat.group.lookAt(0, 0, 0);
                    sat.group.rotateX(Math.PI / 2);
                    if (sat.group.userData && sat.group.userData.beam) {
                        const beamOpacity = targetBeamOpacity * (0.5 + 0.5 * Math.sin(t * 8 + sat.angle));
                        sat.group.userData.beam.material.opacity += (beamOpacity - sat.group.userData.beam.material.opacity) * oEase;
                    }
                    if (sat.group.userData && sat.group.userData.navLight) {
                        sat.group.userData.navLight.material.opacity = 0.2 + 0.8 * Math.pow(Math.sin(t * 3.0 + sat.group.userData.lightPhase), 4);
                    }
                    // Trail
                    const idx = sat.trailIdx % (sat.trailPositions.length / 3);
                    sat.trailPositions[idx * 3]     = x;
                    sat.trailPositions[idx * 3 + 1] = (Math.random() - 0.5) * 0.005;
                    sat.trailPositions[idx * 3 + 2] = z;
                    sat.trail.geometry.attributes.position.needsUpdate = true;
                    sat.trailIdx++;
                });
            });

            // Space dust ambient motion
            spaceDust.rotation.y = t * 0.012;
            spaceDust.rotation.x = t * 0.006;

            renderer.render(scene, camera);
        }
        animate();

        /* ---------- Public API ---------- */
        window.EarthRenderer = {
            setScrollProgress(p) {
                scrollProgress = p;
            },
            setSection(idx) {
                setSceneForSection(idx);
            },
            getSection() {
                return currentSectionIdx;
            }
        };

        // Init with section 0
        setSceneForSection(0);

    }); // end waitForThree
})();