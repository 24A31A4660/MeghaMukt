/* ============================================
   webgl-text.js — Resn WebGL Kinetic Shader Text Engine
   Custom ShaderMaterial with procedural noise waves,
   stroke-to-fill reveals, and 3D mouse distortion.
   ============================================ */

(function () {
    'use strict';

    let scene, camera, renderer, textMesh, material;
    let canvasElement;
    let mouse = { x: 0, y: 0, targetX: 0, targetY: 0 };
    let clock = new THREE.Clock();

    // Custom WebGL Shaders inspired by Resn's pipeline
    const vertexShader = `
        uniform float uTime;
        uniform vec2 uMouse;
        varying vec2 vUv;
        varying vec3 vPosition;
        varying vec3 vNormal;

        void main() {
            vUv = uv;
            vPosition = position;
            vNormal = normal;

            vec3 pos = position;
            
            // Interactive 3D mouse warp tilt
            float dist = length(vUv - vec2(0.5));
            pos.z += sin(dist * 10.0 - uTime * 3.0) * 0.08 * uMouse.x;
            pos.x += pos.y * uMouse.x * 0.05;
            pos.y += pos.x * uMouse.y * 0.05;

            gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
        }
    `;

    const fragmentShader = `
        uniform float uTime;
        uniform float uStrokeProgress;
        uniform float uFillProgress;
        uniform vec2 uMouse;
        uniform sampler2D uTextTexture;
        varying vec2 vUv;
        varying vec3 vPosition;

        void main() {
            vec4 textSample = texture2D(uTextTexture, vUv);
            if (textSample.a < 0.05) discard;

            // Liquid Noise Gradient Sweep Math
            float wave = sin(vUv.x * 12.0 + vUv.y * 8.0 + uTime * 2.5) * 0.5 + 0.5;
            float wave2 = cos(vUv.y * 15.0 - uTime * 1.8) * 0.5 + 0.5;

            vec3 cyan = vec3(0.0, 0.9, 1.0);
            vec3 green = vec3(0.22, 1.0, 0.08);
            vec3 white = vec3(1.0, 1.0, 1.0);
            vec3 purple = vec3(0.5, 0.1, 0.9);

            vec3 gradient = mix(cyan, green, wave);
            gradient = mix(gradient, white, wave2 * 0.4);
            gradient = mix(gradient, purple, sin(uTime + vUv.x * 5.0) * 0.2);

            // Edge stroke detection simulation
            float edge = smoothstep(0.4, 0.5, textSample.a) - smoothstep(0.5, 0.6, textSample.a);
            
            vec3 finalColor = mix(gradient, white, edge * 0.8);
            float alpha = textSample.a * smoothstep(0.0, 1.0, uFillProgress);

            gl_FragColor = vec4(finalColor, alpha);
        }
    `;

    function createTextTexture(textLines) {
        const textCanvas = document.createElement('canvas');
        textCanvas.width = 2048;
        textCanvas.height = 1024;
        const ctx = textCanvas.getContext('2d');

        ctx.clearRect(0, 0, textCanvas.width, textCanvas.height);
        ctx.fillStyle = '#ffffff';
        ctx.font = '900 130px "Syncopate", "Syne", sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        const startY = textCanvas.height / 2 - ((textLines.length - 1) * 140) / 2;
        textLines.forEach((line, i) => {
            ctx.fillText(line, textCanvas.width / 2, startY + i * 140);
        });

        const texture = new THREE.CanvasTexture(textCanvas);
        texture.minFilter = THREE.LinearFilter;
        texture.magFilter = THREE.LinearFilter;
        return texture;
    }

    function initWebGLText() {
        // Disabled: Using DOM text with CSS + GSAP TextReveal instead
        // The WebGL text overlay competed with the Earth globe positioning
        return;

        scene = new THREE.Scene();
        camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.z = 5;

        renderer = new THREE.WebGLRenderer({
            canvas: canvasElement,
            alpha: true,
            antialias: true,
            powerPreference: 'high-performance'
        });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

        const textTexture = createTextTexture(['BEYOND', 'THE CLOUDS']);

        material = new THREE.ShaderMaterial({
            vertexShader: vertexShader,
            fragmentShader: fragmentShader,
            uniforms: {
                uTime: { value: 0 },
                uStrokeProgress: { value: 1.0 },
                uFillProgress: { value: 1.0 },
                uMouse: { value: new THREE.Vector2(0, 0) },
                uTextTexture: { value: textTexture }
            },
            transparent: true,
            side: THREE.DoubleSide
        });

        const geometry = new THREE.PlaneGeometry(4.2, 2.1, 64, 64);
        textMesh = new THREE.Mesh(geometry, material);
        textMesh.position.set(-0.9, 0.2, 0); // Position on left side
        scene.add(textMesh);

        window.addEventListener('resize', onWindowResize);
        document.addEventListener('mousemove', onMouseMove);

        animate();
    }

    function onMouseMove(e) {
        mouse.targetX = (e.clientX / window.innerWidth) * 2 - 1;
        mouse.targetY = -(e.clientY / window.innerHeight) * 2 + 1;
    }

    function onWindowResize() {
        if (!camera || !renderer) return;
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    }

    function animate() {
        requestAnimationFrame(animate);
        const elapsedTime = clock.getElapsedTime();

        mouse.x += (mouse.targetX - mouse.x) * 0.05;
        mouse.y += (mouse.targetY - mouse.y) * 0.05;

        if (material) {
            material.uniforms.uTime.value = elapsedTime;
            material.uniforms.uMouse.value.set(mouse.x, mouse.y);
        }

        if (textMesh) {
            textMesh.rotation.y = mouse.x * 0.15;
            textMesh.rotation.x = -mouse.y * 0.15;
        }

        if (renderer && scene && camera) {
            renderer.render(scene, camera);
        }
    }

    window.WebGLTextEngine = {
        init: initWebGLText,
        triggerReveal: function () {
            if (!material) return;
            if (typeof gsap !== 'undefined') {
                gsap.fromTo(material.uniforms.uFillProgress, 
                    { value: 0.0 }, 
                    { value: 1.0, duration: 1.2, ease: 'expo.out' }
                );
            }
        }
    };

    if (document.readyState === 'complete' || document.readyState === 'interactive') {
        setTimeout(initWebGLText, 100);
    } else {
        window.addEventListener('DOMContentLoaded', () => setTimeout(initWebGLText, 100));
    }
})();
