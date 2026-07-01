/**
 * SPA Router and Animation Controller for MeghaMukt
 */
document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    const authLayer = document.getElementById('spa-auth-layer');
    const dashboardLayer = document.getElementById('spa-dashboard-layer');
    
    const loginModal = document.getElementById('auth-modal-login');
    const signupModal = document.getElementById('auth-modal-signup');
    const launchModal = document.getElementById('auth-modal-launch');

    const heroContent = document.querySelector('.section__content');
    const mainNav = document.getElementById('main-nav');
    
    let currentModal = null;

    // Helper: Close active modal
    const closeModals = (onComplete) => {
        if (!currentModal) {
            if(onComplete) onComplete();
            return;
        }
        
        gsap.to(currentModal, {
            opacity: 0,
            y: 30,
            scale: 0.95,
            duration: 0.4,
            ease: "power2.in",
            onComplete: () => {
                currentModal.style.visibility = 'hidden';
                currentModal = null;
                authLayer.classList.remove('active');
                if(onComplete) onComplete();
            }
        });
    };

    // Helper: Open specific modal
    const openModal = (modalEl) => {
        if (currentModal === modalEl) return;
        
        const showNewModal = () => {
            currentModal = modalEl;
            authLayer.classList.add('active');
            modalEl.style.visibility = 'visible';
            
            gsap.fromTo(modalEl, 
                { opacity: 0, y: 30, scale: 0.95 },
                { opacity: 1, y: 0, scale: 1, duration: 0.5, ease: "back.out(1.5)" }
            );
        };

        if (currentModal) {
            closeModals(showNewModal);
        } else {
            showNewModal();
        }
    };

    // Dashboard Entry Animation
    const enterDashboard = () => {
        // Fade out hero UI
        gsap.to([heroContent, mainNav, '#nav-dots'], {
            opacity: 0,
            duration: 0.8,
            ease: "power2.inOut",
            onComplete: () => {
                if (heroContent) heroContent.style.visibility = 'hidden';
                if (mainNav) mainNav.style.visibility = 'hidden';
            }
        });

        // Trigger Three.js Earth to shrink/move
        if (window.earthGroup && window.camera) {
            gsap.to(window.camera.position, {
                z: window.camera.position.z + 10,
                duration: 2,
                ease: "power3.inOut"
            });
            gsap.to(window.earthGroup.position, {
                x: 8,
                y: 0,
                duration: 2,
                ease: "power3.inOut"
            });
        }

        // Show Dashboard Layer
        dashboardLayer.style.visibility = 'visible';
        dashboardLayer.classList.add('active');
        gsap.to(dashboardLayer, {
            opacity: 1,
            duration: 1.5,
            delay: 0.5,
            ease: "power2.out"
        });
    };

    // Dashboard Exit Animation
    const exitDashboard = () => {
        // Fade out Dashboard Layer
        dashboardLayer.classList.remove('active');
        gsap.to(dashboardLayer, {
            opacity: 0,
            duration: 1.0,
            ease: "power2.in",
            onComplete: () => {
                dashboardLayer.style.visibility = 'hidden';
            }
        });

        // Trigger Three.js Earth to return
        if (window.earthGroup && window.camera) {
            gsap.to(window.camera.position, {
                z: window.camera.position.z - 10,
                duration: 2,
                ease: "power3.inOut"
            });
            gsap.to(window.earthGroup.position, {
                x: 0,
                y: 0,
                duration: 2,
                ease: "power3.inOut"
            });
        }

        // Restore landing page visibility (un-hide from FOUC prevention)
        const heroOverlay = document.getElementById('hero-ui-overlay');
        const scrollCont = document.getElementById('scroll-container');
        if (heroOverlay) { heroOverlay.style.opacity = ''; heroOverlay.style.visibility = ''; }
        if (scrollCont) { scrollCont.style.opacity = ''; scrollCont.style.visibility = ''; }
        document.body.classList.add('app-ready');

        // Fade in hero UI
        if (heroContent) heroContent.style.visibility = 'visible';
        if (mainNav) { mainNav.style.visibility = 'visible'; mainNav.style.opacity = ''; }
        gsap.to([heroContent, mainNav, '#nav-dots'], {
            opacity: 1,
            duration: 1.5,
            delay: 0.5,
            ease: "power2.out"
        });
    };

    // Router
    const navigate = (path) => {
        window.history.pushState({}, '', path);
        handleRoute(path);
    };

    const handleRoute = (path) => {
        // Normal paths
        if (path === '/' || path === '/index.html') {
            closeModals();
            if (dashboardLayer.classList.contains('active')) {
                exitDashboard();
            }
        } else if (path === '/login') {
            openModal(loginModal);
        } else if (path === '/signup') {
            openModal(signupModal);
        } else if (path === '/dashboard' || path === '/uploader.html') {
            closeModals(() => {
                enterDashboard();
            });
        }
    };

    // Bind Nav Buttons
    const btnLaunch = document.getElementById('nav-btn-launch');
    if (btnLaunch) btnLaunch.addEventListener('click', (e) => { e.preventDefault(); openModal(launchModal); });

    const btnSignin = document.getElementById('nav-btn-signin');
    if (btnSignin) btnSignin.addEventListener('click', (e) => { e.preventDefault(); navigate('/login'); });

    const btnSignup = document.getElementById('nav-btn-signup');
    if (btnSignup) btnSignup.addEventListener('click', (e) => { e.preventDefault(); navigate('/signup'); });

    const btnGuest = document.getElementById('nav-btn-guest');
    if (btnGuest) btnGuest.addEventListener('click', (e) => { 
        e.preventDefault(); 
        localStorage.setItem('is_guest', 'true');
        localStorage.setItem('meghamukt_user', 'Guest Officer');
        window.dispatchEvent(new Event('auth-changed'));
        navigate('/dashboard'); 
    });

    // Bind Launch Modal Buttons
    const btnLaunchLogin = document.getElementById('btn-launch-login');
    if(btnLaunchLogin) btnLaunchLogin.addEventListener('click', () => navigate('/login'));
    
    const btnLaunchSignup = document.getElementById('btn-launch-signup');
    if(btnLaunchSignup) btnLaunchSignup.addEventListener('click', () => navigate('/signup'));
    
    const btnLaunchGuest = document.getElementById('btn-launch-guest');
    if(btnLaunchGuest) btnLaunchGuest.addEventListener('click', () => {
        localStorage.setItem('is_guest', 'true');
        localStorage.setItem('meghamukt_user', 'Guest Officer');
        window.dispatchEvent(new Event('auth-changed'));
        navigate('/dashboard');
    });
    
    const btnBackLaunch = document.getElementById('back-from-launch');
    if(btnBackLaunch) btnBackLaunch.addEventListener('click', (e) => {
        e.preventDefault();
        navigate('/');
    });

    // Bind Auth Modal Switches & Backs
    const btnSwitchSignup = document.getElementById('switch-to-signup');
    if(btnSwitchSignup) btnSwitchSignup.addEventListener('click', (e) => {
        e.preventDefault();
        navigate('/signup');
    });

    const btnSwitchLogin = document.getElementById('switch-to-login');
    if(btnSwitchLogin) btnSwitchLogin.addEventListener('click', (e) => {
        e.preventDefault();
        navigate('/login');
    });

    const btnBackLogin = document.getElementById('back-from-login');
    if(btnBackLogin) btnBackLogin.addEventListener('click', (e) => {
        e.preventDefault();
        navigate('/');
    });

    const btnBackSignup = document.getElementById('back-from-signup');
    if(btnBackSignup) btnBackSignup.addEventListener('click', (e) => {
        e.preventDefault();
        navigate('/');
    });

    // Handle API Errors safely
    const showError = (btnId, message) => {
        const btn = document.getElementById(btnId);
        if(!btn) return;
        const originalText = btn.innerText;
        btn.innerText = message;
        btn.style.backgroundColor = 'rgba(255, 50, 50, 0.2)';
        btn.style.color = '#ff6b6b';
        btn.style.borderColor = '#ff6b6b';
        setTimeout(() => {
            btn.innerText = originalText;
            btn.style.backgroundColor = '';
            btn.style.color = '';
            btn.style.borderColor = '';
        }, 3000);
    };

    // Submit actions (Real API integration)
    const btnSubmitLogin = document.getElementById('btn-submit-login');
    if (btnSubmitLogin) btnSubmitLogin.addEventListener('click', async () => {
        const emailInput = document.getElementById('login-email').value; // Backend uses this as username currently via UI mappings
        const passwordInput = document.getElementById('login-password').value;
        if (!emailInput || !passwordInput) return showError('btn-submit-login', 'Missing fields');
        
        btnSubmitLogin.innerText = 'AUTHENTICATING...';
        
        try {
            const res = await fetch('http://localhost:8000/api/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username: emailInput, password: passwordInput })
            });
            const data = await res.json();
            
            if (res.ok && data.access_token) {
                localStorage.setItem('meghamukt_token', data.access_token);
                localStorage.setItem('meghamukt_user', data.username);
                localStorage.setItem('is_guest', 'false');
                window.dispatchEvent(new Event('auth-changed'));
                navigate('/dashboard');
            } else {
                showError('btn-submit-login', data.detail || 'Login failed');
            }
        } catch (e) {
            showError('btn-submit-login', 'Network Error');
        }
    });

    const btnSubmitSignup = document.getElementById('btn-submit-signup');
    if (btnSubmitSignup) btnSubmitSignup.addEventListener('click', async () => {
        const nameInput = document.getElementById('signup-name').value;
        const emailInput = document.getElementById('signup-email').value;
        const passwordInput = document.getElementById('signup-password').value;
        if (!nameInput || !emailInput || !passwordInput) return showError('btn-submit-signup', 'Missing fields');
        
        btnSubmitSignup.innerText = 'CREATING ACCOUNT...';
        
        try {
            const res = await fetch('http://localhost:8000/api/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username: nameInput, email: emailInput, password: passwordInput })
            });
            const data = await res.json();
            
            if (res.ok && data.access_token) {
                localStorage.setItem('meghamukt_token', data.access_token);
                localStorage.setItem('meghamukt_user', data.username);
                localStorage.setItem('is_guest', 'false');
                window.dispatchEvent(new Event('auth-changed'));
                navigate('/dashboard');
            } else {
                showError('btn-submit-signup', data.detail || 'Signup failed');
            }
        } catch (e) {
            showError('btn-submit-signup', 'Network Error');
        }
    });

    // Handle back/forward browser buttons
    window.addEventListener('popstate', () => {
        handleRoute(window.location.pathname);
    });

    // Handle initial load — route immediately to prevent FOUC
    const initialPath = window.location.pathname;
    
    if (initialPath === '/dashboard' || initialPath === '/uploader.html') {
        // Dashboard route: hide landing page immediately, show dashboard
        const heroOverlay = document.getElementById('hero-ui-overlay');
        const scrollCont = document.getElementById('scroll-container');
        if (heroOverlay) { heroOverlay.style.opacity = '0'; heroOverlay.style.visibility = 'hidden'; }
        if (scrollCont) { scrollCont.style.opacity = '0'; scrollCont.style.visibility = 'hidden'; }
        if (mainNav) { mainNav.style.opacity = '0'; mainNav.style.visibility = 'hidden'; }
        
        // Skip the loading screen entirely for dashboard
        const existingLoader = document.getElementById('loading-screen');
        if (existingLoader) existingLoader.remove();
        
        // Show dashboard immediately
        dashboardLayer.style.visibility = 'visible';
        dashboardLayer.classList.add('active');
        dashboardLayer.style.opacity = '1';
        
        // Mark app ready (but landing page stays hidden since we're on dashboard)
        document.body.classList.add('app-ready');
        
        // Move earth out of the way
        const checkEarth = setInterval(() => {
            if (window.earthGroup && window.camera) {
                clearInterval(checkEarth);
                window.camera.position.z += 10;
                window.earthGroup.position.x = 8;
            }
        }, 100);
        setTimeout(() => clearInterval(checkEarth), 5000); // safety cleanup
    } else {
        // Landing page: let loading screen play, then reveal
        // Mark app-ready after a short delay so loading screen coordinates
        const waitForLoader = setInterval(() => {
            const loader = document.getElementById('loading-screen');
            if (!loader || loader.classList.contains('is-hidden')) {
                clearInterval(waitForLoader);
                document.body.classList.add('app-ready');
                handleRoute(initialPath);
            }
        }, 50);
        // Safety: if loader never finishes, force-ready after 2s
        setTimeout(() => {
            clearInterval(waitForLoader);
            document.body.classList.add('app-ready');
            handleRoute(initialPath);
        }, 2000);
    }
});
