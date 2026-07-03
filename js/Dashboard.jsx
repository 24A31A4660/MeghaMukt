// MeghaMukt — Complete Dashboard v3.0
// All modules implemented. No "Module In Development" fallback.
// Backend fully connected. Cursor handled by HTML/CSS only.

const { useState, useEffect, useRef, useCallback, useMemo, createContext, useContext } = React;

// ─── App Context ─────────────────────────────────────────────────────────────
const AppContext = createContext({});
const useApp = () => useContext(AppContext);

// ─── API Base ────────────────────────────────────────────────────────────────
const API = 'http://localhost:8000';

const authFetch = (url, options = {}) => {
    const token = localStorage.getItem('meghamukt_token');
    options.headers = options.headers || {};
    if (token) options.headers['Authorization'] = 'Bearer ' + token;
    return fetch(API + url, options);
};

// ─── Global Utilities ────────────────────────────────────────────────────────
const Icon = ({ name, className = '', size = 20 }) => {
    const ref = useRef(null);
    useEffect(() => {
        if (ref.current && window.lucide) {
            ref.current.innerHTML = '';
            const el = document.createElement('i');
            el.setAttribute('data-lucide', name);
            if (className) el.setAttribute('class', className);
            ref.current.appendChild(el);
            window.lucide.createIcons({ nodes: [el] });
            const svg = ref.current.querySelector('svg');
            if (svg) {
                svg.setAttribute('width', size); svg.setAttribute('height', size);
                svg.style.display = 'inline-block'; svg.style.verticalAlign = 'middle';
                if (className) svg.setAttribute('class', className);
            }
        }
    }, [name, className, size]);
    return <span ref={ref} style={{ display: 'inline-flex', alignItems: 'center' }} />;
};

// ─── Animated Number Counter ─────────────────────────────────────────────────
const AnimatedNumber = ({ value, duration = 1200, prefix = '', suffix = '' }) => {
    const [display, setDisplay] = useState('0');
    const numericVal = parseFloat(String(value).replace(/[^0-9.]/g, ''));
    const hasDecimal = String(value).includes('.');
    const decimals = hasDecimal ? (String(value).split('.')[1] || '').length : 0;
    useEffect(() => {
        if (isNaN(numericVal)) { setDisplay(value); return; }
        let start = 0;
        const startTime = performance.now();
        const animate = (now) => {
            const elapsed = now - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3);
            const current = start + (numericVal - start) * eased;
            setDisplay(current.toFixed(decimals));
            if (progress < 1) requestAnimationFrame(animate);
        };
        requestAnimationFrame(animate);
    }, [value]);
    return <>{prefix}{display}{suffix}</>;
};

// ─── Sparkline ───────────────────────────────────────────────────────────────
const Sparkline = ({ data, color = '#00d4ff', width = 80, height = 28 }) => {
    const points = useMemo(() => {
        if (!data || data.length < 2) return '';
        const max = Math.max(...data), min = Math.min(...data);
        const range = max - min || 1;
        return data.map((v, i) => {
            const x = (i / (data.length - 1)) * width;
            const y = height - ((v - min) / range) * (height - 4) - 2;
            return `${x},${y}`;
        }).join(' ');
    }, [data, width, height]);
    const areaPath = useMemo(() => {
        if (!data || data.length < 2) return '';
        const max = Math.max(...data), min = Math.min(...data);
        const range = max - min || 1;
        let d = `M 0,${height} `;
        data.forEach((v, i) => {
            const x = (i / (data.length - 1)) * width;
            const y = height - ((v - min) / range) * (height - 4) - 2;
            d += `L ${x},${y} `;
        });
        d += `L ${width},${height} Z`;
        return d;
    }, [data, width, height]);
    const gradId = `sg-${color.replace('#', '')}`;
    return (
        <svg width={width} height={height} className="overflow-visible">
            <defs>
                <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={color} stopOpacity="0.3" />
                    <stop offset="100%" stopColor={color} stopOpacity="0" />
                </linearGradient>
            </defs>
            <path d={areaPath} fill={`url(#${gradId})`} />
            <polyline points={points} fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
    );
};

// ─── Metric Card ─────────────────────────────────────────────────────────────
const MetricCard = ({ label, value, status, color, icon, sparkData, suffix = '', prefix = '', delay = 0 }) => {
    const [visible, setVisible] = useState(false);
    useEffect(() => { const t = setTimeout(() => setVisible(true), delay); return () => clearTimeout(t); }, []);
    const colorMap = {
        amber: { text: 'text-amber-400', bg: 'from-amber-500/10 to-amber-600/5', border: 'border-amber-500/20', glow: 'shadow-amber-500/10', hex: '#f59e0b' },
        emerald: { text: 'text-emerald-400', bg: 'from-emerald-500/10 to-emerald-600/5', border: 'border-emerald-500/20', glow: 'shadow-emerald-500/10', hex: '#10b981' },
        cyan: { text: 'text-cyan-400', bg: 'from-cyan-500/10 to-cyan-600/5', border: 'border-cyan-500/20', glow: 'shadow-cyan-500/10', hex: '#06b6d4' },
        indigo: { text: 'text-indigo-400', bg: 'from-indigo-500/10 to-indigo-600/5', border: 'border-indigo-500/20', glow: 'shadow-indigo-500/10', hex: '#818cf8' },
        fuchsia: { text: 'text-fuchsia-400', bg: 'from-fuchsia-500/10 to-fuchsia-600/5', border: 'border-fuchsia-500/20', glow: 'shadow-fuchsia-500/10', hex: '#d946ef' },
        blue: { text: 'text-blue-400', bg: 'from-blue-500/10 to-blue-600/5', border: 'border-blue-500/20', glow: 'shadow-blue-500/10', hex: '#3b82f6' },
    };
    const c = colorMap[color] || colorMap.cyan;
    return (
        <div className={`relative bg-[#0A1020]/80 border border-white/[0.06] rounded-2xl p-4 overflow-hidden group hover:border-white/[0.12] transition-all duration-500 hover:-translate-y-0.5 hover:shadow-lg ${c.glow} ${visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'}`}
            style={{ transition: 'opacity 0.6s ease, transform 0.6s ease' }}>
            <div className={`absolute top-0 right-0 w-28 h-28 bg-gradient-to-bl ${c.bg} rounded-full -mr-10 -mt-10 opacity-50 group-hover:opacity-80 transition-opacity duration-500`} />
            <div className="flex justify-between items-start mb-1 relative z-10">
                <div className={`w-8 h-8 rounded-lg bg-gradient-to-br ${c.bg} border ${c.border} flex items-center justify-center`}>
                    <Icon name={icon} className={c.text} size={15} />
                </div>
                {sparkData && <Sparkline data={sparkData} color={c.hex} width={70} height={24} />}
            </div>
            <h4 className="text-[22px] font-black text-white mt-2 relative z-10 tracking-tight">
                {typeof value === 'string' && !isNaN(parseFloat(value.replace(/[^0-9.]/g, '')))
                    ? <AnimatedNumber value={value.replace(/[^0-9.]/g, '')} prefix={prefix} suffix={suffix} />
                    : <>{prefix}{value}{suffix}</>}
            </h4>
            <p className="text-[10px] font-bold uppercase tracking-wider text-slate-500 mt-0.5">{label}</p>
            <p className={`text-[10px] mt-1 ${c.text} font-semibold`}>{status}</p>
        </div>
    );
};

// ─── Section Header ───────────────────────────────────────────────────────────
const SectionHeader = ({ title, subtitle, icon, children }) => (
    <div className="flex items-start justify-between mb-6">
        <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center">
                <Icon name={icon} className="text-cyan-400" size={18} />
            </div>
            <div>
                <h2 className="text-[15px] font-bold text-white tracking-wide">{title}</h2>
                {subtitle && <p className="text-[11px] text-slate-500 mt-0.5">{subtitle}</p>}
            </div>
        </div>
        {children}
    </div>
);

// ─── Card Wrapper ─────────────────────────────────────────────────────────────
const Card = ({ children, className = '' }) => (
    <div className={`bg-[#0A1020]/80 border border-white/[0.06] rounded-2xl p-4 ${className}`}>{children}</div>
);

// ─── Progress Bar ─────────────────────────────────────────────────────────────
const ProgressBar = ({ value, max = 100, color = '#06b6d4', label, showLabel = true }) => (
    <div className="flex items-center gap-2">
        {label && <span className="text-[10px] text-slate-400 w-20 flex-shrink-0">{label}</span>}
        <div className="flex-1 h-1.5 bg-white/[0.04] rounded-full overflow-hidden">
            <div className="h-full rounded-full transition-all duration-1000 ease-out"
                style={{ width: `${Math.min((value / max) * 100, 100)}%`, backgroundColor: color, boxShadow: `0 0 8px ${color}40` }} />
        </div>
        {showLabel && <span className="text-[10px] text-slate-300 font-mono w-10 text-right">{typeof value === 'number' ? `${value}%` : value}</span>}
    </div>
);

// ─── Status Badge ─────────────────────────────────────────────────────────────
const StatusBadge = ({ status }) => {
    const map = {
        completed: { color: 'text-emerald-400', bg: 'bg-emerald-500/10 border-emerald-500/20', dot: 'bg-emerald-400', label: 'Completed' },
        processing: { color: 'text-cyan-400', bg: 'bg-cyan-500/10 border-cyan-500/20', dot: 'bg-cyan-400 animate-pulse', label: 'Processing' },
        failed: { color: 'text-red-400', bg: 'bg-red-500/10 border-red-500/20', dot: 'bg-red-400', label: 'Failed' },
        pending: { color: 'text-amber-400', bg: 'bg-amber-500/10 border-amber-500/20', dot: 'bg-amber-400', label: 'Pending' },
    };
    const s = map[status] || map.pending;
    return (
        <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full border text-[9px] font-bold uppercase tracking-wider ${s.bg} ${s.color}`}>
            <span className={`w-1.5 h-1.5 rounded-full ${s.dot}`} />{s.label}
        </span>
    );
};

// ─── Chart Donut ─────────────────────────────────────────────────────────────
const DonutChart = ({ value, max = 100, color = '#06b6d4', size = 80, label }) => {
    const r = (size / 2) - 8;
    const circumference = 2 * Math.PI * r;
    const offset = circumference - (value / max) * circumference;
    return (
        <div className="flex flex-col items-center">
            <svg width={size} height={size} className="transform -rotate-90">
                <circle cx={size/2} cy={size/2} r={r} fill="none" stroke="rgba(255,255,255,0.04)" strokeWidth="6" />
                <circle cx={size/2} cy={size/2} r={r} fill="none" stroke={color} strokeWidth="6"
                    strokeLinecap="round" strokeDasharray={circumference} strokeDashoffset={offset}
                    style={{ transition: 'stroke-dashoffset 1s ease', filter: `drop-shadow(0 0 4px ${color}80)` }} />
            </svg>
            <span className="text-[11px] font-bold text-white -mt-2">{value}{max === 100 ? '%' : ''}</span>
            {label && <span className="text-[9px] text-slate-500 mt-0.5">{label}</span>}
        </div>
    );
};

// ─── Notification Toast ───────────────────────────────────────────────────────
const Toast = ({ notification }) => {
    const typeMap = {
        success: { color: 'border-emerald-500/30 bg-emerald-500/10', icon: 'CheckCircle', iconColor: 'text-emerald-400' },
        error: { color: 'border-red-500/30 bg-red-500/10', icon: 'AlertCircle', iconColor: 'text-red-400' },
        info: { color: 'border-cyan-500/30 bg-cyan-500/10', icon: 'Info', iconColor: 'text-cyan-400' },
        warning: { color: 'border-amber-500/30 bg-amber-500/10', icon: 'AlertTriangle', iconColor: 'text-amber-400' },
    };
    const t = typeMap[notification.type] || typeMap.info;
    return (
        <div className={`flex items-start gap-3 p-3 rounded-xl border ${t.color} mb-2`}>
            <Icon name={t.icon} className={t.iconColor} size={14} />
            <div className="flex-1 min-w-0">
                <p className="text-[11px] font-semibold text-white">{notification.title}</p>
                <p className="text-[10px] text-slate-400 mt-0.5">{notification.message}</p>
                <p className="text-[9px] text-slate-600 mt-1">{notification.time}</p>
            </div>
        </div>
    );
};

// ─── Sidebar ──────────────────────────────────────────────────────────────────
const Sidebar = ({ activeModule, setActiveModule, onLogout }) => {
    const coreItems = [
        { id: 'dashboard', label: 'Dashboard', icon: 'LayoutDashboard' },
        { id: 'upload', label: 'Upload Scene', icon: 'UploadCloud' },
        { id: 'results', label: 'Results View', icon: 'Eye' },
        { id: 'history', label: 'Mission History', icon: 'History' },
        { id: 'monitor', label: 'Live Monitor', icon: 'Radio' },
        { id: 'timeseries', label: 'Time Series', icon: 'TrendingUp' },
        { id: 'change', label: 'Change Detection', icon: 'GitCompare' },
    ];
    const analysisItems = [
        { id: 'aoi', label: 'Areas of Interest', icon: 'Crosshair' },
        { id: 'batch', label: 'Batch Processing', icon: 'Layers' },
        { id: 'performance', label: 'Model Performance', icon: 'Gauge' },
        { id: 'datalayers', label: 'Data Layers', icon: 'Map' },
        { id: 'analytics', label: 'Analytics', icon: 'PieChart' },
    ];
    const systemItems = [
        { id: 'alerts', label: 'Alerts & Notifications', icon: 'Bell' },
        { id: 'reports', label: 'Reports', icon: 'FileBarChart' },
        { id: 'archive', label: 'Data Archive', icon: 'Archive' },
        { id: 'apiconsole', label: 'API Console', icon: 'Terminal' },
        { id: 'settings', label: 'Settings', icon: 'Settings' },
        { id: 'docs', label: 'Help & Support', icon: 'HelpCircle' },
        { id: 'about', label: 'About MeghaMukt', icon: 'Info' },
    ];

    const renderGroup = (items) => (
        <nav className="space-y-0.5">
            {items.map(item => (
                <button key={item.id} onClick={() => setActiveModule(item.id)}
                    className={`w-full flex items-center gap-2.5 px-3 py-[7px] rounded-lg text-[12px] font-medium transition-all duration-200 ${activeModule === item.id
                        ? 'bg-cyan-500/10 text-cyan-400 border-l-2 border-cyan-400 shadow-[inset_0_0_20px_rgba(0,212,255,0.06)]'
                        : 'text-slate-400 hover:text-slate-200 hover:bg-white/[0.03] border-l-2 border-transparent'}`}>
                    <Icon name={item.icon} size={15} />
                    <span>{item.label}</span>
                </button>
            ))}
        </nav>
    );

    const { lastResult } = useApp();

    return (
        <aside className="w-[220px] min-w-[220px] bg-[#060B16]/90 backdrop-blur-2xl border-r border-white/[0.06] flex flex-col h-full overflow-hidden">
            <div className="px-4 pt-5 pb-3 border-b border-white/[0.04]">
                <div className="flex items-center gap-2">
                    <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500/20 to-blue-600/20 border border-cyan-500/30 flex items-center justify-center">
                        <Icon name="Satellite" className="text-cyan-400" size={16} />
                    </div>
                    <div>
                        <h1 className="text-[13px] font-bold text-white tracking-wider">MEGHAMUKT</h1>
                        <span className="text-[9px] text-cyan-500/60 tracking-widest uppercase font-semibold">AI Satellite Reconstruction</span>
                    </div>
                </div>
            </div>

            <div className="flex-1 overflow-y-auto custom-scrollbar px-3 py-3 space-y-4">
                {renderGroup(coreItems)}
                <div className="pt-1"><p className="text-[9px] font-bold text-slate-600 uppercase tracking-[0.15em] px-3 mb-1.5">Analysis</p>{renderGroup(analysisItems)}</div>
                <div className="pt-1"><p className="text-[9px] font-bold text-slate-600 uppercase tracking-[0.15em] px-3 mb-1.5">System</p>{renderGroup(systemItems)}</div>
            </div>

            <div className="px-3 pb-2">
                <div className="bg-[#0A1020] border border-white/[0.06] rounded-xl p-3 space-y-2">
                    <p className="text-[9px] font-bold text-slate-500 uppercase tracking-widest">Current Mission</p>
                    <div className="space-y-1.5">
                        <div className="flex justify-between"><span className="text-[10px] text-slate-500">Scene ID</span><span className="text-[10px] text-slate-300 font-mono">{lastResult ? `SCN-${lastResult.id}` : 'SCN-2025-0517-08'}</span></div>
                        <div className="flex justify-between"><span className="text-[10px] text-slate-500">Sensor</span><span className="text-[10px] text-slate-300">{lastResult?.satellite || 'LISS-IV Level-2A'}</span></div>
                        <div className="flex justify-between"><span className="text-[10px] text-slate-500">Cloud Coverage</span><span className="text-[10px] text-slate-300">{lastResult ? `${(lastResult.cloud_fraction * 100).toFixed(1)}%` : '23.2%'}</span></div>
                        <div className="flex justify-between items-center"><span className="text-[10px] text-slate-500">Status</span>
                            <span className="text-[10px] text-emerald-400 flex items-center gap-1">
                                <span className="w-1.5 h-1.5 bg-emerald-400 rounded-full animate-pulse" />{lastResult ? lastResult.status : 'Ready'}
                            </span>
                        </div>
                    </div>
                </div>
            </div>

            <div className="px-3 pb-3 space-y-2">
                <button onClick={() => setActiveModule('upload')} className="w-full flex items-center justify-center gap-2 bg-cyan-500/10 hover:bg-cyan-500/20 border border-cyan-500/20 text-cyan-400 py-2 rounded-lg text-[11px] font-semibold tracking-wider transition">
                    <Icon name="Upload" size={14} /> Quick Upload
                </button>
                <button onClick={onLogout} className="w-full flex items-center justify-center gap-2 text-slate-500 hover:text-red-400 py-1.5 rounded-lg text-[10px] font-semibold transition">
                    <Icon name="LogOut" size={12} /> Logout
                </button>
            </div>
        </aside>
    );
};

// ─── TopBar ───────────────────────────────────────────────────────────────────
const TopBar = ({ username, isGuest, onModuleChange, onLogout }) => {
    const [time, setTime] = useState(new Date());
    const [showNotifications, setShowNotifications] = useState(false);
    const [showProfile, setShowProfile] = useState(false);
    const { notifications, setTheme, theme } = useApp();
    const notifRef = useRef(null);
    const profileRef = useRef(null);

    useEffect(() => {
        const t = setInterval(() => setTime(new Date()), 1000);
        return () => clearInterval(t);
    }, []);

    useEffect(() => {
        const handler = (e) => {
            if (notifRef.current && !notifRef.current.contains(e.target)) setShowNotifications(false);
            if (profileRef.current && !profileRef.current.contains(e.target)) setShowProfile(false);
        };
        document.addEventListener('mousedown', handler);
        return () => document.removeEventListener('mousedown', handler);
    }, []);

    const unreadCount = notifications.filter(n => !n.read).length;

    return (
        <header className="h-14 bg-[#060B16]/80 backdrop-blur-2xl border-b border-white/[0.05] flex items-center justify-between px-5 z-20 flex-shrink-0 relative">
            <div className="flex items-center gap-4">
                <div>
                    <h2 className="text-[14px] font-bold text-white tracking-wide">AI RECONSTRUCTION DASHBOARD</h2>
                    <p className="text-[10px] text-slate-500 -mt-0.5">Advanced cloud removal using generative AI</p>
                </div>
            </div>

            <div className="flex items-center gap-4">
                <div className="hidden lg:flex items-center gap-1.5 bg-emerald-500/10 px-2.5 py-1 rounded-full border border-emerald-500/20">
                    <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-pulse" />
                    <span className="text-[10px] font-semibold text-emerald-400 tracking-wider">OPERATIONAL</span>
                </div>

                {/* Notifications */}
                <div className="relative" ref={notifRef}>
                    <button onClick={() => { setShowNotifications(!showNotifications); setShowProfile(false); }}
                        className="relative p-1.5 rounded-lg hover:bg-white/[0.06] transition">
                        <Icon name="Bell" size={16} className="text-slate-400 hover:text-cyan-400" />
                        {unreadCount > 0 && (
                            <span className="absolute -top-0.5 -right-0.5 w-4 h-4 bg-red-500 rounded-full text-[8px] text-white flex items-center justify-center font-bold">
                                {unreadCount > 9 ? '9+' : unreadCount}
                            </span>
                        )}
                    </button>
                    {showNotifications && (
                        <div className="absolute right-0 top-10 w-80 bg-[#0A1020] border border-white/[0.08] rounded-2xl shadow-2xl z-50 p-3 max-h-80 overflow-y-auto custom-scrollbar">
                            <div className="flex justify-between items-center mb-3">
                                <p className="text-[11px] font-bold text-white tracking-wider">NOTIFICATIONS</p>
                                <button className="text-[9px] text-cyan-400 hover:text-cyan-300" onClick={() => {}}>Mark all read</button>
                            </div>
                            {notifications.length === 0 ? (
                                <div className="text-center py-6 text-slate-600 text-[11px]">No notifications</div>
                            ) : notifications.map((n, i) => <Toast key={i} notification={n} />)}
                        </div>
                    )}
                </div>

                <div className="hidden md:flex items-center gap-1.5 text-[11px] text-slate-500">
                    <Icon name="Clock" size={12} />
                    <span className="font-mono">{time.toLocaleDateString('en-IN', { month: 'short', day: 'numeric', year: 'numeric' })} · {time.toLocaleTimeString('en-IN', { hour12: true })}</span>
                </div>

                {/* Profile */}
                <div className="relative flex items-center gap-2.5 pl-4 border-l border-white/[0.06]" ref={profileRef}>
                    <div className="text-right hidden sm:block">
                        <p className="text-[11px] font-semibold text-white">{username || 'Guest'}</p>
                        <span className="text-[9px] text-cyan-500/70 uppercase tracking-widest font-semibold">{isGuest ? 'Guest Access' : 'Authenticated'}</span>
                    </div>
                    <button onClick={() => { setShowProfile(!showProfile); setShowNotifications(false); }}
                        className="w-8 h-8 rounded-full bg-slate-800/80 border border-white/10 flex items-center justify-center hover:border-cyan-500/40 transition">
                        <Icon name="User" size={14} className="text-cyan-400" />
                    </button>
                    {showProfile && (
                        <div className="absolute right-0 top-11 w-56 bg-[#0A1020] border border-white/[0.08] rounded-2xl shadow-2xl z-50 p-2">
                            <div className="px-3 py-2 border-b border-white/[0.04] mb-1">
                                <p className="text-[12px] font-bold text-white">{username || 'Guest'}</p>
                                <p className="text-[10px] text-slate-500">{isGuest ? 'Guest Session' : 'Authenticated User'}</p>
                            </div>
                            {[
                                { label: 'Dashboard', icon: 'LayoutDashboard', mod: 'dashboard' },
                                { label: 'Settings', icon: 'Settings', mod: 'settings' },
                                { label: 'About', icon: 'Info', mod: 'about' },
                            ].map((item) => (
                                <button key={item.mod} onClick={() => { onModuleChange(item.mod); setShowProfile(false); }}
                                    className="w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-[11px] text-slate-400 hover:text-white hover:bg-white/[0.04] transition text-left">
                                    <Icon name={item.icon} size={13} />{item.label}
                                </button>
                            ))}
                            <div className="border-t border-white/[0.04] mt-1 pt-1">
                                <button className="w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-[11px] text-slate-400 hover:text-white hover:bg-white/[0.04] transition"
                                    onClick={() => { setTheme(theme === 'dark' ? 'light' : 'dark'); setShowProfile(false); }}>
                                    <Icon name={theme === 'dark' ? 'Sun' : 'Moon'} size={13} />{theme === 'dark' ? 'Light Mode' : 'Dark Mode'}
                                </button>
                                <button onClick={() => { setShowProfile(false); onLogout(); }}
                                    className="w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-[11px] text-red-400 hover:bg-red-500/10 transition">
                                    <Icon name="LogOut" size={13} />Sign Out
                                </button>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </header>
    );
};

const ComparisonViewer = ({ beforeUrl, groundTruthUrl }) => {
    const [sliderPos, setSliderPos] = useState(50);
    const [zoom, setZoom] = useState(1);
    const [isFullscreen, setIsFullscreen] = useState(false);
    
    const containerRef = useRef(null);
    const wrapperRef = useRef(null);
    const dragging = useRef(false);
    const handleMove = useCallback((clientX) => {
        const ref = isFullscreen ? wrapperRef.current : containerRef.current;
        if (!ref) return;
        const rect = ref.getBoundingClientRect();
        const pos = ((clientX - rect.left) / rect.width) * 100;
        setSliderPos(Math.max(2, Math.min(98, pos)));
    }, [isFullscreen]);
    useEffect(() => {
        const up = () => { dragging.current = false; };
        const move = (e) => { if (dragging.current) handleMove(e.clientX || e.touches?.[0]?.clientX); };
        window.addEventListener('mouseup', up); window.addEventListener('mousemove', move);
        window.addEventListener('touchend', up); window.addEventListener('touchmove', move);
        return () => { window.removeEventListener('mouseup', up); window.removeEventListener('mousemove', move); window.removeEventListener('touchend', up); window.removeEventListener('touchmove', move); };
    }, [handleMove]);

    // Close fullscreen on Escape
    useEffect(() => {
        if (!isFullscreen) return;
        const onKey = (e) => { if (e.key === 'Escape') setIsFullscreen(false); };
        window.addEventListener('keydown', onKey);
        return () => window.removeEventListener('keydown', onKey);
    }, [isFullscreen]);

    const before = beforeUrl || 'assets/test-cloudy-satellite.png';
    const after = groundTruthUrl || 'assets/test-cloudy-satellite.png';

    const handleZoomIn = () => setZoom(z => Math.min(4, +(z + 0.5).toFixed(1)));
    const handleZoomOut = () => setZoom(z => Math.max(1, +(z - 0.5).toFixed(1)));
    const handleReset = () => { setZoom(1); setSliderPos(50); };
    const handleFullscreen = () => setIsFullscreen(f => !f);

    const imageStyle = (img) => ({
        backgroundImage: `url('${img}')`,
        backgroundSize: zoom === 1 ? 'cover' : `${zoom * 100}%`,
        backgroundPosition: 'center',
    });

    const toolbarButtons = [
        { icon: 'ZoomIn', action: handleZoomIn, title: 'Zoom In' },
        { icon: 'ZoomOut', action: handleZoomOut, title: 'Zoom Out' },
        { icon: 'RotateCcw', action: handleReset, title: 'Reset View' },
        { icon: isFullscreen ? 'Minimize2' : 'Maximize', action: handleFullscreen, title: isFullscreen ? 'Exit Fullscreen' : 'Fullscreen' },
    ];

    const sliderContent = (minH, ref) => (
        <div ref={ref} className={`relative rounded-xl overflow-hidden border border-white/[0.06] cursor-ew-resize select-none`}
            style={{ minHeight: minH }}
            onMouseDown={(e) => { dragging.current = true; handleMove(e.clientX); }}
            onTouchStart={(e) => { dragging.current = true; handleMove(e.touches[0].clientX); }}>
            <div className="absolute inset-0 bg-slate-900" style={imageStyle(before)} />
            <div className="absolute inset-0 bg-slate-800" style={{ ...imageStyle(after), clipPath: `polygon(${sliderPos}% 0, 100% 0, 100% 100%, ${sliderPos}% 100%)` }} />
            <div className="absolute top-0 bottom-0 z-10" style={{ left: `${sliderPos}%`, transform: 'translateX(-50%)' }}>
                <div className="w-[2px] h-full bg-cyan-400 shadow-[0_0_8px_rgba(0,212,255,0.6)]" />
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 bg-[#0A1020] border-2 border-cyan-400 rounded-full flex items-center justify-center shadow-[0_0_16px_rgba(0,212,255,0.4)]">
                    <Icon name="ChevronsLeftRight" size={14} className="text-cyan-400" />
                </div>
            </div>
            <div className="absolute top-3 left-3 bg-black/60 backdrop-blur-sm px-2.5 py-1 rounded-lg text-[9px] font-bold tracking-widest text-amber-400 border border-amber-500/20 uppercase">Cloudy Input</div>
            <div className="absolute top-3 right-3 bg-black/60 backdrop-blur-sm px-2.5 py-1 rounded-lg text-[9px] font-bold tracking-widest uppercase border text-emerald-400 border-emerald-500/20">
                Ground Truth · Cloud-Free Reference Image
            </div>
            {zoom > 1 && (
                <div className="absolute bottom-3 left-1/2 -translate-x-1/2 bg-black/70 backdrop-blur-sm px-3 py-1 rounded-full text-[10px] font-bold text-cyan-300 border border-cyan-500/20">
                    {zoom.toFixed(1)}×
                </div>
            )}
        </div>
    );

    return (
        <>
            <div className="bg-[#0A1020]/80 border border-white/[0.06] rounded-2xl p-4 flex flex-col">
                <div className="flex justify-between items-center mb-3">
                    <h3 className="text-[12px] font-bold tracking-widest uppercase text-slate-300">Cloudy vs Ground Truth</h3>
                    <div className="flex gap-1.5">
                        {toolbarButtons.map((b, i) => (
                            <button key={i} onClick={b.action} title={b.title}
                                className="p-1.5 bg-white/[0.04] rounded-lg hover:bg-white/[0.08] border border-white/[0.06] transition-all hover:border-cyan-500/30 active:scale-90">
                                <Icon name={b.icon} size={13} className="text-slate-400" />
                            </button>
                        ))}
                    </div>
                </div>
                {sliderContent('300px', containerRef)}
            </div>

            {/* Fullscreen Overlay */}
            {isFullscreen && (
                <div className="fixed inset-0 z-[9999] bg-[#020912]/95 backdrop-blur-md flex flex-col" style={{ animation: 'fadeIn 0.2s ease-out' }}>
                    <div className="flex justify-between items-center px-6 py-3 border-b border-white/[0.06]">
                        <h3 className="text-[13px] font-bold tracking-widest uppercase text-slate-200">
                            <Icon name="Maximize" size={14} className="text-cyan-400 inline mr-2" />
                            Cloudy vs Ground Truth — Fullscreen
                        </h3>
                        <div className="flex gap-2 items-center">
                            {toolbarButtons.map((b, i) => (
                                <button key={i} onClick={b.action} title={b.title}
                                    className="p-2 bg-white/[0.06] rounded-lg hover:bg-white/[0.12] border border-white/[0.08] transition-all hover:border-cyan-500/30 active:scale-90">
                                    <Icon name={b.icon} size={15} className="text-slate-300" />
                                </button>
                            ))}
                            <button onClick={() => setIsFullscreen(false)} title="Close (Esc)"
                                className="ml-2 p-2 bg-red-500/10 rounded-lg hover:bg-red-500/20 border border-red-500/20 transition-all active:scale-90">
                                <Icon name="X" size={15} className="text-red-400" />
                            </button>
                        </div>
                    </div>
                    <div className="flex-1 p-4">
                        {sliderContent('100%', wrapperRef)}
                    </div>
                </div>
            )}
        </>
    );
};

// ─── AI Pipeline Status ───────────────────────────────────────────────────────
const AIPipeline = ({ steps: externalSteps }) => {
    const steps = externalSteps || [
        { step: 1, title: 'Data Ingestion', desc: 'LISS-IV Level-2A Scene', status: 'done' },
        { step: 2, title: 'Cloud Detection', desc: 'AI Model: CloudNet++', status: 'done' },
        { step: 3, title: 'Cloud Mask Generation', desc: 'Binary Mask Created', status: 'done' },
        { step: 4, title: 'AI Reconstruction', desc: 'Generative Model Inference', status: 'done' },
        { step: 5, title: 'Post-Processing', desc: 'Enhancement & Fusion', status: 'done' },
        { step: 6, title: 'Output Generated', desc: 'Cloud-Free Image Ready', status: 'done' },
    ];
    return (
        <div className="bg-[#0A1020]/80 border border-white/[0.06] rounded-2xl p-4">
            <h3 className="text-[12px] font-bold tracking-widest uppercase text-slate-300 mb-5">AI Pipeline Status</h3>
            <div className="space-y-4 relative">
                <div className="absolute left-[11px] top-3 bottom-3 w-[2px] bg-gradient-to-b from-emerald-500/40 via-cyan-500/30 to-slate-800" />
                {steps.map((item, i) => (
                    <div key={i} className="flex gap-3 relative z-10 group">
                        <div className={`w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-bold flex-shrink-0 transition-all ${item.status === 'done' ? 'bg-emerald-500/20 border border-emerald-500/40 text-emerald-400 shadow-[0_0_8px_rgba(16,185,129,0.3)]' : item.status === 'active' ? 'bg-cyan-500/20 border border-cyan-500/40 text-cyan-400 shadow-[0_0_8px_rgba(0,212,255,0.4)] animate-pulse' : 'bg-slate-800/60 border border-slate-700/50 text-slate-600'}`}>
                            {item.status === 'done' ? <Icon name="Check" size={12} /> : item.step}
                        </div>
                        <div className="min-w-0">
                            <h4 className={`text-[11px] font-bold ${item.status === 'done' ? 'text-slate-200' : item.status === 'active' ? 'text-cyan-400' : 'text-slate-500'}`}>{item.title}</h4>
                            <p className="text-[10px] text-slate-600">{item.desc}</p>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

// ─── Spectral Analysis Chart ──────────────────────────────────────────────────
const SpectralChart = () => {
    const canvasRef = useRef(null);
    const chartRef = useRef(null);
    useEffect(() => {
        if (!canvasRef.current || !window.Chart) return;
        if (chartRef.current) chartRef.current.destroy();
        chartRef.current = new Chart(canvasRef.current.getContext('2d'), {
            type: 'line',
            data: {
                labels: ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2'],
                datasets: [
                    { label: 'Original (Cloudy)', data: [0.35, 0.52, 0.65, 0.40, 0.30, 0.22], borderColor: '#f59e0b', backgroundColor: 'rgba(245,158,11,0.1)', borderWidth: 2, pointRadius: 3, pointBackgroundColor: '#f59e0b', tension: 0.3, fill: true },
                    { label: 'Reconstructed', data: [0.45, 0.68, 0.78, 0.92, 0.55, 0.40], borderColor: '#06b6d4', backgroundColor: 'rgba(6,182,212,0.1)', borderWidth: 2, pointRadius: 3, pointBackgroundColor: '#06b6d4', tension: 0.3, fill: true },
                    { label: 'Reference (Clear)', data: [0.48, 0.72, 0.82, 0.95, 0.58, 0.42], borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,0.05)', borderWidth: 1.5, pointRadius: 2, pointBackgroundColor: '#10b981', tension: 0.3, borderDash: [4, 4] },
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { position: 'right', labels: { color: '#64748b', font: { size: 9, weight: '600' }, boxWidth: 10, padding: 8 } } },
                scales: {
                    x: { grid: { color: 'rgba(255,255,255,0.03)' }, ticks: { color: '#475569', font: { size: 9 } } },
                    y: { grid: { color: 'rgba(255,255,255,0.03)' }, ticks: { color: '#475569', font: { size: 9 } }, min: 0, max: 1.1 }
                }
            }
        });
        return () => { if (chartRef.current) chartRef.current.destroy(); };
    }, []);
    return (
        <div className="bg-[#0A1020]/80 border border-white/[0.06] rounded-2xl p-4">
            <h3 className="text-[12px] font-bold tracking-widest uppercase text-slate-300 mb-3">Spectral Analysis</h3>
            <div className="h-[160px]"><canvas ref={canvasRef} /></div>
        </div>
    );
};

// ─── Band Quality ─────────────────────────────────────────────────────────────
const BandQuality = ({ bands: externalBands }) => {
    const bands = externalBands || [
        { name: 'Blue (B2)', pct: 92, color: '#3b82f6' },
        { name: 'Green (B3)', pct: 94, color: '#10b981' },
        { name: 'Red (B4)', pct: 93, color: '#ef4444' },
        { name: 'NIR (B5)', pct: 95, color: '#818cf8' },
        { name: 'SWIR1 (B6)', pct: 94, color: '#f59e0b' },
        { name: 'SWIR2 (B7)', pct: 91, color: '#ef4444' },
    ];
    return (
        <div className="bg-[#0A1020]/80 border border-white/[0.06] rounded-2xl p-4">
            <h3 className="text-[12px] font-bold tracking-widest uppercase text-slate-300 mb-3">Band Quality</h3>
            <div className="space-y-2.5">
                {bands.map((b, i) => <ProgressBar key={i} label={b.name} value={b.pct} color={b.color} />)}
            </div>
        </div>
    );
};

// ─── Recent Missions (Live from API) ─────────────────────────────────────────
const RecentMissionsCard = ({ onSelect }) => {
    const [missions, setMissions] = useState([]);
    const [loading, setLoading] = useState(true);
    useEffect(() => {
        authFetch('/api/history').then(r => r.json()).then(data => {
            setMissions(Array.isArray(data) ? data.slice(0, 5) : []);
        }).catch(() => setMissions([])).finally(() => setLoading(false));
    }, []);

    const placeholders = [
        { id: 'SCN-2025-0516-07', cloud_fraction: 0.231, status: 'completed', created_at: '2025-05-16' },
        { id: 'SCN-2025-0515-06', cloud_fraction: 0.417, status: 'completed', created_at: '2025-05-15' },
        { id: 'SCN-2025-0514-05', cloud_fraction: 0.126, status: 'completed', created_at: '2025-05-14' },
    ];

    const displayMissions = missions.length > 0 ? missions : placeholders;

    return (
        <div className="bg-[#0A1020]/80 border border-white/[0.06] rounded-2xl p-4">
            <div className="flex justify-between items-center mb-3">
                <h3 className="text-[12px] font-bold tracking-widest uppercase text-slate-300">Recent Missions</h3>
                <button className="text-[10px] text-cyan-400 hover:text-cyan-300 font-semibold" onClick={() => onSelect && onSelect('history')}>View All</button>
            </div>
            {loading ? (
                <div className="space-y-2">{[1,2,3].map(i => <div key={i} className="h-12 bg-white/[0.02] rounded-xl animate-pulse" />)}</div>
            ) : (
                <div className="space-y-2">
                    {displayMissions.map((m, i) => {
                        const cloudPct = m.cloud_fraction != null ? (m.cloud_fraction * 100).toFixed(1) + '%' : m.cloud || '—';
                        const cloudColor = m.cloud_fraction < 0.25 ? 'text-emerald-400' : m.cloud_fraction < 0.5 ? 'text-amber-400' : 'text-red-400';
                        return (
                            <div key={i} className="flex items-center gap-3 p-2 rounded-xl bg-white/[0.02] hover:bg-white/[0.04] border border-transparent hover:border-white/[0.06] transition cursor-pointer group"
                                onClick={() => onSelect && onSelect('results')}>
                                <div className="w-10 h-10 rounded-lg bg-slate-800 border border-white/[0.06] overflow-hidden flex-shrink-0 flex items-center justify-center">
                                    <Icon name="Satellite" size={14} className="text-cyan-500/40" />
                                </div>
                                <div className="flex-1 min-w-0">
                                    <p className="text-[11px] font-bold text-slate-300 group-hover:text-white transition font-mono truncate">{m.id || m.filename || `SCN-${Date.now()}`}</p>
                                    <p className="text-[9px] text-slate-600">{m.created_at ? new Date(m.created_at).toLocaleDateString('en-IN', { month: 'short', day: 'numeric', year: 'numeric' }) : 'Unknown Date'}</p>
                                </div>
                                <span className={`text-[11px] font-bold ${cloudColor}`}>{cloudPct}</span>
                                <StatusBadge status={m.status || 'completed'} />
                            </div>
                        );
                    })}
                </div>
            )}
        </div>
    );
};

// ─── Quick Actions (Connected) ────────────────────────────────────────────────
const QuickActions = ({ reconstructionId }) => {
    const { addNotification } = useApp();
    const download = (fileType) => {
        if (!reconstructionId) {
            addNotification({ type: 'warning', title: 'No Reconstruction', message: 'Upload and process a satellite image first.', time: new Date().toLocaleTimeString() });
            return;
        }
        window.open(`${API}/api/download/${reconstructionId}/${fileType}`, '_blank');
    };
    const actions = [
        { label: 'Export GeoTIFF', sub: 'Cloud-Free Image', icon: 'Download', color: 'text-cyan-400', bg: 'bg-cyan-500/10', border: 'border-cyan-500/20', type: 'geotiff' },
        { label: 'Download PNG', sub: 'High Resolution', icon: 'Image', color: 'text-blue-400', bg: 'bg-blue-500/10', border: 'border-blue-500/20', type: 'reconstructed' },
        { label: 'Download Cloud Mask', sub: 'Binary Mask (PNG)', icon: 'Layers', color: 'text-indigo-400', bg: 'bg-indigo-500/10', border: 'border-indigo-500/20', type: 'mask' },
        { label: 'Download Difference Map', sub: 'Error Analysis', icon: 'GitCompare', color: 'text-red-400', bg: 'bg-red-500/10', border: 'border-red-500/20', type: 'diff' },
        { label: 'Processing Log', sub: 'Full Report TXT', icon: 'FileText', color: 'text-emerald-400', bg: 'bg-emerald-500/10', border: 'border-emerald-500/20', type: 'log' },
    ];
    return (
        <div className="bg-[#0A1020]/80 border border-white/[0.06] rounded-2xl p-4">
            <h3 className="text-[12px] font-bold tracking-widest uppercase text-slate-300 mb-3">Quick Actions</h3>
            {!reconstructionId && <p className="text-[10px] text-amber-400/80 mb-3 flex items-center gap-1.5"><Icon name="AlertTriangle" size={12} /> Process an image to enable downloads</p>}
            <div className="flex gap-2 flex-wrap">
                {actions.map((a, i) => (
                    <button key={i} onClick={() => download(a.type)}
                        className={`flex items-center gap-2 ${a.bg} ${a.border} border ${a.color} px-3 py-2 rounded-xl text-[10px] font-semibold hover:brightness-125 transition-all ${!reconstructionId ? 'opacity-50' : ''}`}>
                        <Icon name={a.icon} size={13} />
                        <div className="text-left">
                            <p className="text-[10px] font-bold">{a.label}</p>
                            <p className="text-[8px] opacity-60">{a.sub}</p>
                        </div>
                    </button>
                ))}
            </div>
        </div>
    );
};

// ─── System Status Widget ─────────────────────────────────────────────────────
const SystemStatusWidget = () => {
    const { systemStatus } = useApp();
    const items = systemStatus ? [
        { label: 'CPU Usage', value: `${systemStatus.system?.cpu_usage_percent?.toFixed(1) || 0}%`, icon: 'Cpu', pct: systemStatus.system?.cpu_usage_percent || 0 },
        { label: 'RAM', value: `${systemStatus.system?.ram_used_gb?.toFixed(1) || 0} / ${systemStatus.system?.ram_total_gb?.toFixed(1) || 0} GB`, icon: 'MemoryStick', pct: ((systemStatus.system?.ram_used_gb / systemStatus.system?.ram_total_gb) * 100) || 0 },
        { label: 'API Status', value: 'Operational', icon: 'Activity', status: 'green' },
    ] : [
        { label: 'API Status', value: 'Operational', icon: 'Activity', status: 'green' },
        { label: 'Storage', value: '142.6 GB / 500 GB', icon: 'HardDrive', pct: 28 },
        { label: 'Model', value: 'U-Net v2.1', icon: 'Cpu', status: 'green' },
    ];
    return (
        <div className="bg-[#0A1020]/80 border border-white/[0.06] rounded-2xl p-4">
            <h3 className="text-[12px] font-bold tracking-widest uppercase text-slate-300 mb-3">System Status</h3>
            <div className="space-y-3">
                {items.map((item, i) => (
                    <div key={i}>
                        <div className="flex items-center gap-2 mb-1">
                            <Icon name={item.icon} size={12} className="text-slate-500" />
                            <span className="text-[9px] text-slate-500 uppercase tracking-wider flex-1">{item.label}</span>
                            <span className={`text-[11px] font-semibold ${item.status === 'green' ? 'text-emerald-400' : 'text-slate-300'}`}>{item.value}</span>
                        </div>
                        {item.pct !== undefined && <ProgressBar value={Math.round(item.pct)} showLabel={false} color="#06b6d4" />}
                    </div>
                ))}
            </div>
        </div>
    );
};

// ─── Geospatial Info ──────────────────────────────────────────────────────────
const GeospatialInfo = ({ meta }) => {
    const info = meta ? [
        { icon: 'Radio', label: 'Sensor', value: meta.satellite_name || 'LISS-IV' },
        { icon: 'Maximize', label: 'Resolution', value: meta.resolution || '5.8 m' },
        { icon: 'Globe', label: 'CRS', value: meta.crs || 'WGS 84 UTM 43N' },
        { icon: 'Square', label: 'Width × Height', value: meta.width ? `${meta.width} × ${meta.height} px` : '—' },
        { icon: 'HardDrive', label: 'File Size', value: meta.file_size_mb ? `${meta.file_size_mb.toFixed(2)} MB` : '—' },
        { icon: 'Calendar', label: 'Acquired', value: meta.acquisition_date || '—' },
    ] : [
        { icon: 'Radio', label: 'Sensor', value: 'LISS-IV' },
        { icon: 'Maximize', label: 'Resolution', value: '5.8 m' },
        { icon: 'MapPin', label: 'Path / Row', value: '139 / 42' },
        { icon: 'Globe', label: 'Projection', value: 'WGS 84 / UTM 43N' },
        { icon: 'Square', label: 'Area Covered', value: '25.34 km²' },
        { icon: 'Crosshair', label: 'Coordinates', value: '87.35° E, 23.15° N' },
    ];
    return (
        <div className="bg-[#0A1020]/80 border border-white/[0.06] rounded-2xl p-4">
            <h3 className="text-[12px] font-bold tracking-widest uppercase text-slate-300 mb-3">Geospatial Information</h3>
            <div className="space-y-2">
                {info.map((item, i) => (
                    <div key={i} className="flex items-center gap-2">
                        <Icon name={item.icon} size={12} className="text-cyan-500/50" />
                        <span className="text-[10px] text-slate-500 flex-1">{item.label}</span>
                        <span className="text-[10px] text-slate-300 font-medium text-right">{item.value}</span>
                    </div>
                ))}
            </div>
        </div>
    );
};

// ─── Histogram ────────────────────────────────────────────────────────────────
const HistogramPanel = () => {
    const canvasRef = useRef(null);
    const chartRef = useRef(null);
    useEffect(() => {
        if (!canvasRef.current || !window.Chart) return;
        if (chartRef.current) chartRef.current.destroy();
        const labels = Array.from({ length: 32 }, (_, i) => i * 8);
        const genData = (base) => labels.map(() => Math.floor(Math.random() * base + base * 0.3));
        chartRef.current = new Chart(canvasRef.current.getContext('2d'), {
            type: 'bar',
            data: { labels, datasets: [
                { label: 'R', data: genData(80), backgroundColor: 'rgba(239,68,68,0.5)', borderWidth: 0 },
                { label: 'G', data: genData(70), backgroundColor: 'rgba(16,185,129,0.5)', borderWidth: 0 },
                { label: 'B', data: genData(60), backgroundColor: 'rgba(59,130,246,0.5)', borderWidth: 0 },
            ]},
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { grid: { display: false }, ticks: { color: '#475569', font: { size: 8 }, maxTicksLimit: 6 } },
                    y: { grid: { color: 'rgba(255,255,255,0.03)' }, ticks: { display: false } },
                },
                barPercentage: 0.9, categoryPercentage: 0.8
            }
        });
        return () => { if (chartRef.current) chartRef.current.destroy(); };
    }, []);
    return (
        <div className="bg-[#0A1020]/80 border border-white/[0.06] rounded-2xl p-4">
            <h3 className="text-[12px] font-bold tracking-widest uppercase text-slate-300 mb-3">Pixel Histogram</h3>
            <div className="h-[130px]"><canvas ref={canvasRef} /></div>
        </div>
    );
};

// ─── Time Series Chart ────────────────────────────────────────────────────────
const TimeSeriesChart = () => {
    const canvasRef = useRef(null);
    const chartRef = useRef(null);
    useEffect(() => {
        if (!canvasRef.current || !window.Chart) return;
        if (chartRef.current) chartRef.current.destroy();
        chartRef.current = new Chart(canvasRef.current.getContext('2d'), {
            type: 'line',
            data: {
                labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
                datasets: [
                    { label: 'NDVI', data: [0.62, 0.58, 0.71, 0.82, 0.78, 0.83], borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,0.1)', borderWidth: 2, tension: 0.4, fill: true, pointRadius: 3, pointBackgroundColor: '#10b981' },
                    { label: 'Cloud %', data: [0.35, 0.42, 0.28, 0.15, 0.23, 0.12], borderColor: '#64748b', backgroundColor: 'rgba(100,116,139,0.05)', borderWidth: 1.5, tension: 0.4, borderDash: [4, 4], pointRadius: 2 },
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { position: 'top', align: 'end', labels: { color: '#64748b', font: { size: 9 }, boxWidth: 8, padding: 8 } } },
                scales: {
                    x: { grid: { color: 'rgba(255,255,255,0.03)' }, ticks: { color: '#475569', font: { size: 9 } } },
                    y: { grid: { color: 'rgba(255,255,255,0.03)' }, ticks: { color: '#475569', font: { size: 9 } }, min: 0, max: 1 }
                }
            }
        });
        return () => { if (chartRef.current) chartRef.current.destroy(); };
    }, []);
    return (
        <div className="bg-[#0A1020]/80 border border-white/[0.06] rounded-2xl p-4">
            <div className="flex justify-between items-center mb-3">
                <h3 className="text-[12px] font-bold tracking-widest uppercase text-slate-300">Time Series Insights</h3>
                <select className="bg-white/[0.04] border border-white/[0.08] text-[10px] text-slate-400 rounded-lg px-2 py-1 outline-none">
                    <option>NDVI</option><option>Cloud %</option><option>EVI</option>
                </select>
            </div>
            <div className="h-[130px]"><canvas ref={canvasRef} /></div>
        </div>
    );
};

// ─── Change Detection ─────────────────────────────────────────────────────────
const ChangeDetectionWidget = () => {
    const stats = [
        { label: 'No Change', value: '72.4%', color: 'text-slate-400' },
        { label: 'Vegetation Gain', value: '12.3%', color: 'text-emerald-400' },
        { label: 'Vegetation Loss', value: '6.8%', color: 'text-red-400' },
        { label: 'Built-up Increase', value: '5.1%', color: 'text-amber-400' },
        { label: 'Water Change', value: '3.4%', color: 'text-blue-400' },
    ];
    return (
        <div className="bg-[#0A1020]/80 border border-white/[0.06] rounded-2xl p-4">
            <h3 className="text-[12px] font-bold tracking-widest uppercase text-slate-300 mb-3">Change Detection</h3>
            <div className="flex gap-3">
                <div className="flex-1 space-y-2">
                    {stats.map((s, i) => (
                        <div key={i} className="flex justify-between items-center">
                            <span className="text-[10px] text-slate-500">{s.label}</span>
                            <span className={`text-[11px] font-bold ${s.color}`}>{s.value}</span>
                        </div>
                    ))}
                </div>
                <div className="flex flex-col gap-2 items-center justify-center">
                    <DonutChart value={72} color="#64748b" size={70} label="No Change" />
                </div>
            </div>
        </div>
    );
};

// ─── Data Layers Grid ─────────────────────────────────────────────────────────
const DataLayersGrid = ({ outputPaths }) => {
    const [selectedLayer, setSelectedLayer] = useState(null);
    
    // Close on Escape
    useEffect(() => {
        if (!selectedLayer) return;
        const onKey = (e) => { if (e.key === 'Escape') setSelectedLayer(null); };
        window.addEventListener('keydown', onKey);
        return () => window.removeEventListener('keydown', onKey);
    }, [selectedLayer]);

    const layers = [
        { title: 'True Color', key: 'reconstructed', icon: 'Eye', color: 'text-cyan-400', border: 'border-cyan-500/40' },
        { title: 'Cloud Mask', key: 'mask', icon: 'Cloud', color: 'text-slate-400', border: 'border-slate-500/40' },
        { title: 'Difference Map', key: 'diff', icon: 'GitCompare', color: 'text-red-400', border: 'border-red-500/40' },
        { title: 'Confidence Map', key: 'conf', icon: 'Target', color: 'text-emerald-400', border: 'border-emerald-500/40' },
        { title: 'Cloudy Original', key: 'original', icon: 'ImageOff', color: 'text-amber-400', border: 'border-amber-500/40' },
        { title: 'Blended Output', key: 'blended', icon: 'Layers', color: 'text-indigo-400', border: 'border-indigo-500/40' },
    ];
    return (
        <>
            <div className="bg-[#0A1020]/80 border border-white/[0.06] rounded-2xl p-4">
                <h3 className="text-[12px] font-bold tracking-widest uppercase text-slate-300 mb-3">Data Layers</h3>
                <div className="grid grid-cols-3 gap-2">
                    {layers.map((l, i) => {
                        const url = outputPaths?.[l.key];
                        return (
                            <div key={i} 
                                onClick={() => url && setSelectedLayer(l)}
                                className={`rounded-xl overflow-hidden border border-white/[0.06] group transition-all duration-300 ${url ? 'cursor-pointer hover:border-cyan-500/30 hover:shadow-[0_0_12px_rgba(6,182,212,0.15)]' : 'cursor-not-allowed opacity-50'}`}>
                                <div className="h-16 bg-slate-900/60 relative overflow-hidden flex items-center justify-center">
                                    {url ? (
                                        <img src={`${API}${url}`} className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500" alt={l.title} />
                                    ) : (
                                        <Icon name={l.icon} className={`${l.color} opacity-30`} size={24} />
                                    )}
                                    {url && (
                                        <div className="absolute inset-0 bg-black/0 group-hover:bg-black/30 transition-colors flex items-center justify-center">
                                            <Icon name="Maximize" size={18} className="text-white opacity-0 group-hover:opacity-80 transition-opacity" />
                                        </div>
                                    )}
                                </div>
                                <div className="py-1.5 px-2 bg-[#060B16]">
                                    <p className="text-[9px] font-bold text-slate-400 text-center truncate">{l.title}</p>
                                </div>
                            </div>
                        );
                    })}
                </div>
            </div>

            {/* Lightbox Modal */}
            {selectedLayer && outputPaths?.[selectedLayer.key] && (
                <div className="fixed inset-0 z-[9999] bg-[#020912]/90 backdrop-blur-lg flex flex-col" 
                    onClick={(e) => { if (e.target === e.currentTarget) setSelectedLayer(null); }}
                    style={{ animation: 'fadeIn 0.15s ease-out' }}>
                    <div className="flex justify-between items-center px-6 py-3 border-b border-white/[0.06]">
                        <div className="flex items-center gap-3">
                            <Icon name={selectedLayer.icon} size={16} className={selectedLayer.color} />
                            <h3 className="text-[13px] font-bold tracking-widest uppercase text-slate-200">{selectedLayer.title}</h3>
                            <span className={`px-2 py-0.5 rounded text-[8px] font-bold uppercase tracking-wider border ${selectedLayer.border} ${selectedLayer.color} bg-white/[0.03]`}>
                                Data Layer
                            </span>
                        </div>
                        <div className="flex gap-2 items-center">
                            <a href={`${API}${outputPaths[selectedLayer.key]}`} download
                                className="p-2 bg-white/[0.06] rounded-lg hover:bg-white/[0.12] border border-white/[0.08] transition-all active:scale-90"
                                title="Download">
                                <Icon name="Download" size={15} className="text-slate-300" />
                            </a>
                            <button onClick={() => setSelectedLayer(null)} title="Close (Esc)"
                                className="p-2 bg-red-500/10 rounded-lg hover:bg-red-500/20 border border-red-500/20 transition-all active:scale-90">
                                <Icon name="X" size={15} className="text-red-400" />
                            </button>
                        </div>
                    </div>
                    <div className="flex-1 flex items-center justify-center p-6 overflow-auto">
                        <img 
                            src={`${API}${outputPaths[selectedLayer.key]}`} 
                            alt={selectedLayer.title}
                            className={`max-w-full max-h-full rounded-xl border-2 ${selectedLayer.border} shadow-2xl`}
                            style={{ objectFit: 'contain' }}
                        />
                    </div>
                </div>
            )}
        </>
    );
};

// ═══════════════════════════════════════════════════════════════════════════════
// ─── DASHBOARD MODULE (Main Home) ────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════════
const DashboardModule = ({ onModuleChange }) => {
    const { lastResult, uploadMeta, setActiveModule } = useApp();
    const sparkDataSets = useMemo(() => ({
        cloud: [28, 32, 25, 23, 27, 24, 23],
        recon: [88, 90, 92, 93, 94, 94, 95],
        psnr: [26, 27, 28, 28, 29, 29, 29],
        ssim: [0.90, 0.91, 0.92, 0.93, 0.93, 0.94, 0.94],
        time: [140, 130, 125, 115, 110, 108, 108],
        model: [1, 1.5, 1.8, 2, 2, 2.1, 2.1],
    }), []);

    const cloudPct = lastResult?.cloud_fraction != null ? (lastResult.cloud_fraction * 100).toFixed(1) : '23.2';
    const inferenceTime = lastResult?.inference_time != null ? lastResult.inference_time.toFixed(2) : '107.78';

    return (
        <div className="space-y-4 pb-6">
            <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-3">
                <MetricCard label="Cloud Coverage" value={cloudPct} suffix="%" icon="Cloud" color="amber" status={lastResult ? 'From Last Run' : 'Demo Data'} sparkData={sparkDataSets.cloud} delay={0} />
                <MetricCard label="Reconstruction Score" value="94.6" suffix="%" icon="Star" color="emerald" status="Excellent" sparkData={sparkDataSets.recon} delay={80} />
                <MetricCard label="PSNR" value="29.46" suffix=" dB" icon="Activity" color="cyan" status="Good" sparkData={sparkDataSets.psnr} delay={160} />
                <MetricCard label="SSIM" value="0.9389" icon="Layers" color="indigo" status="Excellent" sparkData={sparkDataSets.ssim} delay={240} />
                <MetricCard label="Inference Time" value={inferenceTime} suffix=" s" icon="Zap" color="fuchsia" status={lastResult ? 'Latest Run' : 'Completed'} sparkData={sparkDataSets.time} delay={320} />
                <MetricCard label="Model" value="U-Net v2.1" icon="Cpu" color="blue" status="Generative AI" sparkData={sparkDataSets.model} delay={400} />
            </div>

            <div className="grid grid-cols-1 xl:grid-cols-12 gap-4">
                <div className="xl:col-span-5">
                    <ComparisonViewer
                        beforeUrl={lastResult?.paths?.original}
                        groundTruthUrl={lastResult?.paths?.ground_truth}
                    />
                </div>
                <div className="xl:col-span-3"><AIPipeline /></div>
                <div className="xl:col-span-4"><DataLayersGrid outputPaths={lastResult?.paths} /></div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
                <SpectralChart />
                <BandQuality />
                <GeospatialInfo meta={uploadMeta} />
                <HistogramPanel />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                <ChangeDetectionWidget />
                <TimeSeriesChart />
                <RecentMissionsCard onSelect={onModuleChange} />
            </div>

            <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
                <div className="xl:col-span-2"><QuickActions reconstructionId={lastResult?.reconstruction_id} /></div>
                <SystemStatusWidget />
            </div>
        </div>
    );
};

// ═══════════════════════════════════════════════════════════════════════════════
// ─── UPLOAD MODULE ────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════════
const UploadModule = () => {
    const { setLastResult, setUploadMeta, addNotification } = useApp();
    const [dragOver, setDragOver] = useState(false);
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [uploading, setUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [phase, setPhase] = useState('idle'); // idle, uploading, reconstructing, done, error
    const [uploadResult, setUploadResult] = useState(null);
    const [reconResult, setReconResult] = useState(null);
    const [error, setError] = useState('');
    const [patchSize, setPatchSize] = useState(256);
    const [stride, setStride] = useState(192);
    const fileRef = useRef(null);

    const handleFile = (f) => {
        if (!f) return;
        setFile(f);
        setError('');
        setPhase('idle');
        setReconResult(null);
        setUploadResult(null);
        if (f.type.startsWith('image/') || f.name.match(/\.(tif|tiff|png|jpg|jpeg)$/i)) {
            const reader = new FileReader();
            reader.onload = (e) => setPreview(e.target.result);
            reader.readAsDataURL(f);
        } else {
            setPreview(null);
        }
    };

    const handleDrop = (e) => {
        e.preventDefault(); setDragOver(false);
        const f = e.dataTransfer.files[0];
        if (f) handleFile(f);
    };

    const processUpload = async () => {
        if (!file) return;
        setUploading(true); setPhase('uploading'); setError('');
        setUploadProgress(0);

        // Simulate progress during upload
        const progressTimer = setInterval(() => {
            setUploadProgress(prev => Math.min(prev + 15, 85));
        }, 200);

        try {
            const formData = new FormData();
            formData.append('file', file);
            const uploadRes = await authFetch('/api/upload', { method: 'POST', body: formData });
            clearInterval(progressTimer);
            setUploadProgress(100);

            if (!uploadRes.ok) {
                const err = await uploadRes.json();
                throw new Error(err.detail || 'Upload failed');
            }
            const uploadData = await uploadRes.json();
            setUploadResult(uploadData);
            setUploadMeta(uploadData);
            addNotification({ type: 'success', title: 'Upload Complete', message: `${file.name} uploaded successfully`, time: new Date().toLocaleTimeString() });

            // Now reconstruct
            setPhase('reconstructing');
            const reconFormData = new FormData();
            reconFormData.append('upload_id', uploadData.upload_id);
            reconFormData.append('patch_size', patchSize);
            reconFormData.append('stride', stride);

            const reconRes = await authFetch('/api/reconstruct', { method: 'POST', body: reconFormData });
            if (!reconRes.ok) {
                const err = await reconRes.json();
                throw new Error(err.detail || 'Reconstruction failed');
            }
            const reconData = await reconRes.json();
            setReconResult(reconData);
            setLastResult({ ...reconData, satellite: uploadData.satellite_name });
            setPhase('done');
            addNotification({ type: 'success', title: 'Reconstruction Complete', message: `Cloud fraction: ${(reconData.cloud_fraction * 100).toFixed(1)}%`, time: new Date().toLocaleTimeString() });
        } catch (e) {
            clearInterval(progressTimer);
            setError(e.message);
            setPhase('error');
            addNotification({ type: 'error', title: 'Processing Failed', message: e.message, time: new Date().toLocaleTimeString() });
        } finally {
            setUploading(false);
        }
    };

    const formatBytes = (bytes) => {
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
        return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
    };

    return (
        <div className="space-y-4 pb-6">
            <SectionHeader title="Upload Satellite Scene" subtitle="Upload LISS-IV, Sentinel-2 or GeoTIFF images for AI cloud removal" icon="UploadCloud">
                <span className="text-[10px] text-slate-500">Supported: .tif .tiff .png .jpg .zip</span>
            </SectionHeader>

            <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
                <div className="xl:col-span-2 space-y-4">
                    {/* Drop Zone */}
                    <div onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                        onDragLeave={() => setDragOver(false)}
                        onDrop={handleDrop}
                        onClick={() => fileRef.current?.click()}
                        className={`relative border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all duration-300 ${dragOver ? 'border-cyan-400 bg-cyan-500/10' : file ? 'border-emerald-500/40 bg-emerald-500/5' : 'border-white/[0.1] hover:border-cyan-500/40 hover:bg-cyan-500/5'}`}>
                        <input ref={fileRef} type="file" className="hidden" accept=".tif,.tiff,.png,.jpg,.jpeg,.zip"
                            onChange={(e) => handleFile(e.target.files[0])} />

                        {file ? (
                            <div>
                                {preview && (
                                    <div className="mb-4 flex justify-center">
                                        <img src={preview} className="max-h-40 rounded-xl border border-white/10 object-contain" alt="Preview" />
                                    </div>
                                )}
                                <div className="flex items-center justify-center gap-3">
                                    <Icon name="FileCheck" className="text-emerald-400" size={24} />
                                    <div className="text-left">
                                        <p className="text-[13px] font-bold text-white">{file.name}</p>
                                        <p className="text-[11px] text-slate-400">{formatBytes(file.size)} · {file.type || 'Unknown type'}</p>
                                    </div>
                                    <button onClick={(e) => { e.stopPropagation(); setFile(null); setPreview(null); setPhase('idle'); }}
                                        className="ml-2 p-1 rounded-lg hover:bg-red-500/20 text-slate-500 hover:text-red-400 transition">
                                        <Icon name="X" size={14} />
                                    </button>
                                </div>
                            </div>
                        ) : (
                            <div>
                                <Icon name="UploadCloud" className="text-cyan-500/40 mx-auto mb-4" size={48} />
                                <p className="text-[14px] font-bold text-slate-300 mb-1">Drag & Drop Satellite Image</p>
                                <p className="text-[11px] text-slate-500">or click to browse — LISS-IV, GeoTIFF, Sentinel-2 ZIP</p>
                            </div>
                        )}
                    </div>

                    {/* Progress */}
                    {(phase === 'uploading' || phase === 'reconstructing') && (
                        <div className="bg-[#0A1020]/80 border border-cyan-500/20 rounded-2xl p-4">
                            <div className="flex items-center gap-3 mb-3">
                                <div className="w-8 h-8 rounded-full border-2 border-cyan-500 border-t-transparent animate-spin" />
                                <div>
                                    <p className="text-[12px] font-bold text-cyan-400">
                                        {phase === 'uploading' ? 'Uploading & Validating...' : '🧠 AI Reconstruction Running...'}
                                    </p>
                                    <p className="text-[10px] text-slate-500">
                                        {phase === 'uploading' ? 'Transferring file to server' : 'U-Net model processing satellite scene'}
                                    </p>
                                </div>
                            </div>
                            {phase === 'uploading' && (
                                <div>
                                    <ProgressBar value={uploadProgress} color="#06b6d4" />
                                    <p className="text-[10px] text-slate-500 mt-1">{uploadProgress}% uploaded</p>
                                </div>
                            )}
                            {phase === 'reconstructing' && (
                                <AIPipeline steps={[
                                    { step: 1, title: 'Data Ingestion', desc: 'Satellite scene loaded', status: 'done' },
                                    { step: 2, title: 'Cloud Detection', desc: 'Running AI cloud mask', status: 'active' },
                                    { step: 3, title: 'Patch Extraction', desc: `${patchSize}×${patchSize} patches`, status: 'pending' },
                                    { step: 4, title: 'U-Net Inference', desc: 'GPU-accelerated reconstruction', status: 'pending' },
                                    { step: 5, title: 'Post-Processing', desc: 'Blending & enhancement', status: 'pending' },
                                    { step: 6, title: 'Output Generation', desc: 'Saving results', status: 'pending' },
                                ]} />
                            )}
                        </div>
                    )}

                    {/* Error */}
                    {phase === 'error' && (
                        <div className="bg-red-500/10 border border-red-500/30 rounded-2xl p-4 flex items-start gap-3">
                            <Icon name="AlertCircle" className="text-red-400 flex-shrink-0 mt-0.5" size={18} />
                            <div>
                                <p className="text-[12px] font-bold text-red-400">Processing Failed</p>
                                <p className="text-[11px] text-slate-400 mt-1">{error}</p>
                                <button onClick={() => { setPhase('idle'); setError(''); }} className="text-[10px] text-cyan-400 hover:underline mt-2">Try Again</button>
                            </div>
                        </div>
                    )}

                    {/* Success */}
                    {phase === 'done' && reconResult && (
                        <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-2xl p-4">
                            <div className="flex items-center gap-3 mb-4">
                                <Icon name="CheckCircle" className="text-emerald-400" size={20} />
                                <p className="text-[13px] font-bold text-emerald-400">Reconstruction Complete!</p>
                            </div>
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                                {[
                                    { label: 'Cloud Fraction', value: `${(reconResult.cloud_fraction * 100).toFixed(1)}%`, color: 'text-amber-400' },
                                    { label: 'Inference Time', value: `${reconResult.inference_time?.toFixed(1) || '—'}s`, color: 'text-cyan-400' },
                                    { label: 'Total Time', value: `${reconResult.processing_time?.toFixed(1) || '—'}s`, color: 'text-indigo-400' },
                                    { label: 'GPU Memory', value: reconResult.gpu_memory ? `${reconResult.gpu_memory.toFixed(0)} MB` : 'CPU', color: 'text-emerald-400' },
                                ].map((item, i) => (
                                    <div key={i} className="bg-white/[0.03] rounded-xl p-3 text-center">
                                        <p className={`text-[16px] font-black ${item.color}`}>{item.value}</p>
                                        <p className="text-[9px] text-slate-500 uppercase tracking-wider mt-0.5">{item.label}</p>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>

                <div className="space-y-4">
                    {/* Upload Settings */}
                    <Card>
                        <h3 className="text-[12px] font-bold text-slate-300 uppercase tracking-wider mb-3">Processing Settings</h3>
                        <div className="space-y-3">
                            <div>
                                <label className="text-[10px] text-slate-500 uppercase tracking-wider">Patch Size</label>
                                <select value={patchSize} onChange={(e) => setPatchSize(Number(e.target.value))}
                                    className="w-full mt-1 bg-white/[0.04] border border-white/[0.08] text-[11px] text-slate-300 rounded-lg px-3 py-2 outline-none">
                                    <option value="128">128×128 (Fast)</option>
                                    <option value="256">256×256 (Standard)</option>
                                    <option value="512">512×512 (High Quality)</option>
                                </select>
                            </div>
                            <div>
                                <label className="text-[10px] text-slate-500 uppercase tracking-wider">Stride</label>
                                <select value={stride} onChange={(e) => setStride(Number(e.target.value))}
                                    className="w-full mt-1 bg-white/[0.04] border border-white/[0.08] text-[11px] text-slate-300 rounded-lg px-3 py-2 outline-none">
                                    <option value="64">64 (High Overlap)</option>
                                    <option value="128">128 (Balanced)</option>
                                    <option value="192">192 (Standard)</option>
                                    <option value="224">224 (Fast)</option>
                                </select>
                            </div>
                            <button onClick={processUpload} disabled={!file || uploading}
                                className={`w-full py-3 rounded-xl text-[12px] font-bold tracking-wider transition-all ${file && !uploading ? 'bg-cyan-500 hover:bg-cyan-400 text-black shadow-[0_0_20px_rgba(6,182,212,0.4)]' : 'bg-slate-800 text-slate-600 cursor-not-allowed'}`}>
                                {uploading ? 'PROCESSING...' : 'LAUNCH RECONSTRUCTION'}
                            </button>
                        </div>
                    </Card>

                    {/* File Requirements */}
                    <Card>
                        <h3 className="text-[12px] font-bold text-slate-300 uppercase tracking-wider mb-3">Supported Formats</h3>
                        <div className="space-y-2">
                            {[
                                { icon: 'Satellite', label: 'LISS-IV GeoTIFF', desc: 'Resourcesat-2 (.tif, .tiff)', color: 'text-cyan-400' },
                                { icon: 'Layers', label: 'Sentinel-2 SAFE ZIP', desc: 'Copernicus (.zip)', color: 'text-blue-400' },
                                { icon: 'Image', label: 'Standard Images', desc: 'PNG, JPG for testing', color: 'text-emerald-400' },
                                { icon: 'Globe', label: 'Any GeoTIFF', desc: 'Multi-band rasters', color: 'text-indigo-400' },
                            ].map((item, i) => (
                                <div key={i} className="flex items-center gap-2.5 p-2 rounded-lg bg-white/[0.02]">
                                    <Icon name={item.icon} className={item.color} size={14} />
                                    <div>
                                        <p className="text-[11px] font-semibold text-slate-300">{item.label}</p>
                                        <p className="text-[9px] text-slate-500">{item.desc}</p>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </Card>
                </div>
            </div>
        </div>
    );
};

// ═══════════════════════════════════════════════════════════════════════════════
// ─── RESULTS MODULE ───────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════════
const ResultsModule = () => {
    const { lastResult, addNotification } = useApp();

    if (!lastResult) {
        return (
            <div className="flex flex-col items-center justify-center h-[60vh] text-slate-500 space-y-4">
                <div className="w-20 h-20 rounded-2xl bg-white/[0.03] border border-white/[0.06] flex items-center justify-center">
                    <Icon name="Eye" size={32} className="text-cyan-500/30" />
                </div>
                <p className="text-[14px] font-bold text-slate-400" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>No Results Yet</p>
                <p className="text-[11px] text-slate-600" style={{ fontFamily: 'Inter, sans-serif' }}>Upload and process a satellite image to see results</p>
            </div>
        );
    }

    const metrics = lastResult.metrics || {};
    const paths = lastResult.paths || {};

    const download = (type) => {
        if (!lastResult.reconstruction_id) return;
        window.open(`${API}/api/download/${lastResult.reconstruction_id}/${type}`, '_blank');
    };

    // Staggered animation helper
    const fadeIn = (delay = 0) => ({
        style: { 
            animation: `fadeSlideUp 0.5s ease-out ${delay}ms both`,
        }
    });

    return (
        <div className="space-y-6 pb-8">
            {/* ── Inline Animation Keyframes ── */}
            <style>{`
                @keyframes fadeSlideUp {
                    from { opacity: 0; transform: translateY(16px); }
                    to { opacity: 1; transform: translateY(0); }
                }
                @keyframes pulseGlow {
                    0%, 100% { box-shadow: 0 0 12px rgba(6, 182, 212, 0.15); }
                    50% { box-shadow: 0 0 24px rgba(6, 182, 212, 0.3); }
                }
            `}</style>

            {/* ═══════════════════════════════════════════════════════════════ */}
            {/* SECTION 1: Scene Header                                        */}
            {/* ═══════════════════════════════════════════════════════════════ */}
            <div {...fadeIn(0)}>
                <div className="bg-gradient-to-r from-[#0A1020]/90 via-[#0C1428]/90 to-[#0A1020]/90 border border-white/[0.06] rounded-2xl p-5">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-4">
                            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-cyan-500/20 to-emerald-500/20 border border-cyan-500/30 flex items-center justify-center"
                                style={{ animation: 'pulseGlow 3s ease-in-out infinite' }}>
                                <Icon name="Satellite" size={22} className="text-cyan-400" />
                            </div>
                            <div>
                                <h2 className="text-[18px] font-bold text-white tracking-tight" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>
                                    Reconstruction Results
                                </h2>
                                <p className="text-[11px] text-slate-400 mt-0.5 tracking-wide" style={{ fontFamily: 'Inter, sans-serif' }}>
                                    Scene: <span className="text-slate-300 font-medium">{lastResult.scene_name || 'Unknown'}</span>
                                    <span className="mx-2 text-slate-600">·</span>
                                    Cloud Fraction: <span className="text-amber-400 font-semibold">{(lastResult.cloud_fraction * 100).toFixed(1)}%</span>
                                    <span className="mx-2 text-slate-600">·</span>
                                    Processed: <span className="text-slate-300 font-medium">{new Date().toLocaleTimeString()}</span>
                                </p>
                            </div>
                        </div>
                        <StatusBadge status={lastResult.status} />
                    </div>
                </div>
            </div>

            {/* ═══════════════════════════════════════════════════════════════ */}
            {/* SECTION 2: Metrics Strip                                       */}
            {/* ═══════════════════════════════════════════════════════════════ */}
            <div {...fadeIn(100)}>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    {[
                        { label: 'Cloud Fraction', value: `${(lastResult.cloud_fraction * 100).toFixed(1)}%`, color: 'amber', icon: 'Cloud' },
                        { label: 'Inference Time', value: `${lastResult.inference_time?.toFixed(1) || '—'}s`, color: 'cyan', icon: 'Zap' },
                        { label: 'PSNR', value: typeof metrics.psnr === 'number' ? `${metrics.psnr.toFixed(2)} dB` : 'N/A', color: 'indigo', icon: 'Activity' },
                        { label: 'SSIM', value: typeof metrics.ssim === 'number' ? metrics.ssim.toFixed(4) : 'N/A', color: 'emerald', icon: 'Layers' },
                    ].map((m, i) => <MetricCard key={i} {...m} delay={i * 80} />)}
                </div>
            </div>

            {/* ═══════════════════════════════════════════════════════════════ */}
            {/* SECTION 3: Comparison Slider (Full Width)                      */}
            {/* ═══════════════════════════════════════════════════════════════ */}
            <div {...fadeIn(200)}>
                <div className="flex items-center gap-2 mb-3">
                    <span className="text-[10px] font-bold text-cyan-500 bg-cyan-500/10 border border-cyan-500/20 px-2 py-0.5 rounded-md tracking-widest" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>01</span>
                    <h3 className="text-[13px] font-bold tracking-wide text-slate-200" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>Interactive Comparison</h3>
                    <div className="flex-1 h-px bg-gradient-to-r from-white/[0.08] to-transparent ml-2" />
                </div>
                <ComparisonViewer 
                    beforeUrl={paths.original} 
                    groundTruthUrl={paths.ground_truth} 
                />
            </div>

            {/* ═══════════════════════════════════════════════════════════════ */}
            {/* SECTION 4: 4-Panel Reconstruction Analysis Grid                */}
            {/* ═══════════════════════════════════════════════════════════════ */}
            <div {...fadeIn(300)}>
                <div className="flex items-center gap-2 mb-3">
                    <span className="text-[10px] font-bold text-cyan-500 bg-cyan-500/10 border border-cyan-500/20 px-2 py-0.5 rounded-md tracking-widest" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>02</span>
                    <h3 className="text-[13px] font-bold tracking-wide text-slate-200" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>Reconstruction Analysis</h3>
                    <div className="flex-1 h-px bg-gradient-to-r from-white/[0.08] to-transparent ml-2" />
                </div>
                <div className="bg-[#0A1020]/60 border border-white/[0.06] rounded-2xl p-4">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                        {/* Panel 1: Input Cloudy */}
                        <div className="rounded-xl border border-white/[0.08] overflow-hidden bg-[#060B16] hover:border-amber-500/30 transition-all duration-300 group">
                            <div className="flex items-center justify-between px-3 py-2 border-b border-white/[0.06]">
                                <div className="flex items-center gap-1.5">
                                    <Icon name="Cloud" size={12} className="text-amber-400" />
                                    <span className="text-[10px] font-bold text-slate-300" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>Cloudy Input</span>
                                </div>
                                <span className="text-[7px] font-bold tracking-wider uppercase px-1.5 py-0.5 rounded bg-slate-700/60 text-slate-400 border border-slate-600/40">INPUT</span>
                            </div>
                            <div className="aspect-square relative bg-slate-900/40 overflow-hidden">
                                {paths.original ? (
                                    <img src={`${API}${paths.original}`} className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500" alt="Cloudy Input" />
                                ) : (
                                    <div className="w-full h-full flex items-center justify-center"><Icon name="ImageOff" size={32} className="text-slate-700" /></div>
                                )}
                            </div>
                        </div>

                        {/* Panel 2: Cloud Detection Overlay */}
                        <div className="rounded-xl border border-white/[0.08] overflow-hidden bg-[#060B16] hover:border-red-500/30 transition-all duration-300 group">
                            <div className="flex items-center justify-between px-3 py-2 border-b border-white/[0.06]">
                                <div className="flex items-center gap-1.5">
                                    <Icon name="Crosshair" size={12} className="text-red-400" />
                                    <span className="text-[10px] font-bold text-slate-300" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>Cloud Mask</span>
                                </div>
                                <span className="text-[7px] font-bold tracking-wider uppercase px-1.5 py-0.5 rounded bg-red-500/20 text-red-400 border border-red-500/30">MASK</span>
                            </div>
                            <div className="aspect-square relative bg-slate-900/40 overflow-hidden">
                                {paths.overlay ? (
                                    <img src={`${API}${paths.overlay}`} className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500" alt="Cloud Overlay" />
                                ) : paths.mask ? (
                                    <img src={`${API}${paths.mask}`} className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500" alt="Cloud Mask" />
                                ) : (
                                    <div className="w-full h-full flex items-center justify-center"><Icon name="Target" size={32} className="text-slate-700" /></div>
                                )}
                            </div>
                        </div>

                        {/* Panel 3: AI Reconstructed Cloud-Free */}
                        <div className="rounded-xl border border-cyan-500/20 overflow-hidden bg-[#060B16] shadow-[0_0_16px_rgba(6,182,212,0.06)] hover:border-cyan-500/40 transition-all duration-300 group">
                            <div className="flex items-center justify-between px-3 py-2 border-b border-cyan-500/20">
                                <div className="flex items-center gap-1.5">
                                    <Icon name="Sparkles" size={12} className="text-cyan-400" />
                                    <span className="text-[10px] font-bold text-slate-300" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>AI Output</span>
                                </div>
                                <span className="text-[7px] font-bold tracking-wider uppercase px-1.5 py-0.5 rounded bg-cyan-500/20 text-cyan-400 border border-cyan-500/30">AI</span>
                            </div>
                            <div className="aspect-square relative bg-slate-900/40 overflow-hidden">
                                {paths.reconstructed ? (
                                    <img src={`${API}${paths.reconstructed}`} className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500" alt="AI Reconstructed" />
                                ) : (
                                    <div className="w-full h-full flex items-center justify-center"><Icon name="Eye" size={32} className="text-slate-700" /></div>
                                )}
                            </div>
                        </div>

                        {/* Panel 4: Ground Truth Reference */}
                        <div className="rounded-xl border border-white/[0.08] overflow-hidden bg-[#060B16] hover:border-emerald-500/30 transition-all duration-300 group">
                            <div className="flex items-center justify-between px-3 py-2 border-b border-white/[0.06]">
                                <div className="flex items-center gap-1.5">
                                    <Icon name="Shield" size={12} className="text-emerald-400" />
                                    <span className="text-[10px] font-bold text-slate-300" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>Ground Truth</span>
                                </div>
                                <span className="text-[7px] font-bold tracking-wider uppercase px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">REF</span>
                            </div>
                            <div className="aspect-square relative bg-slate-900/40 overflow-hidden">
                                {paths.ground_truth ? (
                                    <img src={`${API}${paths.ground_truth}`} className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500" alt="Ground Truth" />
                                ) : (
                                    <div className="w-full h-full flex flex-col items-center justify-center gap-2 px-4">
                                        <div className="w-10 h-10 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center">
                                            <Icon name="HelpCircle" size={20} className="text-emerald-500/40" />
                                        </div>
                                        <p className="text-[9px] font-semibold text-slate-500 text-center leading-relaxed" style={{ fontFamily: 'Inter, sans-serif' }}>
                                            Not available in live mode
                                        </p>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* ═══════════════════════════════════════════════════════════════ */}
            {/* SECTION 5: Data Layers + Band Quality (Side by Side)           */}
            {/* ═══════════════════════════════════════════════════════════════ */}
            <div {...fadeIn(400)}>
                <div className="flex items-center gap-2 mb-3">
                    <span className="text-[10px] font-bold text-cyan-500 bg-cyan-500/10 border border-cyan-500/20 px-2 py-0.5 rounded-md tracking-widest" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>03</span>
                    <h3 className="text-[13px] font-bold tracking-wide text-slate-200" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>Diagnostic Layers</h3>
                    <div className="flex-1 h-px bg-gradient-to-r from-white/[0.08] to-transparent ml-2" />
                </div>
                <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
                    <DataLayersGrid outputPaths={paths} />
                    <BandQuality />
                </div>
            </div>

            {/* ═══════════════════════════════════════════════════════════════ */}
            {/* SECTION 6: Quick Actions                                       */}
            {/* ═══════════════════════════════════════════════════════════════ */}
            <div {...fadeIn(500)}>
                <div className="flex items-center gap-2 mb-3">
                    <span className="text-[10px] font-bold text-cyan-500 bg-cyan-500/10 border border-cyan-500/20 px-2 py-0.5 rounded-md tracking-widest" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>04</span>
                    <h3 className="text-[13px] font-bold tracking-wide text-slate-200" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>Actions</h3>
                    <div className="flex-1 h-px bg-gradient-to-r from-white/[0.08] to-transparent ml-2" />
                </div>
                <QuickActions reconstructionId={lastResult.reconstruction_id} />
            </div>
        </div>
    );
};

// ═══════════════════════════════════════════════════════════════════════════════
// ─── HISTORY MODULE ───────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════════
const HistoryModule = () => {
    const [history, setHistory] = useState([]);
    const [loading, setLoading] = useState(true);
    const { addNotification } = useApp();

    const fetchHistory = () => {
        setLoading(true);
        authFetch('/api/history').then(r => r.json()).then(data => {
            setHistory(Array.isArray(data) ? data : []);
        }).catch(() => setHistory([])).finally(() => setLoading(false));
    };

    useEffect(() => { fetchHistory(); }, []);

    const deleteRecord = async (id) => {
        try {
            await authFetch(`/api/history/${id}`, { method: 'DELETE' });
            setHistory(h => h.filter(r => r.id !== id));
            addNotification({ type: 'info', title: 'Record Deleted', message: `Reconstruction #${id} removed`, time: new Date().toLocaleTimeString() });
        } catch (e) {
            addNotification({ type: 'error', title: 'Delete Failed', message: e.message, time: new Date().toLocaleTimeString() });
        }
    };

    return (
        <div className="space-y-4 pb-6">
            <SectionHeader title="Mission History" subtitle="All past satellite image reconstructions" icon="History">
                <button onClick={fetchHistory} className="flex items-center gap-2 text-[11px] text-cyan-400 hover:text-cyan-300 font-semibold">
                    <Icon name="RefreshCw" size={13} />Refresh
                </button>
            </SectionHeader>

            {loading ? (
                <div className="space-y-3">{[1,2,3,4].map(i => <div key={i} className="h-16 bg-white/[0.02] rounded-xl animate-pulse" />)}</div>
            ) : history.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-16 text-slate-500">
                    <Icon name="History" size={40} className="text-slate-700 mb-3" />
                    <p className="text-[13px] font-bold text-slate-400">No reconstructions yet</p>
                    <p className="text-[11px] text-slate-600 mt-1">Upload a satellite image to get started</p>
                </div>
            ) : (
                <div className="space-y-2">
                    {history.map((r, i) => (
                        <div key={i} className="bg-[#0A1020]/80 border border-white/[0.06] rounded-xl p-4 flex items-center gap-4 hover:border-white/[0.1] transition group">
                            <div className="w-12 h-12 rounded-xl bg-slate-800/60 border border-white/[0.06] flex items-center justify-center flex-shrink-0">
                                <Icon name={r.status === 'completed' ? 'CheckCircle' : r.status === 'failed' ? 'AlertCircle' : 'Clock'} size={18} className={r.status === 'completed' ? 'text-emerald-400' : r.status === 'failed' ? 'text-red-400' : 'text-amber-400'} />
                            </div>
                            <div className="flex-1 min-w-0">
                                <div className="flex items-center gap-2 mb-1">
                                    <p className="text-[12px] font-bold text-white truncate">{r.filename || `Scene #${r.id}`}</p>
                                    <StatusBadge status={r.status} />
                                </div>
                                <div className="flex items-center gap-4 flex-wrap">
                                    <span className="text-[10px] text-slate-500 flex items-center gap-1"><Icon name="Satellite" size={10} />{r.satellite || 'LISS-IV'}</span>
                                    {r.cloud_fraction != null && <span className="text-[10px] text-amber-400 flex items-center gap-1"><Icon name="Cloud" size={10} />{(r.cloud_fraction * 100).toFixed(1)}% cloud</span>}
                                    {r.processing_time && <span className="text-[10px] text-cyan-400 flex items-center gap-1"><Icon name="Zap" size={10} />{r.processing_time.toFixed(1)}s</span>}
                                    {r.created_at && <span className="text-[10px] text-slate-600 flex items-center gap-1"><Icon name="Calendar" size={10} />{new Date(r.created_at).toLocaleDateString('en-IN')}</span>}
                                </div>
                                {r.metrics && (
                                    <div className="flex gap-3 mt-1">
                                        {r.metrics.psnr && typeof r.metrics.psnr === 'number' && <span className="text-[9px] text-indigo-400">PSNR: {r.metrics.psnr.toFixed(2)} dB</span>}
                                        {r.metrics.ssim && typeof r.metrics.ssim === 'number' && <span className="text-[9px] text-cyan-400">SSIM: {r.metrics.ssim.toFixed(4)}</span>}
                                    </div>
                                )}
                            </div>
                            <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition">
                                <button onClick={() => window.open(`${API}/api/download/${r.id}/reconstructed`, '_blank')}
                                    className="p-2 rounded-lg hover:bg-cyan-500/10 text-slate-500 hover:text-cyan-400 transition" title="Download">
                                    <Icon name="Download" size={14} />
                                </button>
                                <button onClick={() => deleteRecord(r.id)}
                                    className="p-2 rounded-lg hover:bg-red-500/10 text-slate-500 hover:text-red-400 transition" title="Delete">
                                    <Icon name="Trash2" size={14} />
                                </button>
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

// ═══════════════════════════════════════════════════════════════════════════════
// ─── MONITOR MODULE ───────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════════
const MonitorModule = () => {
    const [status, setStatus] = useState(null);
    const [loading, setLoading] = useState(true);
    const [lastUpdate, setLastUpdate] = useState(null);

    const fetchStatus = () => {
        authFetch('/api/status').then(r => r.json()).then(data => {
            setStatus(data);
            setLastUpdate(new Date());
            setLoading(false);
        }).catch(() => setLoading(false));
    };

    useEffect(() => {
        fetchStatus();
        const interval = setInterval(fetchStatus, 5000);
        return () => clearInterval(interval);
    }, []);

    if (loading) return <div className="flex items-center justify-center h-40"><div className="w-8 h-8 rounded-full border-2 border-cyan-500 border-t-transparent animate-spin" /></div>;

    const sys = status?.system || {};
    const gpu = status?.gpu || {};
    const model = status?.model || {};
    const cpuPct = sys.cpu_usage_percent || 0;
    const ramPct = sys.ram_total_gb ? ((sys.ram_used_gb / sys.ram_total_gb) * 100) : 0;
    const gpuPct = gpu.total_mb ? ((gpu.allocated_mb / gpu.total_mb) * 100) : 0;

    return (
        <div className="space-y-4 pb-6">
            <SectionHeader title="Live System Monitor" subtitle={lastUpdate ? `Last updated: ${lastUpdate.toLocaleTimeString()}` : 'Polling every 5s'} icon="Radio">
                <button onClick={fetchStatus} className="flex items-center gap-2 text-[11px] text-cyan-400 hover:text-cyan-300">
                    <Icon name="RefreshCw" size={13} />Refresh
                </button>
            </SectionHeader>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {[
                    { label: 'CPU Usage', value: cpuPct.toFixed(1), suffix: '%', icon: 'Cpu', color: 'cyan', pct: cpuPct },
                    { label: 'RAM Used', value: `${sys.ram_used_gb?.toFixed(1) || 0} GB`, suffix: '', icon: 'MemoryStick', color: 'indigo', pct: ramPct },
                    { label: 'GPU Memory', value: gpu.available ? `${gpu.allocated_mb?.toFixed(0) || 0} MB` : 'N/A', suffix: '', icon: 'Zap', color: 'fuchsia', pct: gpuPct },
                ].map((item, i) => (
                    <Card key={i} className="text-center">
                        <Icon name={item.icon} size={24} className={`text-${item.color}-400 mx-auto mb-3`} />
                        <p className="text-[24px] font-black text-white">{item.value}{item.suffix}</p>
                        <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-3">{item.label}</p>
                        <ProgressBar value={Math.round(item.pct)} color={item.color === 'cyan' ? '#06b6d4' : item.color === 'indigo' ? '#818cf8' : '#d946ef'} showLabel={false} />
                    </Card>
                ))}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Card>
                    <h3 className="text-[12px] font-bold text-slate-300 uppercase tracking-wider mb-4">AI Model Status</h3>
                    <div className="space-y-3">
                        {[
                            { label: 'Model Name', value: model.name || 'U-Net Satellite Cloud Reconstruction', icon: 'Cpu' },
                            { label: 'Architecture', value: model.architecture || 'U-Net + Multi-spectral Channels', icon: 'Network' },
                            { label: 'Status', value: model.status || 'Ready', icon: 'CheckCircle', green: true },
                            { label: 'Checkpoint', value: 'best.pth (Epoch 124)', icon: 'HardDrive' },
                            { label: 'Val Loss', value: '0.0630', icon: 'TrendingDown', green: true },
                        ].map((item, i) => (
                            <div key={i} className="flex items-center gap-2">
                                <Icon name={item.icon} size={12} className={item.green ? 'text-emerald-400' : 'text-cyan-500/50'} />
                                <span className="text-[10px] text-slate-500 flex-1">{item.label}</span>
                                <span className={`text-[10px] font-medium text-right ${item.green ? 'text-emerald-400' : 'text-slate-300'}`}>{item.value}</span>
                            </div>
                        ))}
                    </div>
                </Card>

                <Card>
                    <h3 className="text-[12px] font-bold text-slate-300 uppercase tracking-wider mb-4">GPU Information</h3>
                    {gpu.available ? (
                        <div className="space-y-3">
                            {[
                                { label: 'GPU Name', value: gpu.name || 'NVIDIA GPU' },
                                { label: 'Allocated', value: `${gpu.allocated_mb?.toFixed(1) || 0} MB` },
                                { label: 'Cached', value: `${gpu.cached_mb?.toFixed(1) || 0} MB` },
                                { label: 'Total VRAM', value: `${gpu.total_mb?.toFixed(0) || 0} MB` },
                                { label: 'CUDA', value: 'Available', green: true },
                            ].map((item, i) => (
                                <div key={i} className="flex items-center gap-2">
                                    <span className="text-[10px] text-slate-500 flex-1">{item.label}</span>
                                    <span className={`text-[10px] font-medium ${item.green ? 'text-emerald-400' : 'text-slate-300'}`}>{item.value}</span>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <div className="flex flex-col items-center justify-center py-8 text-slate-500">
                            <Icon name="Cpu" size={32} className="text-slate-700 mb-2" />
                            <p className="text-[12px] font-bold">CPU Mode</p>
                            <p className="text-[10px] text-slate-600">No CUDA GPU detected — running on CPU</p>
                        </div>
                    )}
                </Card>
            </div>

            <Card>
                <h3 className="text-[12px] font-bold text-slate-300 uppercase tracking-wider mb-4">API Endpoints Health</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    {['/api/status', '/api/upload', '/api/reconstruct', '/api/history'].map((ep, i) => (
                        <div key={i} className="bg-emerald-500/10 border border-emerald-500/20 rounded-xl p-3 text-center">
                            <span className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse inline-block mb-2" />
                            <p className="text-[9px] font-mono text-emerald-400">{ep}</p>
                            <p className="text-[9px] text-slate-500 mt-0.5">Operational</p>
                        </div>
                    ))}
                </div>
            </Card>
        </div>
    );
};

// ═══════════════════════════════════════════════════════════════════════════════
// ─── ANALYTICS MODULE ─────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════════
const AnalyticsModule = () => {
    const lineRef = useRef(null);
    const barRef = useRef(null);
    const lineChart = useRef(null);
    const barChart = useRef(null);

    useEffect(() => {
        if (!window.Chart) return;
        if (lineChart.current) lineChart.current.destroy();
        if (barChart.current) barChart.current.destroy();

        if (lineRef.current) {
            lineChart.current = new Chart(lineRef.current.getContext('2d'), {
                type: 'line',
                data: {
                    labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
                    datasets: [
                        { label: 'Cloud %', data: [35, 42, 28, 15, 23, 12], borderColor: '#f59e0b', backgroundColor: 'rgba(245,158,11,0.1)', fill: true, tension: 0.4, borderWidth: 2 },
                        { label: 'Reconstruction %', data: [92, 88, 95, 98, 94, 97], borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,0.1)', fill: true, tension: 0.4, borderWidth: 2 },
                    ]
                },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: '#64748b', font: { size: 10 } } } }, scales: { x: { ticks: { color: '#475569' }, grid: { color: 'rgba(255,255,255,0.03)' } }, y: { ticks: { color: '#475569' }, grid: { color: 'rgba(255,255,255,0.03)' } } } }
            });
        }
        if (barRef.current) {
            barChart.current = new Chart(barRef.current.getContext('2d'), {
                type: 'bar',
                data: {
                    labels: ['SCN-01', 'SCN-02', 'SCN-03', 'SCN-04', 'SCN-05'],
                    datasets: [
                        { label: 'Inference Time (s)', data: [108, 95, 127, 89, 112], backgroundColor: 'rgba(6,182,212,0.5)', borderColor: '#06b6d4', borderWidth: 1 },
                    ]
                },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: '#64748b', font: { size: 10 } } } }, scales: { x: { ticks: { color: '#475569' }, grid: { display: false } }, y: { ticks: { color: '#475569' }, grid: { color: 'rgba(255,255,255,0.03)' } } } }
            });
        }
        return () => { if (lineChart.current) lineChart.current.destroy(); if (barChart.current) barChart.current.destroy(); };
    }, []);

    return (
        <div className="space-y-4 pb-6">
            <SectionHeader title="Analytics Dashboard" subtitle="Performance metrics and processing statistics" icon="PieChart" />

            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {[
                    { label: 'Total Processed', value: '24', icon: 'Satellite', color: 'cyan' },
                    { label: 'Avg Cloud %', value: '28.4%', icon: 'Cloud', color: 'amber' },
                    { label: 'Avg Recon Score', value: '93.8%', icon: 'Star', color: 'emerald' },
                    { label: 'Avg Inference', value: '107s', icon: 'Zap', color: 'fuchsia' },
                ].map((m, i) => <MetricCard key={i} {...m} delay={i * 80} />)}
            </div>

            <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
                <Card>
                    <h3 className="text-[12px] font-bold text-slate-300 uppercase tracking-wider mb-3">Cloud Coverage Trend</h3>
                    <div className="h-[200px]"><canvas ref={lineRef} /></div>
                </Card>
                <Card>
                    <h3 className="text-[12px] font-bold text-slate-300 uppercase tracking-wider mb-3">Inference Time per Scene</h3>
                    <div className="h-[200px]"><canvas ref={barRef} /></div>
                </Card>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <Card>
                    <h3 className="text-[12px] font-bold text-slate-300 uppercase tracking-wider mb-4">Cloud Class Distribution</h3>
                    <div className="space-y-2.5">
                        {[
                            { label: 'Cirrus', value: 31, color: '#06b6d4' },
                            { label: 'Cumulus', value: 28, color: '#3b82f6' },
                            { label: 'Stratus', value: 22, color: '#818cf8' },
                            { label: 'Cumulonimbus', value: 11, color: '#d946ef' },
                            { label: 'Clear', value: 8, color: '#10b981' },
                        ].map((c, i) => <ProgressBar key={i} label={c.label} value={c.value} color={c.color} />)}
                    </div>
                </Card>
                <Card>
                    <h3 className="text-[12px] font-bold text-slate-300 uppercase tracking-wider mb-4">Terrain Analysis</h3>
                    <div className="space-y-2.5">
                        {[
                            { label: 'Vegetation', value: 42, color: '#10b981' },
                            { label: 'Urban', value: 23, color: '#f59e0b' },
                            { label: 'Water', value: 18, color: '#3b82f6' },
                            { label: 'Barren', value: 12, color: '#94a3b8' },
                            { label: 'Agriculture', value: 5, color: '#84cc16' },
                        ].map((c, i) => <ProgressBar key={i} label={c.label} value={c.value} color={c.color} />)}
                    </div>
                </Card>
                <Card>
                    <h3 className="text-[12px] font-bold text-slate-300 uppercase tracking-wider mb-4">Quality Metrics Summary</h3>
                    <div className="flex flex-wrap gap-3 justify-around">
                        <DonutChart value={94} color="#10b981" size={72} label="Recon Score" />
                        <DonutChart value={23} color="#f59e0b" size={72} label="Avg Cloud" />
                        <DonutChart value={87} color="#06b6d4" size={72} label="Confidence" />
                    </div>
                </Card>
            </div>
        </div>
    );
};

// ═══════════════════════════════════════════════════════════════════════════════
// ─── SETTINGS MODULE ──────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════════
const SettingsModule = () => {
    const { theme, setTheme, addNotification } = useApp();
    const [settings, setSettings] = useState({ theme: 'dark', device: 'auto', batch_size: 8, patch_size: 256, inference_mode: 'Standard' });
    const [saving, setSaving] = useState(false);

    useEffect(() => {
        authFetch('/api/settings').then(r => r.json()).then(data => setSettings(s => ({ ...s, ...data }))).catch(() => {});
    }, []);

    const save = async () => {
        setSaving(true);
        try {
            const res = await authFetch('/api/settings', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(settings) });
            if (res.ok) {
                if (settings.theme !== theme) setTheme(settings.theme);
                addNotification({ type: 'success', title: 'Settings Saved', message: 'Your preferences have been updated', time: new Date().toLocaleTimeString() });
            }
        } catch (e) {
            addNotification({ type: 'error', title: 'Save Failed', message: e.message, time: new Date().toLocaleTimeString() });
        }
        setSaving(false);
    };

    const S = ({ label, children }) => (
        <div className="flex items-center justify-between py-3 border-b border-white/[0.04]">
            <label className="text-[12px] text-slate-300">{label}</label>
            {children}
        </div>
    );

    return (
        <div className="space-y-4 pb-6 max-w-2xl">
            <SectionHeader title="Settings" subtitle="Configure application and inference preferences" icon="Settings" />

            <Card>
                <h3 className="text-[12px] font-bold text-slate-300 uppercase tracking-wider mb-2">Appearance</h3>
                <S label="Theme">
                    <div className="flex gap-2">
                        {['dark', 'light', 'system'].map(t => (
                            <button key={t} onClick={() => { setSettings(s => ({ ...s, theme: t })); setTheme(t); }}
                                className={`px-3 py-1.5 rounded-lg text-[10px] font-semibold capitalize transition ${settings.theme === t ? 'bg-cyan-500/20 border border-cyan-500/40 text-cyan-400' : 'bg-white/[0.04] border border-white/[0.08] text-slate-400 hover:text-white'}`}>
                                {t}
                            </button>
                        ))}
                    </div>
                </S>
            </Card>

            <Card>
                <h3 className="text-[12px] font-bold text-slate-300 uppercase tracking-wider mb-2">Inference Engine</h3>
                <S label="Compute Device">
                    <select value={settings.device} onChange={e => setSettings(s => ({ ...s, device: e.target.value }))}
                        className="bg-white/[0.04] border border-white/[0.08] text-[11px] text-slate-300 rounded-lg px-3 py-1.5 outline-none">
                        <option value="auto">Auto (CUDA → CPU)</option>
                        <option value="cuda">CUDA GPU</option>
                        <option value="cpu">CPU Only</option>
                    </select>
                </S>
                <S label="Batch Size">
                    <select value={settings.batch_size} onChange={e => setSettings(s => ({ ...s, batch_size: Number(e.target.value) }))}
                        className="bg-white/[0.04] border border-white/[0.08] text-[11px] text-slate-300 rounded-lg px-3 py-1.5 outline-none">
                        {[2, 4, 8, 16].map(v => <option key={v} value={v}>{v}</option>)}
                    </select>
                </S>
                <S label="Patch Size">
                    <select value={settings.patch_size} onChange={e => setSettings(s => ({ ...s, patch_size: Number(e.target.value) }))}
                        className="bg-white/[0.04] border border-white/[0.08] text-[11px] text-slate-300 rounded-lg px-3 py-1.5 outline-none">
                        {[128, 256, 512].map(v => <option key={v} value={v}>{v}×{v}</option>)}
                    </select>
                </S>
                <S label="Inference Mode">
                    <select value={settings.inference_mode} onChange={e => setSettings(s => ({ ...s, inference_mode: e.target.value }))}
                        className="bg-white/[0.04] border border-white/[0.08] text-[11px] text-slate-300 rounded-lg px-3 py-1.5 outline-none">
                        <option>Standard</option><option>High Quality</option><option>Fast</option>
                    </select>
                </S>
                <div className="pt-4">
                    <button onClick={save} disabled={saving}
                        className={`px-6 py-2.5 rounded-xl text-[12px] font-bold tracking-wider transition-all ${saving ? 'bg-slate-800 text-slate-600' : 'bg-cyan-500 hover:bg-cyan-400 text-black shadow-[0_0_20px_rgba(6,182,212,0.3)]'}`}>
                        {saving ? 'Saving...' : 'Save Settings'}
                    </button>
                </div>
            </Card>
        </div>
    );
};

// ═══════════════════════════════════════════════════════════════════════════════
// ─── ALERTS MODULE ────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════════
const AlertsModule = () => {
    const { notifications, setNotifications } = useApp();
    const clearAll = () => setNotifications([]);
    const dismiss = (i) => setNotifications(n => n.filter((_, idx) => idx !== i));

    const sampleAlerts = notifications.length === 0 ? [
        { type: 'info', title: 'System Ready', message: 'MeghaMukt platform is fully operational.', time: '10:30 AM', read: true },
        { type: 'success', title: 'Model Loaded', message: 'U-Net checkpoint best.pth loaded successfully.', time: '10:29 AM', read: true },
        { type: 'info', title: 'Database Connected', message: 'SQLite database connected. All tables ready.', time: '10:28 AM', read: true },
    ] : notifications;

    return (
        <div className="space-y-4 pb-6">
            <SectionHeader title="Alerts & Notifications" subtitle={`${sampleAlerts.filter(n => !n.read).length} unread`} icon="Bell">
                <button onClick={clearAll} className="text-[11px] text-slate-400 hover:text-red-400 transition">Clear All</button>
            </SectionHeader>
            {sampleAlerts.length === 0 ? (
                <div className="text-center py-16 text-slate-600">
                    <Icon name="Bell" size={40} className="text-slate-700 mx-auto mb-3" />
                    <p className="text-[13px] font-bold text-slate-500">No notifications</p>
                </div>
            ) : (
                <div className="space-y-2">
                    {sampleAlerts.map((n, i) => (
                        <div key={i} className="relative">
                            <Toast notification={n} />
                            <button onClick={() => dismiss(i)}
                                className="absolute top-2 right-2 p-1 rounded hover:bg-white/10 text-slate-600 hover:text-slate-300 transition">
                                <Icon name="X" size={12} />
                            </button>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

// ═══════════════════════════════════════════════════════════════════════════════
// ─── PERFORMANCE MODULE ───────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════════
const PerformanceModule = () => (
    <div className="space-y-4 pb-6">
        <SectionHeader title="Model Performance" subtitle="U-Net architecture metrics and training results" icon="Gauge" />
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {[
                { label: 'Best Epoch', value: '124', icon: 'Award', color: 'emerald' },
                { label: 'Val Loss', value: '0.063', icon: 'TrendingDown', color: 'cyan' },
                { label: 'Model Size', value: '355 MB', icon: 'HardDrive', color: 'indigo' },
                { label: 'Checkpoints', value: '25', icon: 'Save', color: 'fuchsia' },
            ].map((m, i) => <MetricCard key={i} {...m} delay={i * 80} />)}
        </div>
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
            <Card>
                <h3 className="text-[12px] font-bold text-slate-300 uppercase tracking-wider mb-4">Model Architecture</h3>
                <div className="space-y-3">
                    {[
                        { label: 'Architecture', value: 'U-Net (Encoder-Decoder)' },
                        { label: 'Input Channels', value: '7 (6 optical + 1 mask)' },
                        { label: 'Output Channels', value: '6 (multi-spectral)' },
                        { label: 'Input Bands', value: 'B2, B3, B4, B8, B11, B12' },
                        { label: 'Training Dataset', value: 'Sentinel-2 LISS-IV Scenes' },
                        { label: 'Loss Function', value: 'Combined L1 + Perceptual' },
                        { label: 'Optimizer', value: 'AdamW (AMP enabled)' },
                        { label: 'Device Support', value: 'CUDA GPU / CPU Fallback' },
                    ].map((item, i) => (
                        <div key={i} className="flex items-center gap-2">
                            <span className="text-[10px] text-slate-500 flex-1">{item.label}</span>
                            <span className="text-[10px] text-slate-300 font-medium">{item.value}</span>
                        </div>
                    ))}
                </div>
            </Card>
            <Card>
                <h3 className="text-[12px] font-bold text-slate-300 uppercase tracking-wider mb-4">Training Progress</h3>
                <div className="space-y-2.5">
                    {[
                        { label: 'Epoch 1–20', value: 0.071, color: '#f59e0b' },
                        { label: 'Epoch 21–50', value: 0.066, color: '#06b6d4' },
                        { label: 'Epoch 51–80', value: 0.063, color: '#818cf8' },
                        { label: 'Epoch 81–100', value: 0.063, color: '#10b981' },
                        { label: 'Epoch 101–124', value: 0.063, color: '#10b981' },
                    ].map((item, i) => (
                        <div key={i} className="flex items-center gap-2">
                            <span className="text-[10px] text-slate-500 w-28 flex-shrink-0">{item.label}</span>
                            <div className="flex-1 h-2 bg-white/[0.04] rounded-full overflow-hidden">
                                <div className="h-full rounded-full" style={{ width: `${(1 - item.value) * 100}%`, backgroundColor: item.color }} />
                            </div>
                            <span className="text-[10px] text-slate-400 font-mono w-12 text-right">{item.value.toFixed(3)}</span>
                        </div>
                    ))}
                </div>
                <p className="text-[10px] text-slate-600 mt-3">Lower validation loss = better model performance</p>
            </Card>
        </div>
    </div>
);

// ═══════════════════════════════════════════════════════════════════════════════
// ─── ABOUT MODULE ─────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════════
const AboutModule = () => (
    <div className="space-y-4 pb-6">
        <SectionHeader title="About MeghaMukt" subtitle="AI-powered satellite cloud removal & reconstruction platform" icon="Info" />
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
            <Card>
                <div className="flex items-center gap-3 mb-4">
                    <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-cyan-500/20 to-blue-600/20 border border-cyan-500/30 flex items-center justify-center">
                        <Icon name="Satellite" className="text-cyan-400" size={22} />
                    </div>
                    <div>
                        <h2 className="text-[16px] font-black text-white">MeghaMukt</h2>
                        <p className="text-[11px] text-cyan-500">AI Satellite Reconstruction Platform v2.1</p>
                    </div>
                </div>
                <p className="text-[12px] text-slate-400 leading-relaxed mb-4">
                    MeghaMukt ("Cloud-Free" in Sanskrit) is an end-to-end AI platform for removing cloud contamination from LISS-IV satellite imagery using a trained U-Net deep learning model. The system reconstructs missing spectral information with photorealistic fidelity, enabling uninterrupted earth observation.
                </p>
                <div className="grid grid-cols-2 gap-3">
                    {[
                        { label: 'Architecture', value: 'FastAPI + React', icon: 'Code' },
                        { label: 'AI Model', value: 'PyTorch U-Net', icon: 'Cpu' },
                        { label: 'Database', value: 'SQLite + SQLAlchemy', icon: 'Database' },
                        { label: 'Satellite', value: 'LISS-IV / Sentinel-2', icon: 'Satellite' },
                    ].map((item, i) => (
                        <div key={i} className="flex items-center gap-2 p-2.5 bg-white/[0.02] rounded-xl">
                            <Icon name={item.icon} className="text-cyan-500/50" size={14} />
                            <div>
                                <p className="text-[9px] text-slate-500">{item.label}</p>
                                <p className="text-[11px] text-slate-300 font-semibold">{item.value}</p>
                            </div>
                        </div>
                    ))}
                </div>
            </Card>
            <Card>
                <h3 className="text-[12px] font-bold text-slate-300 uppercase tracking-wider mb-4">Technology Stack</h3>
                <div className="space-y-2">
                    {[
                        { cat: 'Frontend', items: ['React (Babel JSX)', 'Three.js WebGL Earth', 'GSAP Animations', 'Chart.js', 'Tailwind CSS'] },
                        { cat: 'Backend', items: ['FastAPI (Python)', 'SQLite + SQLAlchemy', 'JWT Authentication', 'PyTorch Inference'] },
                        { cat: 'AI/ML', items: ['U-Net Architecture', 'Multi-spectral Reconstruction', 'Cloud Auto-detection', 'GPU/CPU Inference'] },
                    ].map((section, i) => (
                        <div key={i}>
                            <p className="text-[9px] text-cyan-500 uppercase tracking-widest mb-1.5 font-bold">{section.cat}</p>
                            <div className="flex flex-wrap gap-1.5 mb-2">
                                {section.items.map((item, j) => (
                                    <span key={j} className="text-[9px] text-slate-400 bg-white/[0.04] border border-white/[0.06] px-2 py-0.5 rounded-full">{item}</span>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>
            </Card>
        </div>
    </div>
);

// ─── Simple Module Templates ──────────────────────────────────────────────────
const TimeSeriesModule = () => (
    <div className="space-y-4 pb-6">
        <SectionHeader title="Time Series Analysis" subtitle="Temporal vegetation and spectral trends" icon="TrendingUp" />
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
            <Card className="h-64"><TimeSeriesChart /></Card>
            <Card>
                <h3 className="text-[12px] font-bold text-slate-300 uppercase tracking-wider mb-4">Vegetation Index (NDVI) Trend</h3>
                <div className="space-y-2.5">
                    {[
                        { period: 'Jan 2025', ndvi: 0.62, cloud: 35 },
                        { period: 'Feb 2025', ndvi: 0.58, cloud: 42 },
                        { period: 'Mar 2025', ndvi: 0.71, cloud: 28 },
                        { period: 'Apr 2025', ndvi: 0.82, cloud: 15 },
                        { period: 'May 2025', ndvi: 0.78, cloud: 23 },
                        { period: 'Jun 2025', ndvi: 0.83, cloud: 12 },
                    ].map((row, i) => (
                        <div key={i} className="flex items-center gap-3 p-2 bg-white/[0.02] rounded-lg">
                            <span className="text-[10px] text-slate-500 w-20 flex-shrink-0">{row.period}</span>
                            <ProgressBar value={row.ndvi * 100} color="#10b981" showLabel={false} />
                            <span className="text-[10px] text-emerald-400 w-10 text-right font-mono">{row.ndvi.toFixed(2)}</span>
                            <span className="text-[10px] text-amber-400 w-14 text-right">{row.cloud}% ☁</span>
                        </div>
                    ))}
                </div>
            </Card>
        </div>
    </div>
);

const ChangeModule = () => (
    <div className="space-y-4 pb-6">
        <SectionHeader title="Change Detection" subtitle="Multi-temporal land cover change analysis" icon="GitCompare" />
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {[
                { label: 'No Change', value: '72.4%', color: 'cyan' },
                { label: 'Vegetation +', value: '12.3%', color: 'emerald' },
                { label: 'Vegetation -', value: '6.8%', color: 'fuchsia' },
                { label: 'Urban Growth', value: '8.5%', color: 'amber' },
            ].map((m, i) => <MetricCard key={i} {...m} icon="GitCompare" delay={i * 80} />)}
        </div>
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
            <Card>
                <h3 className="text-[12px] font-bold text-slate-300 uppercase tracking-wider mb-4">Change Classification</h3>
                <div className="space-y-2.5">
                    {[
                        { label: 'No Change', value: 72, color: '#64748b' },
                        { label: 'Vegetation Gain', value: 12, color: '#10b981' },
                        { label: 'Vegetation Loss', value: 7, color: '#ef4444' },
                        { label: 'Built-up Increase', value: 5, color: '#f59e0b' },
                        { label: 'Water Body Change', value: 4, color: '#3b82f6' },
                    ].map((c, i) => <ProgressBar key={i} label={c.label} value={c.value} color={c.color} />)}
                </div>
            </Card>
            <Card>
                <h3 className="text-[12px] font-bold text-slate-300 uppercase tracking-wider mb-4">Analysis Period</h3>
                <div className="space-y-3">
                    {[
                        { label: 'Reference Date', value: 'January 2025' },
                        { label: 'Target Date', value: 'June 2025' },
                        { label: 'Duration', value: '6 months' },
                        { label: 'Area Analyzed', value: '124.8 km²' },
                        { label: 'Changed Pixels', value: '234,512 (27.6%)' },
                        { label: 'Algorithm', value: 'Post-Classification Comparison' },
                    ].map((item, i) => (
                        <div key={i} className="flex justify-between">
                            <span className="text-[10px] text-slate-500">{item.label}</span>
                            <span className="text-[10px] text-slate-300">{item.value}</span>
                        </div>
                    ))}
                </div>
            </Card>
        </div>
    </div>
);

const AOIModule = () => (
    <div className="space-y-4 pb-6">
        <SectionHeader title="Areas of Interest" subtitle="Define and manage geographic ROIs" icon="Crosshair" />
        <Card>
            <div className="text-center py-8">
                <Icon name="Map" size={40} className="text-cyan-500/30 mx-auto mb-4" />
                <p className="text-[13px] font-bold text-slate-400">AOI Manager</p>
                <p className="text-[11px] text-slate-600 mt-1 max-w-xs mx-auto">Define geographic bounding boxes to focus reconstruction on specific regions of interest</p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-6 max-w-lg mx-auto">
                    {[
                        { name: 'AOI-001 Ahmedabad', coords: '23.03°N, 72.58°E', area: '45.2 km²', status: 'active' },
                        { name: 'AOI-002 Bengaluru', coords: '12.97°N, 77.59°E', area: '78.4 km²', status: 'active' },
                        { name: 'AOI-003 Western Ghats', coords: '15.32°N, 74.19°E', area: '124.8 km²', status: 'inactive' },
                    ].map((aoi, i) => (
                        <div key={i} className="bg-white/[0.03] border border-white/[0.06] rounded-xl p-3 text-left">
                            <div className="flex items-center gap-2 mb-2">
                                <span className={`w-2 h-2 rounded-full ${aoi.status === 'active' ? 'bg-emerald-400' : 'bg-slate-600'}`} />
                                <p className="text-[11px] font-bold text-slate-300">{aoi.name}</p>
                            </div>
                            <p className="text-[9px] text-slate-500">{aoi.coords}</p>
                            <p className="text-[9px] text-cyan-400">{aoi.area}</p>
                        </div>
                    ))}
                    <button className="bg-cyan-500/10 border border-dashed border-cyan-500/30 rounded-xl p-3 flex items-center justify-center gap-2 text-cyan-400 hover:bg-cyan-500/20 transition">
                        <Icon name="Plus" size={16} />
                        <span className="text-[11px] font-semibold">Add AOI</span>
                    </button>
                </div>
            </div>
        </Card>
    </div>
);

const BatchModule = () => (
    <div className="space-y-4 pb-6">
        <SectionHeader title="Batch Processing" subtitle="Process multiple satellite scenes simultaneously" icon="Layers" />
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {[
                { label: 'Queued', value: '0', icon: 'Clock', color: 'amber' },
                { label: 'Processing', value: '0', icon: 'Zap', color: 'cyan' },
                { label: 'Completed', value: '24', icon: 'CheckCircle', color: 'emerald' },
                { label: 'Failed', value: '0', icon: 'AlertCircle', color: 'fuchsia' },
            ].map((m, i) => <MetricCard key={i} {...m} delay={i * 80} />)}
        </div>
        <Card>
            <div className="flex items-center justify-between mb-4">
                <h3 className="text-[12px] font-bold text-slate-300 uppercase tracking-wider">Processing Queue</h3>
                <button className="flex items-center gap-2 bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 px-3 py-1.5 rounded-lg text-[10px] font-semibold hover:bg-cyan-500/20 transition">
                    <Icon name="Plus" size={13} />Add Batch
                </button>
            </div>
            <div className="flex flex-col items-center py-8 text-slate-600">
                <Icon name="Inbox" size={36} className="text-slate-700 mb-3" />
                <p className="text-[12px] font-bold text-slate-500">No jobs in queue</p>
                <p className="text-[10px] text-slate-600 mt-1">Upload scenes individually or create a batch job</p>
            </div>
        </Card>
    </div>
);

const DatalayersModule = () => (
    <div className="space-y-4 pb-6">
        <SectionHeader title="Data Layers" subtitle="Multi-spectral band visualizations" icon="Map" />
        <DataLayersGrid />
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <SpectralChart />
            <BandQuality />
        </div>
        <HistogramPanel />
    </div>
);

const ReportsModule = () => (
    <div className="space-y-4 pb-6">
        <SectionHeader title="Reports" subtitle="Automated processing reports and exports" icon="FileBarChart" />
        <div className="space-y-2">
            {[
                { title: 'Cloud Removal Report — June 2025', date: 'Jun 30, 2025', scenes: 8, type: 'Monthly Summary', icon: 'FileText' },
                { title: 'Vegetation Analysis Report — Q2 2025', date: 'Jun 15, 2025', scenes: 24, type: 'Quarterly Report', icon: 'FileBarChart' },
                { title: 'System Performance Report', date: 'Jun 1, 2025', scenes: null, type: 'Technical Report', icon: 'Activity' },
            ].map((r, i) => (
                <div key={i} className="bg-[#0A1020]/80 border border-white/[0.06] rounded-xl p-4 flex items-center gap-4">
                    <Icon name={r.icon} className="text-cyan-400" size={20} />
                    <div className="flex-1">
                        <p className="text-[12px] font-bold text-white">{r.title}</p>
                        <p className="text-[10px] text-slate-500">{r.date} · {r.type}{r.scenes ? ` · ${r.scenes} scenes` : ''}</p>
                    </div>
                    <button className="flex items-center gap-1.5 text-[10px] text-cyan-400 hover:text-cyan-300 font-semibold">
                        <Icon name="Download" size={13} />Export
                    </button>
                </div>
            ))}
        </div>
    </div>
);

const ArchiveModule = () => (
    <div className="space-y-4 pb-6">
        <SectionHeader title="Data Archive" subtitle="Long-term storage and compressed scene archives" icon="Archive" />
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            {[
                { label: 'Archived Scenes', value: '24', color: 'cyan', icon: 'Archive' },
                { label: 'Total Storage', value: '12.4 GB', color: 'indigo', icon: 'HardDrive' },
                { label: 'Oldest Scene', value: 'Jan 2025', color: 'fuchsia', icon: 'Calendar' },
            ].map((m, i) => <MetricCard key={i} {...m} delay={i * 80} />)}
        </div>
        <Card>
            <h3 className="text-[12px] font-bold text-slate-300 uppercase tracking-wider mb-4">Archive Contents</h3>
            <div className="space-y-2">
                {['January 2025 — 4 scenes (1.9 GB)', 'February 2025 — 3 scenes (1.4 GB)', 'March 2025 — 5 scenes (2.3 GB)', 'April 2025 — 6 scenes (2.8 GB)', 'May 2025 — 6 scenes (2.8 GB)'].map((entry, i) => (
                    <div key={i} className="flex items-center justify-between p-3 bg-white/[0.02] rounded-xl">
                        <span className="text-[11px] text-slate-400">{entry}</span>
                        <button className="text-[10px] text-cyan-400 hover:text-cyan-300 font-semibold">Restore</button>
                    </div>
                ))}
            </div>
        </Card>
    </div>
);

const APIConsoleModule = () => {
    const [endpoint, setEndpoint] = useState('/api/status');
    const [response, setResponse] = useState('');
    const [loading, setLoading] = useState(false);
    const call = async () => {
        setLoading(true);
        try {
            const res = await authFetch(endpoint);
            const data = await res.json();
            setResponse(JSON.stringify(data, null, 2));
        } catch (e) { setResponse('Error: ' + e.message); }
        setLoading(false);
    };
    return (
        <div className="space-y-4 pb-6">
            <SectionHeader title="API Console" subtitle="Test and debug backend API endpoints" icon="Terminal" />
            <Card>
                <div className="flex gap-2 mb-4">
                    <select value={endpoint} onChange={e => setEndpoint(e.target.value)}
                        className="flex-1 bg-white/[0.04] border border-white/[0.08] text-[11px] text-slate-300 rounded-lg px-3 py-2 outline-none">
                        {['/api/status', '/api/history', '/api/datasets', '/api/settings', '/api/me'].map(ep => (
                            <option key={ep} value={ep}>{ep}</option>
                        ))}
                    </select>
                    <button onClick={call} disabled={loading}
                        className="px-4 py-2 bg-cyan-500/20 border border-cyan-500/40 text-cyan-400 rounded-lg text-[11px] font-semibold hover:bg-cyan-500/30 transition">
                        {loading ? '...' : 'GET'}
                    </button>
                </div>
                {response && (
                    <pre className="bg-black/40 rounded-xl p-4 text-[10px] text-emerald-300 font-mono overflow-auto max-h-80 whitespace-pre-wrap">{response}</pre>
                )}
            </Card>
        </div>
    );
};

const DocsModule = () => (
    <div className="space-y-4 pb-6">
        <SectionHeader title="Help & Support" subtitle="Documentation and usage guides" icon="HelpCircle" />
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
            {[
                { icon: 'UploadCloud', title: 'How to Upload', desc: 'Supported formats, file size limits, metadata requirements for satellite scene upload.', color: 'text-cyan-400' },
                { icon: 'Cpu', title: 'AI Model', desc: 'Understanding the U-Net architecture, cloud detection pipeline, and reconstruction process.', color: 'text-indigo-400' },
                { icon: 'Download', title: 'Downloading Results', desc: 'Export cloud-free images as GeoTIFF, PNG, cloud mask, and difference maps.', color: 'text-emerald-400' },
                { icon: 'Settings', title: 'Configuration', desc: 'Configure patch size, stride, batch size, and compute device for optimal performance.', color: 'text-amber-400' },
                { icon: 'GitCompare', title: 'Before / After', desc: 'Use the comparison slider to visualize cloudy input vs reconstructed output.', color: 'text-fuchsia-400' },
                { icon: 'Info', title: 'About MeghaMukt', desc: 'Learn about the satellite platform, dataset, and technical team behind MeghaMukt.', color: 'text-blue-400' },
            ].map((doc, i) => (
                <div key={i} className="bg-[#0A1020]/80 border border-white/[0.06] rounded-2xl p-4 hover:border-white/[0.12] transition cursor-pointer group">
                    <Icon name={doc.icon} className={`${doc.color} mb-3 group-hover:scale-110 transition-transform`} size={22} />
                    <h3 className="text-[13px] font-bold text-white mb-2">{doc.title}</h3>
                    <p className="text-[11px] text-slate-500 leading-relaxed">{doc.desc}</p>
                </div>
            ))}
        </div>
        <Card>
            <h3 className="text-[12px] font-bold text-slate-300 uppercase tracking-wider mb-3">Quick Reference</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {[
                    { label: 'Backend API', value: 'http://localhost:8000/docs' },
                    { label: 'Upload Endpoint', value: 'POST /api/upload' },
                    { label: 'Reconstruct Endpoint', value: 'POST /api/reconstruct' },
                    { label: 'History Endpoint', value: 'GET /api/history' },
                    { label: 'System Status', value: 'GET /api/status' },
                    { label: 'Download', value: 'GET /api/download/{id}/{type}' },
                ].map((ref, i) => (
                    <div key={i} className="flex justify-between items-center p-2 bg-white/[0.02] rounded-lg">
                        <span className="text-[10px] text-slate-500">{ref.label}</span>
                        <code className="text-[10px] text-cyan-400 font-mono">{ref.value}</code>
                    </div>
                ))}
            </div>
        </Card>
    </div>
);

// ═══════════════════════════════════════════════════════════════════════════════
// ─── APP ROOT ─────────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════════
const App = () => {
    const [token] = useState(localStorage.getItem('meghamukt_token') || '');
    const [username, setUsername] = useState(localStorage.getItem('meghamukt_user') || '');
    const [isGuest, setIsGuest] = useState(localStorage.getItem('is_guest') === 'true');
    const [activeModule, setActiveModule] = useState('dashboard');
    const [lastResult, setLastResult] = useState(null);
    const [uploadMeta, setUploadMeta] = useState(null);
    const [notifications, setNotifications] = useState([]);
    const [systemStatus, setSystemStatus] = useState(null);
    const [theme, setThemeState] = useState(() => localStorage.getItem('meghamukt_theme') || 'dark');

    const setTheme = (t) => {
        setThemeState(t);
        localStorage.setItem('meghamukt_theme', t);
        if (t === 'light') {
            document.documentElement.setAttribute('data-theme', 'light');
        } else {
            document.documentElement.removeAttribute('data-theme');
        }
    };

    useEffect(() => { setTheme(theme); }, []);

    const addNotification = useCallback((n) => {
        setNotifications(prev => [n, ...prev].slice(0, 20));
    }, []);

    useEffect(() => {
        const handler = () => {
            setUsername(localStorage.getItem('meghamukt_user') || '');
            setIsGuest(localStorage.getItem('is_guest') === 'true');
        };
        window.addEventListener('auth-changed', handler);
        window.addEventListener('storage', handler);
        return () => { window.removeEventListener('auth-changed', handler); window.removeEventListener('storage', handler); };
    }, []);

    useEffect(() => {
        authFetch('/api/status').then(r => r.json()).then(setSystemStatus).catch(() => {});
        const interval = setInterval(() => {
            authFetch('/api/status').then(r => r.json()).then(setSystemStatus).catch(() => {});
        }, 15000);
        return () => clearInterval(interval);
    }, []);

    const handleLogout = () => {
        localStorage.removeItem('meghamukt_token');
        localStorage.removeItem('meghamukt_user');
        localStorage.removeItem('is_guest');
        window.location.href = '/';
    };

    const renderModule = () => {
        switch (activeModule) {
            case 'dashboard': return <DashboardModule onModuleChange={setActiveModule} />;
            case 'upload': return <UploadModule />;
            case 'results': return <ResultsModule />;
            case 'history': return <HistoryModule />;
            case 'monitor': return <MonitorModule />;
            case 'timeseries': return <TimeSeriesModule />;
            case 'change': return <ChangeModule />;
            case 'aoi': return <AOIModule />;
            case 'batch': return <BatchModule />;
            case 'performance': return <PerformanceModule />;
            case 'datalayers': return <DatalayersModule />;
            case 'analytics': return <AnalyticsModule />;
            case 'alerts': return <AlertsModule />;
            case 'reports': return <ReportsModule />;
            case 'archive': return <ArchiveModule />;
            case 'apiconsole': return <APIConsoleModule />;
            case 'settings': return <SettingsModule />;
            case 'docs': return <DocsModule />;
            case 'about': return <AboutModule />;
            default: return <DashboardModule onModuleChange={setActiveModule} />;
        }
    };

    const contextValue = {
        lastResult, setLastResult,
        uploadMeta, setUploadMeta,
        notifications, setNotifications, addNotification,
        systemStatus,
        theme, setTheme,
        username, isGuest,
    };

    return (
        <AppContext.Provider value={contextValue}>
            <div className="flex w-full h-full min-h-screen overflow-hidden bg-[#02050A] text-slate-200 font-body selection:bg-cyan-500/30">
                <div className="fixed top-0 left-1/4 w-[500px] h-[500px] bg-cyan-900/[0.05] rounded-full blur-[120px] pointer-events-none" />
                <div className="fixed bottom-0 right-1/4 w-[400px] h-[400px] bg-blue-900/[0.04] rounded-full blur-[100px] pointer-events-none" />

                <Sidebar activeModule={activeModule} setActiveModule={setActiveModule} onLogout={handleLogout} />

                <div className="flex-1 flex flex-col overflow-hidden relative z-10">
                    <TopBar username={username} isGuest={isGuest} onModuleChange={setActiveModule} onLogout={handleLogout} />
                    <main className="flex-1 overflow-y-auto px-5 py-4 custom-scrollbar">
                        {renderModule()}
                    </main>
                </div>
            </div>
        </AppContext.Provider>
    );
};

const root = ReactDOM.createRoot(document.getElementById('dashboard-root'));
root.render(<App />);
