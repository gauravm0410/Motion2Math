import cv2
import numpy as np
import matplotlib
import streamlit as st
from skimage import measure
from PIL import Image
import os
import tempfile
import time
import re
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from matplotlib.figure import Figure

# --- BACKEND SETUP ---
if "backend_set" not in st.session_state:
    try:
        matplotlib.use('TkAgg')
    except Exception:
        try:
            matplotlib.use('Qt5Agg')
        except Exception:
            pass
    st.session_state.backend_set = True

import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Motion2Math",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🌌"
)

# -----------------------------------------------------------------------------
# STATE INITIALIZATION
# -----------------------------------------------------------------------------
if "frame_data" not in st.session_state:
    st.session_state.frame_data = []
if "all_eqs" not in st.session_state:
    st.session_state.all_eqs = []
if "processed" not in st.session_state:
    st.session_state.processed = False
if "video_size_mb" not in st.session_state:
    st.session_state.video_size_mb = 0.0
if "current_frame" not in st.session_state:
    st.session_state.current_frame = 0
if "is_playing" not in st.session_state:
    st.session_state.is_playing = False

# -----------------------------------------------------------------------------
# PREMIUM STYLING (FIXING THE "EMPTY BOX" AND SIDEBAR TOGGLE BUG)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
/* Animated Cyberpunk Background */
.stApp {
    background:
        radial-gradient(circle at 20% 30%, rgba(168,85,247,0.12), transparent 40%),
        radial-gradient(circle at 80% 70%, rgba(34,211,238,0.12), transparent 40%),
        linear-gradient(135deg, #0b0f1a, #0f172a, #1e1b4b);
    background-size: 200% 200%;
    animation: gradientShift 20s ease infinite;
    color: white;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* FIX: Ensure toggle button is visible by keeping header but making it transparent */
header {
    background-color: rgba(0,0,0,0) !important;
}
[data-testid="stSidebarNav"] { display: none; }

/* Eliminate default Streamlit padding that causes 'empty boxes' */
[data-testid="column"] > div {
    padding: 0 !important;
}

.block-container {
    max-width: 1400px;
    padding-top: 1.5rem !important;
}

/* Premium Header Styling */
.gradient-text {
    background: linear-gradient(90deg,#c084fc,#22d3ee,#c084fc);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: titleShift 6s linear infinite;
    font-weight: 800;
}

@keyframes titleShift {
    0% { background-position: 0% center; }
    100% { background-position: 200% center; }
}

/* THE GLASS FIX: 
   Instead of manually wrapping elements in divs, we style 
   Streamlit's native blocks that contain our 'glass-trigger' headers.
*/
div[data-testid="stVerticalBlock"] > div:has(.glass-header) {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(20px);
    border-radius: 20px;
    padding: 1.5rem !important;
    border: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 40px rgba(0,0,0,0.3);
}

.glass-header {
    color: #22d3ee;
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 1.2rem;
    border-left: 4px solid #22d3ee;
    padding-left: 12px;
}

/* Equation Box */
.eq-box {
    background: rgba(0, 0, 0, 0.4);
    color: #22d3ee;
    padding: 15px;
    border-radius: 12px;
    font-size: 11px;
    font-family: 'Fira Code', 'Courier New', monospace;
    border: 1px solid rgba(255,255,255,0.1);
    max-height: 250px;
    overflow-y: auto;
    white-space: pre-wrap;
    margin-top: 5px;
}

/* Interactive Elements */
.upload-wrapper {
    background: rgba(255,255,255,0.02);
    border: 1px dashed rgba(255,255,255,0.2);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    margin-bottom: 0.5rem;
}

.stButton button {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
    height: 48px !important;
    font-weight: 600 !important;
    width: 100%;
    color: white !important;
    transition: all 0.3s ease !important;
}

.stButton button:hover {
    border-color: #22d3ee !important;
    box-shadow: 0 0 15px rgba(34,211,238,0.4) !important;
    background: rgba(34, 211, 238, 0.1) !important;
}

/* Progress Bar Cyan */
.stProgress > div > div > div > div {
    background-color: #22d3ee;
}

/* Hide footer and Main Menu */
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# CORE MATH FUNCTIONS
# -----------------------------------------------------------------------------

def process_frame_to_paths(frame, t1, t2, blur_k, scale_percent, simplify_factor):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    w = int(gray.shape[1] * scale_percent / 100)
    h = int(gray.shape[0] * scale_percent / 100)
    low_res = cv2.resize(gray, (max(1, w), max(1, h)), interpolation=cv2.INTER_AREA)
    if blur_k % 2 == 0: blur_k += 1
    blurred = cv2.GaussianBlur(low_res, (blur_k, blur_k), 0)
    edges = cv2.Canny(blurred, t1, t2)
    contours = measure.find_contours(edges, 0.5)
    
    paths = []
    for contour in contours:
        cnt = contour.astype(np.float32).reshape(-1, 1, 2)
        eps = simplify_factor * cv2.arcLength(cnt, True) / 100.0 if simplify_factor > 0 else 0
        approx = cv2.approxPolyDP(cnt, eps, False)
        paths.append(approx.reshape(-1, 2))
    return edges, paths, max(1, w), max(1, h)

def sort_paths_nearest_neighbor(paths):
    if not paths: return []
    remaining = list(paths)
    sorted_paths = [remaining.pop(0)]
    while remaining:
        last_pt = sorted_paths[-1][-1]
        best_idx = 0
        min_dist = float('inf')
        for i, p in enumerate(remaining):
            d = np.linalg.norm(last_pt - p[0])
            if d < min_dist:
                min_dist = d
                best_idx = i
        sorted_paths.append(remaining.pop(best_idx))
    return sorted_paths

def get_fourier_coefficients(paths, width, height, n_coeffs=50):
    cx, cy = width / 2, height / 2
    if not paths: return None, None, None, 0
    optimized_paths = sort_paths_nearest_neighbor(paths)
    combined_pts = []
    for i, path in enumerate(optimized_paths):
        path_pts = [complex(p[1] - cx, -(p[0] - cy)) for p in path]
        combined_pts.extend(path_pts)
        if i < len(optimized_paths) - 1:
            last = path_pts[-1]
            start_next = complex(optimized_paths[i+1][0][1] - cx, -(optimized_paths[i+1][0][0] - cy))
            for step in np.linspace(0, 1, 5):
                combined_pts.append(last * (1-step) + start_next * step)
    N = len(combined_pts)
    coeffs = np.fft.fft(combined_pts)
    freqs = np.fft.fftfreq(N)
    indices = np.argsort(np.abs(coeffs))[::-1][:n_coeffs]
    return coeffs, freqs, indices, N

def generate_fourier_eq(paths, frame_idx, width, height, n_coeffs=50):
    res = get_fourier_coefficients(paths, width, height, n_coeffs)
    if res[0] is None: return f"% Frame {frame_idx}: No paths"
    coeffs, freqs, indices, N = res
    x_p, y_p = [], []
    for idx in indices:
        amp, phase, freq = np.abs(coeffs[idx])/N, np.angle(coeffs[idx]), freqs[idx]*N
        x_p.append(f"{amp:.2f}\\cos({freq:.1f}t + {phase:.2f})")
        y_p.append(f"{amp:.2f}\\sin({freq:.1f}t + {phase:.2f})")
    return f"({ ' + '.join(x_p) }, { ' + '.join(y_p) }) \\{{{frame_idx} <= T < {frame_idx + 1}\\}}"

def generate_linear_eqs(paths, frame_idx, width, height):
    eqs = []
    cx, cy = width / 2, height / 2
    time_block = f"\\{{{frame_idx} <= T < {frame_idx + 1}\\}}"
    for path in paths:
        for i in range(len(path) - 1):
            y1, x1 = -(path[i][0] - cy), path[i][1] - cx
            y2, x2 = -(path[i+1][0] - cy), path[i+1][1] - cx
            if abs(x2 - x1) > 0.001:
                m = (y2 - y1) / (x2 - x1)
                eqs.append(f"y - {y1:.2f} = {m:.3f}(x - {x1:.2f}) \\{{{min(x1,x2):.2f} <= x <= {max(x1,x2):.2f}\\}}{time_block}")
            else:
                eqs.append(f"x = {x1:.2f} \\{{{min(y1,y2):.2f} <= y <= {max(y1,y2):.2f}\\}}{time_block}")
    return eqs

def launch_native_playback(all_frame_data, color, line_width, target_duration_sec):
    try:
        plt.close('all')
        fig, ax = plt.subplots(num="Motion2Math Native Playback", figsize=(10, 9))
        plt.subplots_adjust(bottom=0.25)
        fig.patch.set_facecolor('#020617')
        ax.set_facecolor('#020617')
        ax.set_aspect('equal')
        ax.axis('off')
        first = all_frame_data[0]
        ax.set_xlim(-first['w']/2 - 5, first['w']/2 + 5)
        ax.set_ylim(-first['h']/2 - 5, first['h']/2 + 5)
        line_collection = []
        playback = {'current': 0, 'playing': True}
        def draw_frame(f_idx):
            for l in line_collection: l.remove()
            line_collection.clear()
            curr = all_frame_data[f_idx]
            cx, cy = curr['w']/2, curr['h']/2
            for path in curr['paths']:
                line, = ax.plot(path[:, 1]-cx, -(path[:, 0]-cy), color=color, lw=line_width)
                line_collection.append(line)
            ax.set_title(f"Symbolic Motion: Frame {f_idx}", color='#22d3ee', fontsize=14, fontweight='bold')
            fig.canvas.draw_idle()
        ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03], facecolor='#1e293b')
        slider = Slider(ax_slider, 'Frame ', 0, len(all_frame_data)-1, valinit=0, valstep=1, color='#22d3ee')
        ax_btn = plt.axes([0.45, 0.03, 0.1, 0.04])
        btn = Button(ax_btn, 'Pause', color='#0ea5e9', hovercolor='#06b6d4')
        slider.on_changed(lambda v: draw_frame(int(v)))
        btn.on_clicked(lambda e: (playback.update(playing=not playback['playing']), btn.label.set_text('Play' if not playback['playing'] else 'Pause')))
        def update(i):
            if playback['playing']:
                playback['current'] = (playback['current'] + 1) % len(all_frame_data)
                slider.set_val(playback['current'])
            return line_collection
        draw_frame(0)
        ani = FuncAnimation(fig, update, frames=len(all_frame_data), interval=(target_duration_sec*1000)/len(all_frame_data), blit=False)
        plt.show(block=True)
    except Exception as e: st.error(f"Native Playback Error: {e}")

# -----------------------------------------------------------------------------
# UI RENDERING
# -----------------------------------------------------------------------------

# Main Title Section
st.markdown("""
<div style="text-align:center; margin-bottom:3.5rem;">
    <h1 class="gradient-text" style="font-size:4rem; margin-bottom: 0;">Motion2Math</h1>
    <p style="color:#94a3b8; font-size:1.15rem; margin-top: 0;">Transform video motion into time-parameterized mathematical equations</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for tuning
with st.sidebar:
    st.markdown("### 🛠️ Engine Parameters")
    mode = st.radio("Math Model", ["Fourier (One-Function)", "Linear (Classic)"])
    
    with st.expander("Fine-Tuning", expanded=True):
        f_detail = st.slider("Fourier Detail", 10, 300, 100) if mode == "Fourier (One-Function)" else 0
        fps_limit = st.slider("Sampling FPS", 1, 15, 5)
        max_f = st.slider("Max Frames", 5, 200, 50)
        res_scale = st.slider("Resolution Scale %", 5, 50, 15)
        sim_strength = st.slider("Simplification", 0.1, 5.0, 1.0)
        t1 = st.slider("Canny Sensitivity", 0, 500, 100)
        t2 = st.slider("Canny Strength", 0, 500, 200)

# Layout Grid
col_left, col_right = st.columns([3.5, 6.5], gap="large")

with col_left:
    # 1. INPUT SOURCE
    with st.container():
        st.markdown('<div class="glass-header">📽️ Input Source</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="upload-wrapper">
            <div style="font-size: 2.5rem; color: #c084fc; margin-bottom: 0.5rem;">⬆</div>
            <div style="font-weight: 600; font-size: 0.95rem;">Video Stream Ingest</div>
            <div style="font-size: 0.8rem; color: #94a3b8;">Limit 200MB • MP4, MOV, AVI</div>
        </div>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=["mp4","mov","avi"], label_visibility="collapsed")

    # 2. EXECUTION
    with st.container():
        st.markdown('<div class="glass-header">⚡ Execution</div>', unsafe_allow_html=True)
        if st.button("🚀 Process Trajectory", disabled=(uploaded_file is None)):
            st.session_state.video_size_mb = uploaded_file.size / (1024 * 1024)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as t_vid:
                t_vid.write(uploaded_file.getbuffer())
                v_path = t_vid.name
            
            try:
                cap = cv2.VideoCapture(v_path)
                orig_fps = cap.get(cv2.CAP_PROP_FPS)
                skip = max(1, int(orig_fps / fps_limit))
                f_data, m_eqs = [], []
                p_bar = st.progress(0)
                status = st.empty()
                f_idx, processed = 0, 0
                while cap.isOpened() and processed < max_f:
                    ret, frame = cap.read()
                    if not ret: break
                    if f_idx % skip == 0:
                        edges, paths, w, h = process_frame_to_paths(frame, t1, t2, 3, res_scale, sim_strength)
                        if mode == "Fourier (One-Function)":
                            m_eqs.append(generate_fourier_eq(paths, processed, w, h, f_detail))
                        else:
                            m_eqs.extend(generate_linear_eqs(paths, processed, w, h))
                        f_data.append({'paths': paths, 'w': w, 'h': h, 'edges': edges, 'id': processed})
                        processed += 1
                        p_bar.progress(processed / max_f)
                        status.caption(f"Mapping Frame {processed}/{max_f}...")
                    f_idx += 1
                cap.release()
                st.session_state.frame_data = f_data
                st.session_state.all_eqs = m_eqs
                st.session_state.processed = True
                st.session_state.current_frame = 0
            finally:
                if os.path.exists(v_path): os.remove(v_path)

        if st.session_state.processed:
            st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
            st.session_state.current_frame = st.slider("Scrub Position", 0, len(st.session_state.frame_data)-1, st.session_state.current_frame)
            
            c_p, c_n = st.columns(2)
            with c_p:
                label = "⏸ Pause" if st.session_state.is_playing else "▶ Play"
                if st.button(label):
                    st.session_state.is_playing = not st.session_state.is_playing
            with c_n:
                if st.button("📺 Desktop View"):
                    launch_native_playback(st.session_state.frame_data, "#22d3ee", 1.5, 5)

with col_right:
    # 3. VISUALIZATION
    with st.container():
        st.markdown('<div class="glass-header">📊 Symbolic Trajectory Visualization</div>', unsafe_allow_html=True)
        if st.session_state.processed:
            curr = st.session_state.frame_data[st.session_state.current_frame]
            fig = Figure(figsize=(10, 5), facecolor="none")
            ax = fig.subplots()
            ax.set_facecolor("none")
            cx, cy = curr['w']/2, curr['h']/2
            for p in curr['paths']:
                ax.plot(p[:, 1]-cx, -(p[:, 0]-cy), color="#22d3ee", lw=1.8, alpha=0.9)
            ax.set_aspect('equal')
            ax.axis('off')
            st.pyplot(fig, transparent=True)
            
            st.markdown(f"<p style='color:#94a3b8; font-size:0.85rem; margin-top: 10px;'>Active Functional Set (T={st.session_state.current_frame}):</p>", unsafe_allow_html=True)
            f_eqs = [e for e in st.session_state.all_eqs if f"\\{{{st.session_state.current_frame} <= T < {st.session_state.current_frame+1}\\}}" in e]
            st.markdown(f"<div class='eq-box'>{chr(10).join(f_eqs) if f_eqs else 'N/A'}</div>", unsafe_allow_html=True)
        else:
            st.markdown('<div style="height:350px; display:flex; align-items:center; justify-content:center; color:#64748b; font-style:italic;">Awaiting Video Feed Input...</div>', unsafe_allow_html=True)

    # 4. ANALYSIS
    if st.session_state.processed:
        with st.container():
            st.markdown('<div class="glass-header">📦 Compression Analysis</div>', unsafe_allow_html=True)
            eq_str = "\n".join(st.session_state.all_eqs)
            size_kb = len(eq_str.encode('utf-8')) / 1024
            v_kb = st.session_state.video_size_mb * 1024
            reduction = 100 * (1 - (size_kb / v_kb)) if v_kb > 0 else 0
            
            cA, cB = st.columns(2)
            with cA:
                st.markdown(f"<div style='color:#94a3b8; font-size:0.9rem;'>Raw Stream</div><div style='font-size:2.1rem; font-weight:700; color:#22d3ee;'>{st.session_state.video_size_mb:.2f} MB</div>", unsafe_allow_html=True)
            with cB:
                st.markdown(f"<div style='color:#94a3b8; font-size:0.9rem;'>Symbolic Model</div><div style='font-size:2.1rem; font-weight:700; color:#c084fc;'>{size_kb:.1f} KB</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='color:#10b981; font-weight:600; font-size:0.85rem;'>↓ {reduction:.1f}% Reduction</div>", unsafe_allow_html=True)

    # 5. METHODOLOGY
    with st.container():
        st.markdown('<div class="glass-header">🔍 Engineering Methodology</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;">
            <div style="background: rgba(255,255,255,0.03); padding: 12px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); text-align: center;">
                <div style="color: #22d3ee; font-weight: 700; font-size: 0.85rem;">Edge Map</div>
                <div style="font-size: 0.7rem; color: #94a3b8; margin-top: 4px;">Structural pixel boundaries extracted via Canny logic.</div>
            </div>
            <div style="background: rgba(255,255,255,0.03); padding: 12px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); text-align: center;">
                <div style="color: #c084fc; font-weight: 700; font-size: 0.85rem;">Vectorize</div>
                <div style="font-size: 0.7rem; color: #94a3b8; margin-top: 4px;">Nearest-neighbor path sorting and bridge smoothing.</div>
            </div>
            <div style="background: rgba(255,255,255,0.03); padding: 12px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); text-align: center;">
                <div style="color: #22c55e; font-weight: 700; font-size: 0.85rem;">Functionize</div>
                <div style="font-size: 0.7rem; color: #94a3b8; margin-top: 4px;">FFT or Linear mapping into T-parameterized strings.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# INTERNAL PLAYBACK LOOP
# -----------------------------------------------------------------------------
if st.session_state.is_playing and st.session_state.processed:
    time.sleep(0.06) # Throttle playback for Streamlit loop
    st.session_state.current_frame += 1
    if st.session_state.current_frame >= len(st.session_state.frame_data):
        st.session_state.current_frame = 0
    st.rerun()