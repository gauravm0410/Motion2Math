import cv2
import numpy as np
import matplotlib
import streamlit as st
from skimage import measure
from PIL import Image
import os
import tempfile
import time
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from matplotlib.figure import Figure

# --- BACKEND SETUP ---
# We use interactive backends ONLY for the native playback window.
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

st.set_page_config(page_title="Motion2Math - Symbolic Video Engine", layout="wide")

# -----------------------------
# Custom Styling (Cooler Cyan/Blue Theme)
# -----------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #020617, #0f172a);
    color: #e2e8f0;
}
h1, h2, h3 { color: #f1f5f9; }

/* Refined Card Styling */
.card-header {
    background: rgba(34, 211, 238, 0.1);
    padding: 15px 20px;
    border-radius: 12px 12px 0 0;
    border-left: 4px solid #22d3ee;
    margin-bottom: 0px;
    font-weight: bold;
    color: #22d3ee;
}

.card-content {
    background: rgba(255, 255, 255, 0.03);
    padding: 20px;
    border-radius: 0 0 12px 12px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.05);
    border-top: none;
    margin-bottom: 20px;
}

/* Button Styling: Cyan to Blue */
div.stButton > button {
    background: linear-gradient(90deg, #06b6d4, #0ea5e9);
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
    border: none;
    font-weight: 600;
    transition: all 0.3s ease;
    width: 100%;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #0ea5e9, #06b6d4);
    box-shadow: 0 0 15px rgba(6, 182, 212, 0.4);
    transform: translateY(-1px);
}

.eq-box {
    background: #020617;
    color:#22d3ee;
    padding:15px;
    border-radius:10px;
    font-size:13px;
    font-family: 'Fira Code', 'Courier New', monospace;
    text-align:center;
    border: 1px solid #1e293b;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);
    overflow-x: auto;
}

/* Progress bar color */
.stProgress > div > div > div > div {
    background-color: #22d3ee;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Session State & Logic
# -----------------------------
if "frame_data" not in st.session_state:
    st.session_state.frame_data = []
if "all_eqs" not in st.session_state:
    st.session_state.all_eqs = []
if "processed" not in st.session_state:
    st.session_state.processed = False
if "video_size_mb" not in st.session_state:
    st.session_state.video_size_mb = 0.0

# --- CORE MATH FUNCTIONS ---

def process_frame_to_paths(frame, t1, t2, blur_k, scale_percent, simplify_factor):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    width = int(gray.shape[1] * scale_percent / 100)
    height = int(gray.shape[0] * scale_percent / 100)
    width, height = max(1, width), max(1, height)
    low_res = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
    
    if blur_k % 2 == 0: blur_k += 1
    blurred = cv2.GaussianBlur(low_res, (blur_k, blur_k), 0)
    edges = cv2.Canny(blurred, t1, t2)
    
    contours = measure.find_contours(edges, 0.5)
    simplified_paths = []
    for contour in contours:
        cnt = contour.astype(np.float32).reshape(-1, 1, 2)
        epsilon = simplify_factor * cv2.arcLength(cnt, True) / 100.0 if simplify_factor > 0 else 0
        approx = cv2.approxPolyDP(cnt, epsilon, False)
        simplified_paths.append(approx.reshape(-1, 2))
        
    return edges, simplified_paths, width, height

def generate_fourier_eq(paths, frame_idx, width, height, n_coeffs=20):
    if not paths: return f"% Frame {frame_idx}: No paths"
    cx, cy = width / 2, height / 2
    path = max(paths, key=len)
    complex_pts = [complex(p[1] - cx, -(p[0] - cy)) for p in path]
    N = len(complex_pts)
    coeffs = np.fft.fft(complex_pts)
    freqs = np.fft.fftfreq(N)
    indices = np.argsort(np.abs(coeffs))[::-1][:n_coeffs]
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
        slider.label.set_color('white')
        
        ax_btn = plt.axes([0.45, 0.03, 0.1, 0.04])
        btn = Button(ax_btn, 'Pause', color='#0ea5e9', hovercolor='#06b6d4')
        btn.label.set_color('white')

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
    except Exception as e: st.error(f"Playback Error: {e}")

# -----------------------------
# Premium UI Layout
# -----------------------------
st.markdown("""
<h1 style='text-align:center; font-size:54px;
background: linear-gradient(90deg,#22d3ee,#0ea5e9);
-webkit-background-clip: text;
-webkit-text-fill-color: transparent; margin-bottom: 0;'>
Motion2Math
</h1>
<p style='text-align:center; color:#94a3b8; font-size:18px; margin-top: 0;'>
Transforming visual motion into interpretable mathematical language
</p>
""", unsafe_allow_html=True)

st.markdown("<div style='height: 2px; background: linear-gradient(90deg, transparent, #1e293b, transparent); margin: 20px 0;'></div>", unsafe_allow_html=True)

c1, c2 = st.columns([1, 1.6])

with c1:
    st.markdown("<div class='card-header'>🎛 Controls</div><div class='card-content'>", unsafe_allow_html=True)
    
    video_file = st.file_uploader("Upload Video Source", type=["mp4", "mov"])
    
    with st.expander("⚙️ Vector Settings"):
        mode = st.radio("Math Model", ["Fourier (One-Function)", "Linear (Classic)"])
        fps_limit = st.slider("Sampling FPS", 1, 15, 5)
        max_f = st.slider("Max Frames", 5, 150, 40)
        res = st.slider("Resolution Scale %", 5, 50, 15)
        sim = st.slider("Simplification", 0.5, 10.0, 2.0)
        t1 = st.slider("Sensitivity", 0, 500, 100)
        t2 = st.slider("Strength", 0, 500, 200)

    if video_file and st.button("🚀 Process & Generate Math"):
        # Dynamic calculation of video size
        st.session_state.video_size_mb = video_file.size / (1024 * 1024)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as t_vid:
            t_vid.write(video_file.getbuffer())
            v_path = t_vid.name
        
        try:
            cap = cv2.VideoCapture(v_path)
            orig_fps = cap.get(cv2.CAP_PROP_FPS)
            skip = max(1, int(orig_fps / fps_limit))
            
            f_data, m_eqs = [], []
            p_bar = st.progress(0)
            
            f_idx, processed = 0, 0
            while cap.isOpened() and processed < max_f:
                ret, frame = cap.read()
                if not ret: break
                if f_idx % skip == 0:
                    edges, paths, w, h = process_frame_to_paths(frame, t1, t2, 3, res, sim)
                    if mode == "Fourier (One-Function)":
                        m_eqs.append(generate_fourier_eq(paths, processed, w, h, 30))
                    else:
                        m_eqs.extend(generate_linear_eqs(paths, processed, w, h))
                    
                    f_data.append({'paths': paths, 'w': w, 'h': h, 'edges': edges, 'id': processed})
                    processed += 1
                    p_bar.progress(processed / max_f)
                f_idx += 1
            
            cap.release()
            st.session_state.frame_data = f_data
            st.session_state.all_eqs = m_eqs
            st.session_state.processed = True
            st.success("Symbolic mapping complete!")
        finally:
            if os.path.exists(v_path): os.remove(v_path)

    st.markdown("</div>", unsafe_allow_html=True)
    
    if st.session_state.processed:
        st.markdown("<div class='card-header'>🎞 Playback</div><div class='card-content'>", unsafe_allow_html=True)
        dur = st.slider("Duration (s)", 1, 30, 5)
        if st.button("📺 Launch Native Viewer"):
            launch_native_playback(st.session_state.frame_data, "#22d3ee", 2.0, dur)
        st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown("<div class='card-header'>📊 Symbolic Visualization</div><div class='card-content'>", unsafe_allow_html=True)
    
    if st.session_state.processed:
        idx = st.slider("Scrub Through Math", 0, len(st.session_state.frame_data)-1, 0)
        curr = st.session_state.frame_data[idx]
        
        fig = Figure(figsize=(8, 5), facecolor="#020617")
        ax = fig.subplots()
        ax.set_facecolor("#020617")
        
        cx, cy = curr['w']/2, curr['h']/2
        for p in curr['paths']:
            ax.plot(p[:, 1]-cx, -(p[:, 0]-cy), color="#22d3ee", lw=1.5)
            
        ax.set_aspect('equal')
        ax.axis('off')
        st.pyplot(fig)
        
        st.markdown("<p style='color:#94a3b8; font-size:13px; margin-bottom: 5px;'>Active Function (T-Parameterized):</p>", unsafe_allow_html=True)
        frame_eq = next((e for e in st.session_state.all_eqs if f"\\{{{idx} <= T" in e), "N/A")
        st.markdown(f"<div class='eq-box'>{frame_eq}</div>", unsafe_allow_html=True)
    else:
        st.info("Awaiting video input to generate symbolic vectors.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Stats Card (Now Dynamic)
    st.markdown("<div class='card-header'>📦 Symbolic Compression Analysis</div><div class='card-content'>", unsafe_allow_html=True)
    cA, cB = st.columns(2)
    with cA:
        label = f"{st.session_state.video_size_mb:.2f} MB" if st.session_state.processed else "--"
        st.markdown(f"<p style='color:#94a3b8; margin:0;'>Raw Video Stream</p><h2 style='margin:0;'>{label}</h2>", unsafe_allow_html=True)
    with cB:
        if st.session_state.processed:
            # Calculate actual size of the generated equation strings
            eq_string = "\n".join(st.session_state.all_eqs)
            size_kb = len(eq_string.encode('utf-8')) / 1024
            
            # Calculate real reduction ratio
            video_size_kb = st.session_state.video_size_mb * 1024
            reduction = 100 * (1 - (size_kb / video_size_kb)) if video_size_kb > 0 else 0
            
            st.markdown(f"<p style='color:#94a3b8; margin:0;'>Symbolic Model</p><h2 style='color:#22d3ee; margin:0;'>{size_kb:.1f} KB ↓ {reduction:.1f}%</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:#94a3b8; margin:0;'>Symbolic Model</p><h2 style='color:#22d3ee; margin:0;'>--</h2>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with st.expander("🔍 Scientific Methodology"):
    st.markdown("""
    - **Extraction:** Canny Edge Detection identifies structural boundaries.
    - **Vectorization:** Douglas-Peucker simplification reduces paths into discrete nodes.
    - **Transformation:** Fast Fourier Transform (FFT) or Linear Interpolation maps visual nodes into continuous time-parameterized equations.
    - **Compression:** Massive data reduction while maintaining structural semantics.
    """)