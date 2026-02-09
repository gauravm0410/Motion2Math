import cv2
import numpy as np
import matplotlib
import streamlit as st
from skimage import measure
from PIL import Image
import os
import tempfile
from matplotlib.animation import FuncAnimation

# --- BACKEND SETUP ---
try:
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
except Exception:
    try:
        matplotlib.use('Qt5Agg')
        import matplotlib.pyplot as plt
    except Exception:
        import matplotlib.pyplot as plt

st.set_page_config(page_title="Phase 2: Video-to-Fourier", layout="wide")

# --- CORE FUNCTIONS ---

def process_frame_to_paths(frame, t1, t2, blur_k, scale_percent, simplify_factor):
    """Step 3: Apply image-to-path pipeline."""
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
    """Converts the longest path of a frame into a single Fourier function."""
    if not paths:
        return f"% Frame {frame_idx}: No paths found"
    
    cx, cy = width / 2, height / 2
    path = max(paths, key=len)
    complex_pts = [complex(p[1] - cx, -(p[0] - cy)) for p in path]
    
    N = len(complex_pts)
    coeffs = np.fft.fft(complex_pts)
    freqs = np.fft.fftfreq(N)
    indices = np.argsort(np.abs(coeffs))[::-1][:n_coeffs]
    
    x_parts = []
    y_parts = []
    for idx in indices:
        amp = np.abs(coeffs[idx]) / N
        phase = np.angle(coeffs[idx])
        freq = freqs[idx] * N
        x_parts.append(f"{amp:.2f}\\cos({freq:.1f}t + {phase:.2f})")
        y_parts.append(f"{amp:.2f}\\sin({freq:.1f}t + {phase:.2f})")
    
    time_window = f"\\{{{frame_idx} <= T < {frame_idx + 1}\\}}"
    eq = f"({ ' + '.join(x_parts) }, { ' + '.join(y_parts) }) {time_window}"
    return eq

def generate_linear_eqs(paths, frame_idx, width, height):
    """Standard Linear Equations for a frame."""
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
        fig, ax = plt.subplots(num="Math Video Playback", figsize=(10, 8))
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        ax.set_aspect('equal')
        ax.axis('off')

        first_frame = all_frame_data[0]
        w, h = first_frame['w'], first_frame['h']
        ax.set_xlim(-w/2 - 5, w/2 + 5)
        ax.set_ylim(-h/2 - 5, h/2 + 5)

        line_collection = []

        # CALCULATE DYNAMIC INTERVAL: Total Time / Number of Frames
        # Ensures consistent playback speed regardless of frame count
        interval_ms = (target_duration_sec * 1000) / len(all_frame_data)

        def update(f_idx):
            for l in line_collection: l.remove()
            line_collection.clear()
            
            current_data = all_frame_data[f_idx]
            cx, cy = current_data['w']/2, current_data['h']/2
            for path in current_data['paths']:
                line, = ax.plot(path[:, 1] - cx, -(path[:, 0] - cy), color=color, linewidth=line_width)
                line_collection.append(line)
            
            ax.set_title(f"Math Playback: Frame {f_idx} | Pace: {target_duration_sec}s total", color='white')
            return line_collection

        ani = FuncAnimation(fig, update, frames=len(all_frame_data), interval=interval_ms, blit=False, repeat=True)
        plt.show(block=True)
    except Exception as e:
        st.error(f"Playback error: {e}")

# --- STREAMLIT UI ---

st.title("🎬 Phase 2: Video-to-Math Pipeline")
st.markdown("Convert video frames into **Single-Function Fourier Epicycles** or Linear Equations.")

with st.sidebar:
    st.header("Step 1: Input Video")
    video_file = st.file_uploader("Upload Short Clip", type=["mp4", "mov", "avi"])
    
    st.divider()
    st.header("Step 2: Playback Pace")
    # New slider to control the "same space" / speed
    target_duration = st.slider("Target Playback Duration (sec)", 1, 30, 5)
    
    st.divider()
    st.header("Step 3: Mode Selection")
    mode = st.radio("Equation Mode", ["Linear (Many Equations)", "Fourier (One Function Per Frame)"])
    if mode == "Fourier (One Function Per Frame)":
        f_coeffs = st.slider("Fourier Detail (Coefficients)", 5, 100, 30)
    
    st.divider()
    st.header("Step 4: Sampling")
    fps_limit = st.slider("Target FPS", 1, 15, 5)
    max_frames = st.slider("Max Frames", 5, 200, 50)
    
    st.divider()
    st.header("Step 5: Vision")
    res_scale = st.slider("Res Scale %", 5, 50, 15)
    simplify = st.slider("Simplification", 0.5, 10.0, 2.0)
    t1 = st.slider("Edge Sensitivity", 0, 500, 100)
    t2 = st.slider("Edge Strength", 0, 500, 200)

if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as t_vid:
        t_vid.write(video_file.getbuffer())
        video_path = t_vid.name

    try:
        cap = cv2.VideoCapture(video_path)
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        skip_step = max(1, int(orig_fps / fps_limit))
        
        frame_data_list = []
        all_math_equations = []
        
        progress_bar = st.progress(0)
        status = st.empty()
        
        f_idx = 0
        processed_count = 0
        
        while cap.isOpened() and processed_count < max_frames:
            success, frame = cap.read()
            if not success: break
            
            if f_idx % skip_step == 0:
                edges, paths, w, h = process_frame_to_paths(frame, t1, t2, 3, res_scale, simplify)
                
                if mode == "Fourier (One Function Per Frame)":
                    eq = generate_fourier_eq(paths, processed_count, w, h, f_coeffs)
                    all_math_equations.append(eq)
                else:
                    eqs = generate_linear_eqs(paths, processed_count, w, h)
                    all_math_equations.extend(eqs)
                
                frame_data_list.append({'paths': paths, 'w': w, 'h': h, 'edges': edges, 'id': processed_count})
                processed_count += 1
                status.text(f"Vectorizing Frame {processed_count}...")
                progress_bar.progress(processed_count / max_frames)
            f_idx += 1
        
        cap.release()
        
        st.divider()
        col_view, col_math = st.columns(2)
        
        with col_view:
            st.subheader("Sequence Previews")
            if frame_data_list:
                indices = [0, len(frame_data_list)//2, len(frame_data_list)-1]
                indices = sorted(list(set(indices)))
                preview_cols = st.columns(len(indices))
                for i, idx in enumerate(indices):
                    preview_cols[i].image(Image.fromarray(frame_data_list[idx]['edges']), caption=f"T={idx}")
            
            if st.button("🚀 Play Math Video (Native Plotter)", use_container_width=True):
                launch_native_playback(frame_data_list, "#4ade80", 1.0, target_duration)

        with col_math:
            st.subheader("Equation Output")
            st.metric("Total Equations", len(all_math_equations))
            st.text_area("Final Math Sequence", value="\n".join(all_math_equations), height=400)

    finally:
        try:
            if os.path.exists(video_path): os.remove(video_path)
        except: pass
else:
    st.info("Upload a video to begin.")