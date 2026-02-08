import cv2
import numpy as np
import matplotlib
import streamlit as st
from skimage import measure
from PIL import Image
import os
import tempfile

# Backend setup for native window
try:
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
except Exception:
    try:
        matplotlib.use('Qt5Agg')
        import matplotlib.pyplot as plt
    except Exception:
        import matplotlib.pyplot as plt

st.set_page_config(page_title="Ultra Math Vectorizer", layout="wide")

def get_simplified_data(image_path, t1, t2, blur_k, scale_percent, simplify_factor):
    # Open and close immediately to avoid file locking issues
    img = cv2.imread(image_path)
    if img is None: return None, [], 0, 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    width = int(gray.shape[1] * scale_percent / 100)
    height = int(gray.shape[0] * scale_percent / 100)
    
    # Ensure dimensions are at least 1px to avoid resize errors
    width = max(1, width)
    height = max(1, height)
    
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

def generate_crazy_equations(paths, width, height, mode, animate):
    eqs = []
    cx, cy = width / 2, height / 2
    total_segments = sum(len(p) - 1 for p in paths)
    current_seg = 0

    if mode == "Linear (Classic)":
        for path in paths:
            for i in range(len(path) - 1):
                y1, x1 = -(path[i][0] - cy), path[i][1] - cx
                y2, x2 = -(path[i+1][0] - cy), path[i+1][1] - cx
                
                anim_cond = f"\\{{{current_seg/total_segments:.3f} <= T\\}}" if animate else ""
                
                if abs(x2 - x1) > 0.001:
                    m = (y2 - y1) / (x2 - x1)
                    eq = f"y - {y1:.2f} = {m:.3f}(x - {x1:.2f}) \\{{{min(x1,x2):.2f} <= x <= {max(x1,x2):.2f}\\}}{anim_cond}"
                else:
                    eq = f"x = {x1:.2f} \\{{{min(y1,y2):.2f} <= y <= {max(y1,y2):.2f}\\}}{anim_cond}"
                eqs.append(eq)
                current_seg += 1

    elif mode == "3D Projection (Sphere)":
        eqs.append("R = 10")
        for path in paths:
            for i in range(len(path) - 1):
                y, x = -(path[i][0] - cy), path[i][1] - cx
                lon = (x / width) * 2 * np.pi
                lat = (y / height) * np.pi
                eq = f"(R \\cos({lon:.3f}) \\sin({lat:.3f}), R \\sin({lon:.3f}) \\sin({lat:.3f}), R \\cos({lat:.3f}))"
                eqs.append(eq)

    elif mode == "Fourier (Epicycles)":
        if paths:
            longest = max(paths, key=len)
            complex_pts = [complex(p[1]-cx, -(p[0]-cy)) for p in longest]
            if len(complex_pts) > 1:
                coeffs = np.fft.fft(complex_pts)
                freqs = np.fft.fftfreq(len(complex_pts))
                N = min(20, len(complex_pts)) 
                sorted_indices = np.argsort(np.abs(coeffs))[::-1][:N]
                f_parts = []
                for idx in sorted_indices:
                    amp = np.abs(coeffs[idx]) / len(complex_pts)
                    phase = np.angle(coeffs[idx])
                    freq = freqs[idx] * len(complex_pts)
                    f_parts.append(f"{amp:.2f} \\exp(i( {freq:.1f}t + {phase:.2f} ))")
                eqs.append("f(t) = " + " + ".join(f_parts))
                eqs.append("Plot f(t) from 0 to 2\\pi")
    return eqs

def launch_matplotlib(paths, color, width, title):
    try:
        plt.close('all')
        fig, ax = plt.subplots(num=f"Plot: {title}", figsize=(10, 8))
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        for path in paths:
            ax.plot(path[:, 1], -path[:, 0], color=color, linewidth=width)
        ax.set_aspect('equal')
        ax.axis('off')
        plt.show(block=True)
    except Exception as e:
        st.error(f"Window Error: {e}")

# --- UI ---
st.title("🚀 Ultra Calculator Art Vectorizer")

with st.sidebar:
    st.header("1. Crazy Modes")
    mode = st.selectbox("Render Mode", ["Linear (Classic)", "3D Projection (Sphere)", "Fourier (Epicycles)"])
    animate = st.checkbox("Add Animation (T slider)", value=False)
    
    st.divider()
    st.header("2. Resolution & Math")
    res_scale = st.slider("Resolution Scale %", 5, 100, 20)
    simplify = st.slider("Simplification Strength", 0.0, 10.0, 2.0)
    
    st.divider()
    st.header("3. Edge Detection")
    t1 = st.slider("Sensitivity", 0, 500, 100)
    t2 = st.slider("Strength", 0, 500, 200)
    blur = st.slider("Blur", 1, 15, 3, step=2)

if uploaded_file := st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"]):
    # Use delete=False and close the file manually to ensure Windows releases the handle
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as t_file:
        t_file.write(uploaded_file.getbuffer())
        img_path = t_file.name

    try:
        edges, paths, w, h = get_simplified_data(img_path, t1, t2, blur, res_scale, simplify)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Downsampled Edge Preview")
            if edges is not None:
                # Explicitly convert NumPy array to PIL Image to fix the AttributeError
                preview_pil = Image.fromarray(edges)
                st.image(preview_pil, use_container_width=True, caption=f"Resolution: {w}x{h}")
            else:
                st.error("Failed to process edges.")
        
        with col2:
            st.subheader(f"Equation Output ({mode})")
            equations = generate_crazy_equations(paths, w, h, mode, animate)
            st.metric("Total Equations", len(equations))
            
            full_eq_text = "\n".join(equations)
            st.text_area("Copy-Paste to Graphing Calculator", value=full_eq_text, height=300)
            
            if animate:
                st.info("💡 Pro Tip: In the calculator, add a slider for 'T' from 0 to 1 to see the animation!")

        if st.button("🚀 View Native Plot", use_container_width=True):
            launch_matplotlib(paths, "#4ade80", 1.0, uploaded_file.name)

    finally:
        # We delete the file in a finally block to ensure it happens after all processing is done
        # Wrap in another try/except because Windows file handles can be stubborn
        try:
            if os.path.exists(img_path):
                os.remove(img_path)
        except Exception:
            pass # If we can't delete it now, the OS will clean the temp folder later
else:
    st.info("Upload an image to start the madness.")