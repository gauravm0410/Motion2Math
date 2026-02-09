import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

st.set_page_config(layout="wide")

# -----------------------------
# Custom Styling
# -----------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: #e2e8f0;
}

h1, h2, h3 {
    color: #f8fafc;
}

.card {
    background: rgba(255,255,255,0.05);
    padding: 25px;
    border-radius: 18px;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.4);
    margin-bottom: 20px;
}

div.stButton > button {
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: white;
    border-radius: 10px;
    padding: 8px 20px;
    border: none;
    font-weight: 600;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #8b5cf6, #6366f1);
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Session State Setup
# -----------------------------
if "processed" not in st.session_state:
    st.session_state.processed = False

if "playing" not in st.session_state:
    st.session_state.playing = False

if "current_frame" not in st.session_state:
    st.session_state.current_frame = 0

# -----------------------------
# Fake backend function
# -----------------------------
def generate_equations(frame_index):
    return f"y = sin(x + {frame_index})"

# -----------------------------
# Premium Title
# -----------------------------
st.markdown("""
<h1 style='text-align:center; font-size:48px;
background: linear-gradient(90deg,#6366f1,#8b5cf6);
-webkit-background-clip: text;
-webkit-text-fill-color: transparent;'>
Motion2Math
</h1>
<p style='text-align:center; color:#94a3b8; font-size:18px;'>
Transforming visual motion into interpretable mathematical language
</p>
""", unsafe_allow_html=True)

st.markdown("<hr style='border:1px solid #334155;'>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1.5])

# -----------------------------
# Controls Card
# -----------------------------
with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("🎛 Controls")

    uploaded_file = st.file_uploader("Upload Video", type=["mp4"])

    if uploaded_file and st.button("Process Video"):
        with st.spinner("Extracting symbolic contours..."):
            time.sleep(2)
        st.session_state.processed = True
        st.success("Video processed successfully!")

    frame_index = st.slider(
        "Frame",
        0,
        100,
        st.session_state.current_frame,
        disabled=(not st.session_state.processed or st.session_state.playing)
    )

    if st.session_state.processed:
        if st.button("▶ Play Animation"):
            st.session_state.playing = True

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Graph Card
# -----------------------------
with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("📊 Symbolic Motion Visualization")

    graph_placeholder = st.empty()
    equation_placeholder = st.empty()

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    with st.expander("🔍 How does this work?"):
        st.markdown("""
        - Extract structural contours from visual input  
        - Approximate shapes using mathematical functions  
        - Introduce **time (t)** as a variable  
        - Represent motion symbolically instead of pixel-by-pixel  

        This creates a compact, interpretable mathematical model of motion.
        """)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📦 Symbolic Compression Analysis")

    colA, colB = st.columns(2)

    with colA:
        st.markdown("<h3>Raw Pixel Data</h3><h2>2.4 MB</h2>", unsafe_allow_html=True)

    with colB:
        st.markdown("<h3>Equation Representation</h3><h2 style='color:#22d3ee;'>18 KB ↓ 99%</h2>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Core Graph Logic
# -----------------------------
x = np.linspace(-10, 10, 400)

def plot_graph(frame):
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#1e293b")
    ax.set_facecolor("#1e293b")

    y = np.sin(x + frame)

    ax.plot(x, y, linewidth=3, color="#8b5cf6")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-2, 2)
    ax.set_title("Time-Parameterized Graph", fontsize=16, color="white")
    ax.set_xlabel("x", color="white")
    ax.set_ylabel("y", color="white")
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.2, color="white")

    return fig

if st.session_state.processed:

    if st.session_state.playing:
        for i in range(st.session_state.current_frame, 101):
            st.session_state.current_frame = i
            graph_placeholder.pyplot(plot_graph(i))

            equation_placeholder.markdown(f"""
            <div style="
            background: linear-gradient(90deg,#111827,#1f2937);
            color:#22d3ee;
            padding:15px;
            border-radius:12px;
            font-size:18px;
            font-family: monospace;
            text-align:center;">
            {generate_equations(i)}
            </div>
            """, unsafe_allow_html=True)

            time.sleep(0.03)

        st.session_state.playing = False
        st.session_state.current_frame = 0

    else:
        graph_placeholder.pyplot(plot_graph(frame_index))

        equation_placeholder.markdown(f"""
        <div style="
        background: linear-gradient(90deg,#111827,#1f2937);
        color:#22d3ee;
        padding:15px;
        border-radius:12px;
        font-size:18px;
        font-family: monospace;
        text-align:center;">
        {generate_equations(frame_index)}
        </div>
        """, unsafe_allow_html=True)

else:
    st.info("Upload and process a video to begin.")
