import streamlit as st # type: ignore
import time

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Motion2Math",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🌌"
)

# -----------------------------------------------------------------------------
# STATE
# -----------------------------------------------------------------------------
if "video_file" not in st.session_state:
    st.session_state.video_file = None
if "is_processed" not in st.session_state:
    st.session_state.is_processed = False
if "current_frame" not in st.session_state:
    st.session_state.current_frame = 0
if "is_playing" not in st.session_state:
    st.session_state.is_playing = False

# -----------------------------------------------------------------------------
# PREMIUM BACKGROUND + STYLING
# -----------------------------------------------------------------------------
st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at 20% 30%, rgba(168,85,247,0.15), transparent 40%),
        radial-gradient(circle at 80% 70%, rgba(34,211,238,0.15), transparent 40%),
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

.block-container {
    max-width: 1440px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.glass {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(18px);
    border-radius: 20px;
    padding: 1.75rem;
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 40px rgba(0,0,0,0.4);
}

.glass:hover {
    transform: translateY(-4px);
    transition: 0.3s ease;
}

.gradient-text {
    background: linear-gradient(90deg,#c084fc,#22d3ee,#c084fc);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: titleShift 6s linear infinite;
}

@keyframes titleShift {
    0% { background-position: 0% center; }
    100% { background-position: 200% center; }
}

.upload-box {
    margin-top: 1rem;
    background: rgba(255,255,255,0.04);
    border: 1px dashed rgba(255,255,255,0.25);
    border-radius: 16px;
    padding: 2.5rem 1rem;
    text-align: center;
}

.upload-icon {
    font-size: 2.5rem;
    margin-bottom: 0.8rem;
    color: #c084fc;
}

.upload-text {
    font-weight: 600;
    margin-bottom: 0.4rem;
}

.upload-subtext {
    font-size: 0.85rem;
    color: #94a3b8;
}

.stButton button {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 12px;
    height: 48px;
    font-weight: 600;
}

.stButton button:hover {
    border-color:#22d3ee;
    box-shadow:0 0 15px rgba(34,211,238,0.4);
}
/* HOW IT WORKS CARDS */
.how-card {
    display: flex;
    align-items: center;
    gap: 1.2rem;
    padding: 1.5rem;
    border-radius: 18px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 1rem;
    transition: all 0.3s ease;
}

.how-card:hover {
    background: rgba(255,255,255,0.07);
    transform: translateY(-3px);
}

.icon-box {
    width: 48px;
    height: 48px;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.4rem;
    color: white;
}

.icon-blue { background: linear-gradient(135deg,#22d3ee,#0ea5e9); }
.icon-pink { background: linear-gradient(135deg,#c084fc,#ec4899); }
.icon-green { background: linear-gradient(135deg,#22c55e,#10b981); }

.how-title { font-weight: 600; font-size: 1.05rem; }
.how-sub { font-size: 0.9rem; color: #94a3b8; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------
st.markdown("""
<div style="text-align:center; margin-bottom:4rem;">
<h1 class="gradient-text" style="font-size:3.8rem;font-weight:700;">
Motion2Math
</h1>
<p style="color:#a1a1aa; font-size:1.1rem; max-width:650px; margin:auto;">
Transform video motion into time-parameterized mathematical equations and visualize movement symbolically
</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# MAIN LAYOUT
# -----------------------------------------------------------------------------
col_left, col_right = st.columns([3,7], gap="large")

# ---------------- LEFT COLUMN ----------------
with col_left:

    # Upload Card
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("### Upload Video")

    st.markdown("""
    <div class="upload-box">
        <div class="upload-icon">⬆</div>
        <div class="upload-text">Drag and drop file here</div>
        <div class="upload-subtext">Limit 200MB • MP4, MOV, AVI</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["mp4","mov","avi"], label_visibility="collapsed")

    if uploaded_file:
        st.session_state.video_file = uploaded_file
        st.success("Video Uploaded Successfully")

    st.markdown("</div>", unsafe_allow_html=True)

    # Controls Card
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("### Controls")

    if st.button("⚡ Process Video", disabled=(st.session_state.video_file is None)):
        with st.spinner("Processing trajectory..."):
            time.sleep(1.2)
        st.session_state.is_processed = True
        st.session_state.current_frame = 0

    st.markdown(
        f"<div style='display:flex; justify-content:space-between;'>"
        f"<span>Frame Position</span>"
        f"<span style='color:#22d3ee'>{st.session_state.current_frame} / 120</span>"
        f"</div>",
        unsafe_allow_html=True
    )

    st.session_state.current_frame = st.slider("",0,120,st.session_state.current_frame,label_visibility="collapsed")

    play_label = "⏸ Pause Animation" if st.session_state.is_playing else "▶ Play Animation"
    if st.button(play_label):
        st.session_state.is_playing = not st.session_state.is_playing

    st.caption("🟢 Ready" if st.session_state.is_processed else "⚪ Awaiting processing")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- RIGHT COLUMN ----------------
with col_right:

    # Motion Trajectory
    st.markdown('<div class="glass" style="height:420px;">', unsafe_allow_html=True)
    st.markdown("### Motion Trajectory")

    if not st.session_state.is_processed:
        st.markdown("""
        <div style="height:300px; display:flex; align-items:center; justify-content:center; color:#64748b;">
            Upload and process a video to view trajectory
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="height:300px; background:rgba(0,0,0,0.35);
        border-radius:14px; position:relative;">
            <div style="
                position:absolute;
                top:50%;
                left:{10+(st.session_state.current_frame/120)*80}%;
                width:18px;
                height:18px;
                background:#22d3ee;
                border-radius:50%;
                box-shadow:0 0 20px #22d3ee;
                transform:translate(-50%,-50%);
            "></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # How It Works
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    with st.expander("How It Works"):
        st.markdown("""
        <div style="margin-bottom:1.5rem; color:#a1a1aa;">
    Motion2Math converts video footage into mathematical equations by analyzing object trajectories and fitting them to parametric models. This enables efficient compression and symbolic manipulation.
    </div>

    <div class="how-card">
        <div class="icon-box icon-blue">🖥</div>
        <div>
            <div class="how-title">Object Detection</div>
            <div class="how-sub">Computer vision algorithms track objects across video frames</div>
        </div>
    </div>

    <div class="how-card">
        <div class="icon-box icon-pink">📈</div>
        <div>
            <div class="how-title">Trajectory Analysis</div>
            <div class="how-sub">Motion data is extracted and fitted to parametric equations</div>
        </div>
    </div>

    <div class="how-card">
        <div class="icon-box icon-green">✂</div>
        <div>
            <div class="how-title">Symbolic Representation</div>
            <div class="how-sub">Mathematical models replace raw pixel data for compression</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Compression Cards
    if st.session_state.is_processed:

        colA, colB = st.columns(2)

        with colA:
            st.markdown("""
            <div class="glass">
                <div style="color:#94a3b8;">Original Video</div>
                <div style="font-size:1.9rem; font-weight:700; color:#22d3ee;">24.8 MB</div>
                <div style="color:#64748b; font-size:0.85rem;">1920×1080, 120 frames</div>
            </div>
            """, unsafe_allow_html=True)

        with colB:
            st.markdown("""
            <div class="glass" style="position:relative;">
                <div style="
                    position:absolute;
                    top:0;
                    right:0;
                    background:linear-gradient(to bottom right, rgba(192,132,252,0.25), rgba(34,211,238,0.25));
                    padding:6px 14px;
                    border-bottom-left-radius:14px;
                    font-size:0.75rem;
                    font-weight:600;">
                    98.7% smaller
                </div>
                <div style="color:#94a3b8;">Mathematical Model</div>
                <div style="font-size:1.9rem; font-weight:700; color:#c084fc;">312 KB</div>
                <div style="color:#64748b; font-size:0.85rem;">Parametric equations + metadata</div>
            </div>
            """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PLAYBACK LOOP
# -----------------------------------------------------------------------------
if st.session_state.is_playing and st.session_state.is_processed:
    time.sleep(0.04)
    st.session_state.current_frame += 1
    if st.session_state.current_frame > 120:
        st.session_state.current_frame = 0
    st.rerun()
