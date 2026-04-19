"""
AUV IMU Dashboard — MPU-6050 Direct Serial (no background thread)
Run with: streamlit run imu_dashboard.py
Requires: pip install streamlit pyserial plotly pillow
"""

import collections
import io
import time

import plotly.graph_objects as go
import serial
import serial.tools.list_ports
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="AUV · IMU",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;400;500&display=swap');
:root {
    --bg:#080a08; --panel:#0d110d; --border:#1a2e1a;
    --amber:#ffb000; --green:#39ff14; --red:#ff4500; --dim:#4a5c4a; --text:#c8d8c8;
}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--text);font-family:'Barlow',sans-serif;}
[data-testid="stSidebar"]{background:var(--panel)!important;border-right:1px solid var(--border);}
[data-testid="stHeader"]{display:none;}
html::after{content:'';position:fixed;inset:0;pointer-events:none;z-index:9999;
  background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(255,176,0,.025) 2px,rgba(255,176,0,.025) 4px);}
.hud{font-family:'Share Tech Mono',monospace;font-size:1rem;color:var(--amber);letter-spacing:.2em;text-transform:uppercase;}
.sub{font-family:'Share Tech Mono',monospace;font-size:.6rem;color:var(--dim);letter-spacing:.15em;text-transform:uppercase;}
.lbl{font-family:'Share Tech Mono',monospace;font-size:.6rem;color:var(--dim);letter-spacing:.18em;text-transform:uppercase;
     border-bottom:1px solid var(--border);padding-bottom:5px;margin:16px 0 10px;}
.card{background:var(--panel);border:1px solid var(--border);border-radius:3px;padding:10px 14px 12px;margin-bottom:8px;}
.clbl{font-family:'Share Tech Mono',monospace;font-size:.58rem;color:var(--dim);letter-spacing:.14em;text-transform:uppercase;}
.cval{font-family:'Share Tech Mono',monospace;font-size:1.5rem;letter-spacing:.04em;margin-top:2px;}
.ping{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:6px;vertical-align:middle;}
.live{background:var(--green);box-shadow:0 0 10px var(--green);animation:blink 1.4s infinite;}
.off{background:var(--dim);}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.25}}
.stxt{font-family:'Share Tech Mono',monospace;font-size:.72rem;color:var(--amber);letter-spacing:.1em;vertical-align:middle;}
.stButton>button{background:transparent!important;border:1px solid var(--amber)!important;color:var(--amber)!important;
  font-family:'Share Tech Mono',monospace!important;font-size:.68rem!important;letter-spacing:.1em!important;border-radius:2px!important;}
.stButton>button:hover{background:var(--amber)!important;color:var(--bg)!important;}
[data-testid="stSelectbox"] label,[data-testid="stSlider"] label{
  font-family:'Share Tech Mono',monospace!important;font-size:.6rem!important;
  color:var(--dim)!important;letter-spacing:.12em!important;text-transform:uppercase!important;}
.seg-box{background:var(--panel);border:1px dashed var(--border);border-radius:3px;
  min-height:260px;display:flex;flex-direction:column;align-items:center;
  justify-content:center;gap:10px;padding:24px;}
.seg-txt{font-family:'Share Tech Mono',monospace;font-size:.68rem;color:var(--dim);letter-spacing:.1em;text-align:center;}
</style>
""", unsafe_allow_html=True)

# ── Persistent state via session_state ────────────────────────────────────────
# Serial object lives in session_state so it persists across reruns
# without a background thread — we read it directly each cycle.

HIST = 200

if "ax" not in st.session_state:
    st.session_state.ax      = collections.deque([0.0] * HIST, maxlen=HIST)
    st.session_state.ay      = collections.deque([0.0] * HIST, maxlen=HIST)
    st.session_state.az      = collections.deque([0.0] * HIST, maxlen=HIST)
    st.session_state.latest  = {"ax": 0.0, "ay": 0.0, "az": 0.0}
    st.session_state.ser     = None   # serial.Serial object
    st.session_state.running = False
    st.session_state.error   = None
    st.session_state.seg_img = None


def open_port(port, baud):
    """Open serial port, close existing one first."""
    close_port()
    try:
        s = serial.Serial(port, baud, timeout=0)  # timeout=0 → non-blocking
        st.session_state.ser     = s
        st.session_state.running = True
        st.session_state.error   = None
    except serial.SerialException as e:
        st.session_state.error   = str(e)
        st.session_state.running = False


def close_port():
    s = st.session_state.get("ser")
    if s:
        try:
            if s.is_open:
                s.close()
        except Exception:
            pass
    st.session_state.ser     = None
    st.session_state.running = False


def read_port(n_lines=20):
    """
    Non-blocking read: pull up to n_lines from the buffer each rerun.
    Returns number of samples parsed.
    """
    s = st.session_state.get("ser")
    if not s or not s.is_open:
        return 0
    parsed = 0
    try:
        waiting = s.in_waiting
        if waiting == 0:
            return 0
        # read all available bytes at once, split into lines
        raw_bytes = s.read(waiting)
        lines = raw_bytes.decode("utf-8", errors="ignore").splitlines()
        for line in lines[-n_lines:]:  # only use most recent n_lines
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) >= 3:
                try:
                    ax = float(parts[0])
                    ay = float(parts[1])
                    az = float(parts[2])
                    st.session_state.ax.append(ax)
                    st.session_state.ay.append(ay)
                    st.session_state.az.append(az)
                    st.session_state.latest = {"ax": ax, "ay": ay, "az": az}
                    parsed += 1
                except ValueError:
                    pass
    except Exception as e:
        st.session_state.error   = str(e)
        st.session_state.running = False
        close_port()
    return parsed


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="hud">◈ AUV / IMU</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub">MPU-6050 · TELEMETRY</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="lbl">Serial</div>', unsafe_allow_html=True)
    ports    = [p.device for p in serial.tools.list_ports.comports()]
    port_sel = st.selectbox("Port", options=ports if ports else ["No ports found"])
    baud_sel = st.selectbox("Baud", options=[9600, 19200, 38400, 57600, 115200], index=4)

    ca, cb = st.columns(2)
    with ca:
        if st.button("▶ OPEN"):
            if port_sel != "No ports found":
                open_port(port_sel, baud_sel)
    with cb:
        if st.button("■ CLOSE"):
            close_port()

    running = st.session_state.running
    dot     = "live" if running else "off"
    stxt    = "LIVE" if running else "OFFLINE"
    st.markdown(
        f'<span class="ping {dot}"></span><span class="stxt">{stxt}</span>',
        unsafe_allow_html=True,
    )
    if st.session_state.error:
        st.error(st.session_state.error)

    st.markdown('<div class="lbl">Display</div>', unsafe_allow_html=True)
    refresh_ms = st.slider("Refresh ms", 100, 2000, 250, step=50)
    show_grid  = st.checkbox("Grid", value=True)
    history_n  = st.slider("Window", 50, HIST, 150, step=10)
    fill_area  = st.checkbox("Fill", value=True)

    with st.expander("DIAGNOSTICS"):
        n = read_port()   # extra drain inside diagnostics
        s = st.session_state.get("ser")
        st.code(
            f"parsed     : {n}\n"
            f"port open  : {s.is_open if s else False}\n"
            f"ax={st.session_state.latest['ax']:+.4f} "
            f"ay={st.session_state.latest['ay']:+.4f} "
            f"az={st.session_state.latest['az']:+.4f}",
            language=None,
        )
        if st.button("INJECT TEST"):
            import math
            for i in range(40):
                t = i / 8
                st.session_state.ax.append(0.3 * math.sin(t))
                st.session_state.ay.append(0.2 * math.cos(t * 1.3))
                st.session_state.az.append(1.0 + 0.1 * math.sin(t * 2))
            st.session_state.latest = {"ax": 0.0, "ay": 0.0, "az": 1.0}

# ── Read serial this rerun ────────────────────────────────────────────────────
read_port()

# ── Layout ────────────────────────────────────────────────────────────────────
st.markdown('<p class="hud" style="font-size:.85rem;margin-bottom:14px;">▸ ACCEL TELEMETRY // MPU-6050</p>', unsafe_allow_html=True)

left, right = st.columns([3, 2], gap="large")

with left:
    ax_d = list(st.session_state.ax)[-history_n:]
    ay_d = list(st.session_state.ay)[-history_n:]
    az_d = list(st.session_state.az)[-history_n:]
    xs   = list(range(len(ax_d)))
    fill = "tozeroy" if fill_area else "none"

    fig = go.Figure()
    for y, name, col, fc in [
        (ax_d, "AX", "#ffb000", "rgba(255,176,0,0.07)"),
        (ay_d, "AY", "#39ff14", "rgba(57,255,20,0.06)"),
        (az_d, "AZ", "#ff4500", "rgba(255,69,0,0.06)"),
    ]:
        fig.add_trace(go.Scatter(
            x=xs, y=y, name=name, mode="lines",
            line=dict(color=col, width=1.5),
            fill=fill, fillcolor=fc,
        ))

    gc = "#111d11" if show_grid else "rgba(0,0,0,0)"
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#080a08",
        font=dict(family="Share Tech Mono, monospace", color="#4a5c4a", size=10),
        margin=dict(l=44, r=8, t=8, b=36), height=340,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                    font=dict(size=10, color="#4a5c4a"), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=show_grid, gridcolor=gc, zeroline=False,
                   tickfont=dict(size=9), title=dict(text="SAMPLE", font=dict(size=9)),
                   linecolor="#1a2e1a"),
        yaxis=dict(showgrid=show_grid, gridcolor=gc, zeroline=True,
                   zerolinecolor="#1a2e1a", tickfont=dict(size=9),
                   title=dict(text="g", font=dict(size=9)), linecolor="#1a2e1a"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#0d110d", font_color="#ffb000",
                        font_family="Share Tech Mono", bordercolor="#1a2e1a"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with right:
    st.markdown('<div class="lbl">Live Readings</div>', unsafe_allow_html=True)
    lat = st.session_state.latest
    for label, key, col in [("AX", "ax", "#ffb000"), ("AY", "ay", "#39ff14"), ("AZ", "az", "#ff4500")]:
        val   = lat[key]
        bar_w = min(abs(val) * 80, 100)
        st.markdown(
            f'<div class="card">'
            f'<div class="clbl">ACCEL {label}</div>'
            f'<div class="cval" style="color:{col};">{val:+.4f}'
            f'<span style="font-size:.6rem;color:var(--dim);margin-left:4px;">g</span></div>'
            f'<div style="margin-top:6px;height:2px;background:#1a2e1a;border-radius:2px;">'
            f'<div style="width:{bar_w}%;height:100%;background:{col};opacity:.6;border-radius:2px;"></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="lbl" style="margin-top:20px;">CNN Segmentation</div>', unsafe_allow_html=True)
    up = st.file_uploader("Drop PNG", type=["png","jpg","jpeg"], key="seg_up", label_visibility="collapsed")
    if up:
        st.session_state.seg_img = Image.open(io.BytesIO(up.read()))

    if st.session_state.seg_img:
        st.image(st.session_state.seg_img, use_container_width=True, caption="Segmentation output")
    else:
        st.markdown(
            '<div class="seg-box">'
            '<svg width="50" height="50" viewBox="0 0 50 50" fill="none" style="opacity:.2">'
            '<rect x="1" y="1" width="48" height="48" rx="3" stroke="#ffb000" stroke-width="1.5" stroke-dasharray="4 3"/>'
            '<circle cx="25" cy="19" r="6" stroke="#ffb000" stroke-width="1.2"/>'
            '<path d="M7 38 Q15 27 21 31 Q27 35 33 25 Q39 17 43 29" stroke="#ffb000" stroke-width="1.2" fill="none"/>'
            '</svg>'
            '<div class="seg-txt">NO OUTPUT LOADED<br>'
            '<span style="font-size:.58rem;opacity:.4;">UPLOAD WHEN MODEL IS READY</span></div>'
            '</div>',
            unsafe_allow_html=True,
        )

# ── Refresh ───────────────────────────────────────────────────────────────────
time.sleep(refresh_ms / 1000.0)
st.rerun()