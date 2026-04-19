import streamlit as st
from ti_kit_board_communication.main import TiKitBoard

if "board" not in st.session_state:
    st.session_state.board = TiKitBoard(port="COM8")
    st.session_state.board.connect_with_retries()

if "cmd" not in st.session_state:
    st.session_state.cmd = "x"

board = st.session_state.board

_, col_fwd, _ = st.columns(3)
with col_fwd:
    if st.button("▲", use_container_width=True):
        st.session_state.cmd = "w"

col_l, col_stop, col_r = st.columns(3)
with col_l:
    if st.button("◀", use_container_width=True):
        st.session_state.cmd = "a"
with col_stop:
    if st.button("■", use_container_width=True, type="primary"):
        st.session_state.cmd = "x"
with col_r:
    if st.button("▶", use_container_width=True):
        st.session_state.cmd = "d"

_, col_rev, _ = st.columns(3)
with col_rev:
    if st.button("▼", use_container_width=True):
        st.session_state.cmd = "s"

@st.fragment(run_every="0.0000000000001s")
def poll():
    board.send_message(st.session_state.cmd.encode("ascii"))

poll()