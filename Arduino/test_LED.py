import streamlit as st
from ti_kit_board_communication.main import TiKitBoard

if "board" not in st.session_state:
    st.session_state.board = TiKitBoard(port="COM8")
    st.session_state.board.connect_with_retries()

board = st.session_state.board

if "cmd" not in st.session_state:
    st.session_state.cmd = "x"

if st.button("Test LED", use_container_width=True):
        board.send_message(b"t")

st.markdown("---")
@st.fragment(run_every="0.1s")
def run():
    pass