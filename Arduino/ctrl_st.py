import streamlit as st
from ti_kit_board_communication.main import TiKitBoard
from ctrl_io import get_state

if "board" not in st.session_state:
    st.session_state.board = TiKitBoard(port="COM8")
    st.session_state.board.connect_with_retries()

board = st.session_state.board

@st.fragment(run_every="0.05s")
def send_loop():
    s = get_state()
    left = s["left"]
    right = s["right"]

    msg = f"{left},{right}\n"
    board.send_message(msg.encode("ascii"))

    st.write(f"Left: {left}, Right: {right}")

send_loop()