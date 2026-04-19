from inputs import get_gamepad
import threading
import time

_lock = threading.Lock()
_state = {"left": 0, "right": 0}

def get_state():
    with _lock:
        return dict(_state)

def clamp(x, min_val=-255, max_val=255):
    return max(min(x, max_val), min_val)

def controller_loop():
    forward = 0.0
    turn = 0.0
    while True:
        events = get_gamepad()
        for e in events:
            if e.code == "ABS_Y":
                forward = -(e.state / 32768.0)
            elif e.code == "ABS_X":
                turn = e.state / 32768.0

        # Deadzone
        if abs(forward) < 0.1:
            forward = 0.0
        if abs(turn) < 0.1:
            turn = 0.0

        left  = int(clamp((forward + turn) * 255))
        right = int(clamp((forward - turn) * 255))

        with _lock:
            _state["left"]  = left
            _state["right"] = right

threading.Thread(target=controller_loop, daemon=True).start()