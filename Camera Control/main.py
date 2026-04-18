import struct
import time

import cv2
import numpy as np
import serial

USB_CAMERA_PORT: int = 0
SMALL_SIZE: tuple[int, int] = (96, 96)  # Change this to whatever you want!
TITLE_HEIGHT: int = 45
PADDING: int = 10

SERIAL_PORT: str = "/dev/cu.usbmodem1401"
BAUD_RATE: int = 921600
SEND_EVERY_N_FRAMES: int = 3

# Open serial connection to Pico
ser: serial.Serial | None
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1, write_timeout=0.1)
    time.sleep(2.0)
    print(f"Connected to Pico on {SERIAL_PORT} at {BAUD_RATE} baud")
except Exception as e:
    print(f"Warning: could not open serial port: {e}")
    raise SystemExit

cap: cv2.VideoCapture = cv2.VideoCapture(USB_CAMERA_PORT)

if not cap.isOpened():
    print("Could not open camera")
    if ser is not None:
        ser.close()
    raise SystemExit

frame_counter: int = 0
prev_frame_time: float = time.time()
smoothed_fps: float = 0.0  # Used to fix the FPS display oscillation!

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break

    current_time: float = time.time()

    # Calculate FPS and smooth it with an Exponential Moving Average
    time_diff = current_time - prev_frame_time
    current_fps: float = 1.0 / time_diff if time_diff > 0 else 0.0
    smoothed_fps = (smoothed_fps * 0.9) + (current_fps * 0.1)
    prev_frame_time = current_time

    original: np.ndarray = frame.copy()
    h, w = original.shape[:2]

    # Create dynamic grayscale image
    gray: np.ndarray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray_small: np.ndarray = cv2.resize(gray, SMALL_SIZE, interpolation=cv2.INTER_AREA)
    gray_float: np.ndarray = gray_small.astype(np.float32) / 255.0

    # Send packet to Pico
    if ser is not None and frame_counter % SEND_EVERY_N_FRAMES == 0:
        try:
            header: bytes = b"IMG0"
            meta: bytes = struct.pack("<HH", SMALL_SIZE[0], SMALL_SIZE[1])
            payload: bytes = gray_small.tobytes()
            ser.write(header + meta + payload)
            # ser.flush() # Removed to prevent serial blocking!
        except Exception as e:
            print(f"Serial write error: {e}")
            ser.close()
            ser = None

    # For display in OpenCV
    gray_small_bgr: np.ndarray = cv2.cvtColor(gray_small, cv2.COLOR_GRAY2BGR)

    gray_upscaled: np.ndarray = cv2.resize(
        gray_small, (w, h), interpolation=cv2.INTER_NEAREST
    )
    gray_upscaled_bgr: np.ndarray = cv2.cvtColor(gray_upscaled, cv2.COLOR_GRAY2BGR)

    # Dynamic Canvas size calculation
    # Ensure bottom area is at least 100px high so text always fits
    bottom_h: int = max(SMALL_SIZE[1], 100)
    canvas_h: int = TITLE_HEIGHT + h + PADDING + bottom_h + PADDING
    canvas_w: int = PADDING + w + PADDING + w + PADDING
    canvas: np.ndarray = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = (25, 25, 25)

    # Titles
    cv2.putText(
        canvas,
        "Original",
        (PADDING, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"Upscaled Grayscale (from {SMALL_SIZE[0]}x{SMALL_SIZE[1]})",
        (PADDING + w + PADDING, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"{SMALL_SIZE[0]}x{SMALL_SIZE[1]} Grayscale",
        (PADDING, TITLE_HEIGHT + h + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Place images
    y_top: int = TITLE_HEIGHT
    x_left: int = PADDING
    canvas[y_top : y_top + h, x_left : x_left + w] = original

    x_right: int = PADDING + w + PADDING
    canvas[y_top : y_top + h, x_right : x_right + w] = gray_upscaled_bgr

    # Dynamically place the small image at bottom-left
    y_bottom: int = TITLE_HEIGHT + h + PADDING
    canvas[y_bottom : y_bottom + SMALL_SIZE[1], x_left : x_left + SMALL_SIZE[0]] = (
        gray_small_bgr
    )

    # Stats
    cv2.putText(
        canvas,
        f"uint8 bytes sent: {gray_small.size}",
        (x_left + SMALL_SIZE[0] + 20, y_bottom + int(SMALL_SIZE[1] / 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"FPS: {int(smoothed_fps)}",
        (canvas_w - 150, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow("Camera Processing View", canvas)

    frame_counter += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
if ser is not None:
    ser.close()
