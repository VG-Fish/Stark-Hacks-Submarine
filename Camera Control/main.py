import cv2
import numpy as np

USB_CAMERA_PORT: int = 0
SMALL_SIZE: tuple[int, int] = (128, 128)
TITLE_HEIGHT: int = 45
PADDING: int = 10

cap: cv2.VideoCapture = cv2.VideoCapture(USB_CAMERA_PORT)

if not cap.isOpened():
    print("Could not open camera")
    raise SystemExit

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break

    # Original frame
    original: np.ndarray = frame.copy()
    h, w = original.shape[:2]

    # Create 128x128 grayscale normalized image
    gray: np.ndarray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray_128: np.ndarray = cv2.resize(gray, SMALL_SIZE, interpolation=cv2.INTER_AREA)
    gray_float: np.ndarray = gray_128.astype(np.float32) / 255.0

    # For display in OpenCV
    gray_128_display: np.ndarray = (gray_float * 255).astype(np.uint8)
    gray_128_bgr: np.ndarray = cv2.cvtColor(gray_128_display, cv2.COLOR_GRAY2BGR)

    # Upscaled grayscale view for right side
    gray_upscaled: np.ndarray = cv2.resize(
        gray_128_display, (w, h), interpolation=cv2.INTER_NEAREST
    )
    gray_upscaled_bgr: np.ndarray = cv2.cvtColor(gray_upscaled, cv2.COLOR_GRAY2BGR)

    # Canvas size
    bottom_h: int = 128
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
        "Upscaled Grayscale (from 128x128)",
        (PADDING + w + PADDING, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        "128x128 Grayscale [0,1]",
        (PADDING, TITLE_HEIGHT + h + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Place original
    y_top: int = TITLE_HEIGHT
    x_left: int = PADDING
    canvas[y_top : y_top + h, x_left : x_left + w] = original

    # Place upscaled grayscale on right
    x_right: int = PADDING + w + PADDING
    canvas[y_top : y_top + h, x_right : x_right + w] = gray_upscaled_bgr

    # Place 128x128 image at bottom-left
    y_bottom: int = TITLE_HEIGHT + h + PADDING
    canvas[y_bottom : y_bottom + 128, x_left : x_left + 128] = gray_128_bgr

    # Optional text with actual array info
    cv2.putText(
        canvas,
        f"shape={gray_float.shape}, min={gray_float.min():.3f}, max={gray_float.max():.3f}",
        (x_left + 140, y_bottom + 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    cv2.imshow("Camera Processing View", canvas)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
