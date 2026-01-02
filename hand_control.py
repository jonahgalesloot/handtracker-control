import cv2
import zmq
import numpy as np
import subprocess
import sys
import atexit
import json
import pyautogui
import time
import mediapipe as mp
import math
import keyboard  # For global keybinds
import easyocr
from pprint import pprint

# Initialize EasyOCR reader (using English, no GPU for lightweight operation)
DEBUG_DISPLAY = False  # Set to True to show a debug window for the drawing image.
DEBUG_ZMQ = False  # When True, print all incoming ZeroMQ message contents to console for debugging
reader = easyocr.Reader(['en'], gpu=False)

# Start the hand tracker as a subprocess.
handtracker_proc = subprocess.Popen([sys.executable, "handtracker_module.py"])
atexit.register(lambda: handtracker_proc.terminate())

# ZeroMQ setup for receiving data from the hand tracker.
context = zmq.Context()
socket_sub = context.socket(zmq.SUB)
socket_sub.connect("tcp://localhost:5556")
socket_sub.setsockopt_string(zmq.SUBSCRIBE, "")

# Get screen dimensions.
screen_width, screen_height = pyautogui.size()

# Frame and scroll settings.
frame_count = 0
scroll_frames = 0

# For click debounce.
last_click_time = 0
click_debounce_seconds = 1.0

# For displaying messages.
click_message_time = 0
CLICK_MESSAGE_DURATION = 1.0  # seconds

# Get Mediapipe Hand Connections.
mp_hands = mp.solutions.hands
HAND_CONNECTIONS = mp_hands.HAND_CONNECTIONS

# Configurable variables.
INDEX_CIRCLE_RADIUS = 30     # Red circle around index tip (used for clicking)
THUMB_CIRCLE_RADIUS = 30     # Green circle around thumb tip (used for visual)
GESTURE_THRESHOLD = INDEX_CIRCLE_RADIUS       # Distance threshold for index-middle finger activation
MARGIN_PCT = 0.20            # 20% margin from each edge of the camera FOV
SCROLL_THRESHOLD_ANGLE = 20  # Degrees threshold for scrolling activation
RING_THUMB_THRESHOLD = 30    # Threshold (in pixels) for ring finger & thumb touching (for drawing)

# Global flags for overlay visibility and program exit.
overlayVisible = False  # Off by default.
exitRequested = False

# Global drawing variables.
drawing_points = []      # List to store (x,y) points for index finger drawing.
non_touch_frames = 0     # Counter for consecutive frames when drawing gesture is not detected.

# Window size variables.
WINDOW_NAME = "Hand Tracking"
FULL_WINDOW_WIDTH = 640
FULL_WINDOW_HEIGHT = 480
MINIMIZED_WINDOW_WIDTH = 64
MINIMIZED_WINDOW_HEIGHT = 64

# Windows API constants for borderless window.
GWL_STYLE = -16
WS_CAPTION = 0x00C00000
WS_THICKFRAME = 0x00040000

pyautogui.FAILSAFE = False

def calculate_angle(x1, y1, x2, y2):
    """Calculate angle of line (x1, y1) -> (x2, y2) relative to vertical."""
    delta_x = x2 - x1
    delta_y = y1 - y2  # Flip y because screen coordinates start at top-left.
    angle = math.degrees(math.atan2(delta_y, delta_x))
    return angle

def process_drawing_easyocr(points):
    if not points:
        return ""
    
    img_size = 512
    # Create a gray background (3-channel image, with medium gray = 128)
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 128

    # Compute bounding box of points.
    pts = np.array(points)
    min_x, max_x = pts[:, 0].min(), pts[:, 0].max()
    min_y, max_y = pts[:, 1].min(), pts[:, 1].max()
    
    # Avoid very small drawings.
    if max_x - min_x < 5 or max_y - min_y < 5:
        return ""
    
    margin = 20
    scale_x = (img_size - 2 * margin) / (max_x - min_x)
    scale_y = (img_size - 2 * margin) / (max_y - min_y)
    scale = min(scale_x, scale_y)
    
    normalized_points = []
    for (x, y) in points:
        new_x = int((x - min_x) * scale + margin)
        new_y = int((y - min_y) * scale + margin)
        normalized_points.append((new_x, new_y))
        # Draw thicker dots (radius=6) in black to make strokes bolder for OCR.
        cv2.circle(img, (new_x, new_y), 6, (0, 0, 0), -1)
    
    # Draw thicker lines (thickness=8) connecting consecutive points.
    for i in range(1, len(normalized_points)):
        cv2.line(img, normalized_points[i-1], normalized_points[i], (0, 0, 0), 8)
    
    # Preprocess: convert to grayscale, blur, and threshold.
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
    # Use non-inverted binary threshold so characters are dark on a light background
    _, thresh_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    processed_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel)
    # Strengthen strokes to improve OCR consistency
    dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    processed_img = cv2.dilate(processed_img, dilate_k, iterations=1)
    
    # Optionally display the debug window.
    if DEBUG_DISPLAY:
        cv2.imshow("Drawing Debug", processed_img)
        cv2.waitKey(1)
    
    # Run EasyOCR on the processed image.
    t0 = time.time()
    results = reader.readtext(processed_img)
    ocr_time = time.time() - t0
    recognized = ""
    # Accept moderately low-confidence single-character results for handwritten letters
    threshold = 0.0
    for bbox, text, score in results:
        if score >= threshold:
            recognized = text
            print(f"EasyOCR recognized: {text} (score: {score:.2f}) in {ocr_time:.3f} s")
            break
    if recognized:
        return recognized
    else:
        print(f"EasyOCR: no result >= {threshold:.2f}; results:", [(t, s) for _, t, s in results])
        return ""


def set_borderless(window_name):
    """Removes the border from the OpenCV window using Windows API calls."""
    import ctypes
    hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
    if hwnd == 0:
        print(f"Window '{window_name}' not found.")
        return
    style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_STYLE)
    style &= ~(WS_CAPTION | WS_THICKFRAME)
    ctypes.windll.user32.SetWindowLongW(hwnd, GWL_STYLE, style)
    ctypes.windll.user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0, 0x0002 | 0x0001 | 0x0020)

def toggle_display():
    global overlayVisible
    overlayVisible = not overlayVisible
    if overlayVisible:
        cv2.resizeWindow(WINDOW_NAME, FULL_WINDOW_WIDTH, FULL_WINDOW_HEIGHT)
        cv2.moveWindow(WINDOW_NAME, 0, 0)
    else:
        cv2.resizeWindow(WINDOW_NAME, MINIMIZED_WINDOW_WIDTH, MINIMIZED_WINDOW_HEIGHT)
        cv2.moveWindow(WINDOW_NAME, -1000, -1000)
    print(f"Overlay Display {'On' if overlayVisible else 'Off'}")

def exit_program():
    global exitRequested
    exitRequested = True
    print("Exit requested via keybind.")

# Set up hotkeys.
keyboard.add_hotkey('ctrl+alt+d', toggle_display)
keyboard.add_hotkey('ctrl+alt+q', exit_program)

def main():
    global frame_count, last_click_time, click_message_time, scroll_frames
    global exitRequested, drawing_points, non_touch_frames, overlayVisible

    # Create the window initially.
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    set_borderless(WINDOW_NAME)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
    cv2.resizeWindow(WINDOW_NAME, FULL_WINDOW_WIDTH, FULL_WINDOW_HEIGHT)

    while True:
        if exitRequested:
            break

        try:
            parts = socket_sub.recv_multipart()
            # ZMQ debug: optionally print raw parts and decoded JSON/image sizes
            if DEBUG_ZMQ:
                try:
                    print('--- ZMQ MESSAGE START ---')
                    print('parts lengths:', [len(p) for p in parts])
                    if len(parts) >= 1:
                        try:
                            json_data_dbg = parts[0].decode('utf-8')
                            print('JSON payload string:')
                            print(json_data_dbg)
                            try:
                                jd = json.loads(json_data_dbg)
                                print('Decoded JSON:')
                                pprint(jd)
                            except Exception as e:
                                print('JSON decode error:', e)
                        except Exception as e:
                            print('Error decoding JSON part:', e)
                    if len(parts) >= 2:
                        try:
                            print('Image bytes length:', len(parts[1]))
                        except Exception as e:
                            print('Error accessing image part:', e)
                    print('--- ZMQ MESSAGE END ---')
                except Exception as e:
                    print('ZMQ debug print error:', e)

            if len(parts) != 2:
                continue

            json_data = parts[0].decode('utf-8')
            try:
                hand_data = json.loads(json_data)
            except json.JSONDecodeError:
                continue

            jpg_bytes = parts[1]
            np_img = np.frombuffer(jpg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            frame_height, frame_width = frame.shape[:2]
            frame_count += 1

            # Draw effective FOV white rectangle.
            left_bound   = int(MARGIN_PCT * frame_width)
            right_bound  = int((1 - MARGIN_PCT) * frame_width)
            top_bound    = int(MARGIN_PCT * frame_height)
            bottom_bound = int((1 - MARGIN_PCT) * frame_height)
            cv2.rectangle(frame, (left_bound, top_bound), (right_bound, bottom_bound), (255, 255, 255), 2)

            # Always process hand data.
            if "landmarks" in hand_data and hand_data["landmarks"]:
                first_hand = hand_data["landmarks"][0]
                landmark_pixels = []
                for idx, lm in enumerate(first_hand):
                    x_pixel = int(lm["x"] * frame_width)
                    y_pixel = int(lm["y"] * frame_height)
                    landmark_pixels.append((x_pixel, y_pixel))
                    if idx == 8:
                        cv2.circle(frame, (x_pixel, y_pixel), 4, (255, 0, 0), -1)
                    else:
                        cv2.circle(frame, (x_pixel, y_pixel), 4, (0, 255, 0), -1)
                for connection in HAND_CONNECTIONS:
                    start_idx, end_idx = connection
                    if start_idx < len(landmark_pixels) and end_idx < len(landmark_pixels):
                        cv2.line(frame, landmark_pixels[start_idx], landmark_pixels[end_idx], (0, 255, 255), 2)

                if len(first_hand) > 16:
                    index_tip_norm  = first_hand[8]
                    index_base_norm = first_hand[5]  # MCP joint of index.
                    middle_tip_norm = first_hand[12]
                    thumb_tip_norm  = first_hand[4]
                    ring_tip_norm   = first_hand[16]  # Ring finger tip.

                    def effective_coord(norm_coord):
                        clipped = np.clip(norm_coord, MARGIN_PCT, 1 - MARGIN_PCT)
                        return (clipped - MARGIN_PCT) / (1 - 2 * MARGIN_PCT)
                    effective_x = effective_coord(index_tip_norm["x"])
                    effective_y = effective_coord(index_tip_norm["y"])
                    desired_x = int(effective_x * screen_width)
                    desired_y = int(effective_y * screen_height)

                    index_x  = int(index_tip_norm["x"] * frame_width)
                    index_y  = int(index_tip_norm["y"] * frame_height)
                    middle_x = int(middle_tip_norm["x"] * frame_width)
                    middle_y = int(middle_tip_norm["y"] * frame_height)
                    thumb_x  = int(thumb_tip_norm["x"] * frame_width)
                    thumb_y  = int(thumb_tip_norm["y"] * frame_height)
                    ring_x   = int(ring_tip_norm["x"] * frame_width)
                    ring_y   = int(ring_tip_norm["y"] * frame_height)

                    cv2.circle(frame, (index_x, index_y), INDEX_CIRCLE_RADIUS, (0, 0, 255), 2)

                    # Cursor movement: when index touches middle.
                    distance_index_middle = np.sqrt((index_x - middle_x)**2 + (index_y - middle_y)**2)
                    if distance_index_middle < GESTURE_THRESHOLD:
                        if frame_count % 5 == 0:
                            current_mouse = pyautogui.position()
                            delta_x = desired_x - current_mouse[0]
                            delta_y = desired_y - current_mouse[1]
                            pyautogui.moveRel(delta_x, delta_y)
                        cv2.putText(frame, "Tracking Active", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "Gesture Inactive", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # Thumb circle and scroll detection.
                    cv2.circle(frame, (thumb_x, thumb_y), THUMB_CIRCLE_RADIUS, (0, 255, 0), 2)
                    distance_middle_thumb = np.sqrt((middle_x - thumb_x)**2 + (middle_y - thumb_y)**2)
                    if distance_middle_thumb < THUMB_CIRCLE_RADIUS:
                        angle = calculate_angle(index_base_norm["x"] * frame_width,
                                                index_base_norm["y"] * frame_height,
                                                index_x, index_y)
                        if frame_count % 5 == 0:
                            if 90 - SCROLL_THRESHOLD_ANGLE <= angle <= 90 + SCROLL_THRESHOLD_ANGLE:
                                pyautogui.scroll(math.ceil(5 * math.log2(scroll_frames + 3)))
                            elif 180 - SCROLL_THRESHOLD_ANGLE <= angle % 360 <= 180 + SCROLL_THRESHOLD_ANGLE:
                                pyautogui.scroll(-math.ceil(5 * math.log2(scroll_frames + 3)))
                        angle_text = f"Index Angle: {angle:.2f}°"
                        cv2.putText(frame, angle_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        if 90 - SCROLL_THRESHOLD_ANGLE <= angle <= 90 + SCROLL_THRESHOLD_ANGLE:
                            threshold_text = "Scroll Up"
                            scroll_frames += 1
                        elif 180 - SCROLL_THRESHOLD_ANGLE <= angle % 360 <= 180 + SCROLL_THRESHOLD_ANGLE:
                            threshold_text = "Scroll Down"
                            scroll_frames += 1
                        else:
                            threshold_text = "No Scroll"
                        cv2.putText(frame, threshold_text, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                    # Click detection: index touching thumb.
                    distance_index_thumb = np.sqrt((index_x - thumb_x)**2 + (index_y - thumb_y)**2)
                    if distance_index_thumb < INDEX_CIRCLE_RADIUS:
                        current_time = time.time()
                        if current_time - last_click_time > click_debounce_seconds:
                            pyautogui.click()
                            last_click_time = current_time
                            click_message_time = current_time

                    # Drawing Mode for Handwriting:
                    # When ring finger touches thumb, record index tip position.
                    distance_ring_thumb = np.sqrt((ring_x - thumb_x)**2 + (ring_y - thumb_y)**2)
                    if distance_ring_thumb < RING_THUMB_THRESHOLD:
                        drawing_points.append((index_x, index_y))
                        non_touch_frames = 0
                    else:
                        non_touch_frames += 1
                        if non_touch_frames >= 5 and len(drawing_points) > 3:
                            recognized = process_drawing_easyocr(drawing_points)
                            if recognized:
                                print("Recognized letter:", recognized)
                                pyautogui.typewrite(recognized)
                            drawing_points = []
                            non_touch_frames = 0

                    # Draw the drawing trail.
                    for i, pt in enumerate(drawing_points):
                        cv2.circle(frame, pt, 3, (255, 0, 0), -1)
                        if i > 0:
                            cv2.line(frame, drawing_points[i-1], pt, (255, 0, 0), 2)

            if time.time() - click_message_time < CLICK_MESSAGE_DURATION:
                cv2.putText(frame, "Click!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Overlay window handling.
            if overlayVisible:
                cv2.imshow(WINDOW_NAME, frame)
            else:
                # Move window off-screen.
                cv2.moveWindow(WINDOW_NAME, -1000, -1000)

            cv2.waitKey(1)

        except KeyboardInterrupt:
            print("Shutting down...")
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
