import cv2
import mediapipe as mp
import time
import os
import math
from datetime import datetime
import model

# ----------------- Config -----------------
PALM_LANDMARKS = [0, 1, 5, 9, 13, 17]  # wrist + MCPs
CLOSE_MIN = 300     # minimum palm pixel width to trigger
CLOSE_MAX = 350     # maximum palm pixel width to trigger
SAVE_COOLDOWN_SEC = 2.0
OUTPUT_DIR = "palm_shots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------- Smoothing Config -----------------
SMOOTHING_ALPHA = 0.4  # 0 = very smooth but laggy, 1 = no smoothing

def get_orientation(landmarks, image_width, image_height):
    import math
    wrist = landmarks[0]
    index_mcp = landmarks[5]
    pinky_mcp = landmarks[17]
    middle_mcp = landmarks[9]

    def to_px(pt):
        return int(pt.x * image_width), int(pt.y * image_height)

    index_px = to_px(index_mcp)
    pinky_px = to_px(pinky_mcp)
    middle_px = to_px(middle_mcp)

    hand_vector_x = index_px[0] - pinky_px[0]
    hand_vector_y = index_px[1] - pinky_px[1]
    angle = math.degrees(math.atan2(hand_vector_y, hand_vector_x))

    if -180 <= angle <= -150:
        orientation = "Straight"
    elif angle > 150:
        orientation = "Tilted Left"
    elif 0 < angle <= 150:
        orientation = "Left"
    elif angle > -150 and angle < -130:
        orientation = "Tilted Right"
    elif angle > -130:
        orientation = "Right"
    else:
        orientation = "Unknown"

    return orientation, angle


def liv_vdo(max_samples=3, debug=False):
    # ----------------- MediaPipe Setup -----------------
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )

    cap = cv2.VideoCapture(0)
    prev_time = 0.0
    last_saved_time = 0.0
    dist = 0.0
    saved_paths = []
    collected_embs = []

    smoothed_box = None  # for bounding box smoothing

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Couldn't read from camera.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Process frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        palm_crop = None
        should_save = False
        status_text = "Searching..."

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Collect palm landmarks
                palm_points = []
                for idx in PALM_LANDMARKS:
                    lx = int(hand_landmarks.landmark[idx].x * w)
                    ly = int(hand_landmarks.landmark[idx].y * h)
                    palm_points.append((lx, ly))
                h, w, _ = frame.shape
                orientation_text, angle_val = get_orientation(hand_landmarks.landmark, w, h)
                cv2.putText(frame, f"Orientation: {orientation_text}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Store the current detected orientation globally
                global last_orientation
                last_orientation = orientation_text

                # Compute raw bounding box (unrotated)
                xs = [pt[0] for pt in palm_points]
                ys = [pt[1] for pt in palm_points]
                x_left, x_right = max(min(xs), 0), min(max(xs), w - 1)
                y_top, y_bottom = max(min(ys), 0), min(max(ys), h - 1)

                # --- Smooth the bounding box (anti-jitter) ---
                current_box = [x_left, y_top, x_right, y_bottom]
                if smoothed_box is None:
                    smoothed_box = current_box
                else:
                    smoothed_box = [
                        int(SMOOTHING_ALPHA * current_box[i] + (1 - SMOOTHING_ALPHA) * smoothed_box[i])
                        for i in range(4)
                    ]
                x_left, y_top, x_right, y_bottom = smoothed_box

                # Clip to frame
                x_left = max(x_left, 0)
                y_top = max(y_top, 0)
                x_right = min(x_right, w - 1)
                y_bottom = min(y_bottom, h - 1)

                # Crop
                if y_bottom > y_top and x_right > x_left:
                    palm_crop = frame[y_top:y_bottom, x_left:x_right].copy()

                # Draw debug rectangle
                cv2.rectangle(frame, (x_left, y_top), (x_right, y_bottom), (0, 255, 0), 2)

                # Palm width proxy (distance between landmarks 5 and 17)
                x5, y5 = palm_points[2]
                x17, y17 = palm_points[5]
                palm_pixel_width = math.dist((x5, y5), (x17, y17))
                dist = palm_pixel_width

                # Trigger capture
                if CLOSE_MIN < palm_pixel_width < CLOSE_MAX:
                    status_text = f"READY - Capturing at {palm_pixel_width:.2f} px"
                    now = time.time()
                    if now - last_saved_time >= SAVE_COOLDOWN_SEC:
                        if palm_crop is not None and palm_crop.size > 0:
                            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                            out_path = os.path.join(OUTPUT_DIR, f"palm_{ts}.png")
                            emb = model.get_embedding(palm_crop, debugg=debug)
                            collected_embs.append(emb)
                            cv2.imwrite(out_path, palm_crop)
                            saved_paths.append(out_path)
                            last_saved_time = now
                            print(f"Saved: {out_path} at distance {palm_pixel_width:.2f} px "
                                  f"(w={x_right - x_left}px, h={y_bottom - y_top}px)")
                            if len(saved_paths) >= max_samples:
                                cap.release()
                                cv2.destroyAllWindows()
                                return collected_embs, palm_crop
                        should_save = True
                else:
                    if palm_pixel_width <= CLOSE_MIN:
                        status_text = f"Too Far ({dist:.2f} px)"
                    else:
                        status_text = f"Too Close ({dist:.2f} px)"

        # FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if prev_time != 0 else 0.0
        prev_time = curr_time
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Status
        cv2.putText(frame, status_text, (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0) if should_save else (0, 0, 255), 2)

        # Show
        if palm_crop is not None and palm_crop.size > 0:
            cv2.imshow("Palm Only (Crop)", palm_crop)
        else:
            cv2.imshow("Palm Only (Crop)", frame)

        cv2.imshow("Full Frame (Debug)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

last_orientation = "Unknown"

def get_last_orientation():
    global last_orientation
    return last_orientation
