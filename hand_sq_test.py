import cv2
import mediapipe as mp
import time
import os
import math
from datetime import datetime
import model

# ----------------- Config -----------------
PALM_LANDMARKS = [0, 1, 5, 9, 13, 17]  # wrist + MCPs
CLOSE_MIN = 300
CLOSE_MAX = 350
SAVE_COOLDOWN_SEC = 2.0
OUTPUT_DIR = "palm_shots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------- Smoothing Config -----------------
SMOOTHING_ALPHA = 0.4  # 0 = very smooth but laggy, 1 = no smoothing

def liv_vdo(max_samples=3, debug=False):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )

    cap = cv2.VideoCapture(0)
    prev_time = 0.0
    last_saved_time = 0.0
    saved_paths = []
    collected_embs = []

    smoothed_box = None  # For bounding box smoothing

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Couldn't read from camera.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        palm_crop = None
        should_save = False
        status_text = "Searching..."
        dist = 0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Collect landmarks
                palm_points = []
                for idx in PALM_LANDMARKS:
                    lx = int(hand_landmarks.landmark[idx].x * w)
                    ly = int(hand_landmarks.landmark[idx].y * h)
                    palm_points.append((lx, ly))

                xs = [pt[0] for pt in palm_points]
                ys = [pt[1] for pt in palm_points]

                # Straight bounding box
                x_left, x_right = max(min(xs), 0), min(max(xs), w - 1)
                y_top, y_bottom = max(min(ys), 0), min(max(ys), h - 1)

                # --- Smooth the bounding box ---
                current_box = [x_left, y_top, x_right, y_bottom]
                if smoothed_box is None:
                    smoothed_box = current_box
                else:
                    smoothed_box = [
                        int(SMOOTHING_ALPHA * current_box[i] + (1 - SMOOTHING_ALPHA) * smoothed_box[i])
                        for i in range(4)
                    ]
                x_left, y_top, x_right, y_bottom = smoothed_box

                # Crop palm
                if x_right > x_left and y_bottom > y_top:
                    palm_crop = frame[y_top:y_bottom, x_left:x_right].copy()
                    palm_crop = cv2.resize(palm_crop, (224, 224))

                # Draw bounding box
                cv2.rectangle(frame, (x_left, y_top), (x_right, y_bottom), (0, 255, 0), 2)

                # Palm width proxy for trigger
                x5, y5 = palm_points[2]
                x17, y17 = palm_points[5]
                palm_pixel_width = math.dist((x5, y5), (x17, y17))
                dist = palm_pixel_width

                # Trigger capture
                if CLOSE_MIN < palm_pixel_width < CLOSE_MAX:
                    status_text = f"READY - Capturing at {palm_pixel_width:.2f}px"
                    now = time.time()
                    if now - last_saved_time >= SAVE_COOLDOWN_SEC and palm_crop is not None:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        out_path = os.path.join(OUTPUT_DIR, f"palm_{ts}.png")
                        cv2.imwrite(out_path, palm_crop)

                        emb = model.get_embedding(palm_crop, debugg=debug)
                        collected_embs.append(emb)
                        saved_paths.append(out_path)
                        last_saved_time = now
                        should_save = True
                        print(f"✅ Saved: {out_path} (w={x_right - x_left}, h={y_bottom - y_top})")

                        if len(saved_paths) >= max_samples:
                            cap.release()
                            cv2.destroyAllWindows()
                            return collected_embs, palm_crop
                else:
                    status_text = f"Too Far ({dist:.2f}px)" if palm_pixel_width <= CLOSE_MIN else f"Too Close ({dist:.2f}px)"

        # FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Status
        cv2.putText(frame, status_text, (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0) if should_save else (0, 0, 255), 2)

        # Show frames
        if palm_crop is not None:
            cv2.imshow("Palm Only (Crop)", palm_crop)
        cv2.imshow("Full Frame (Debug)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in [27, ord('q')]:
            break

    cap.release()
    cv2.destroyAllWindows()


# Run ##########################
liv_vdo(3, True)
