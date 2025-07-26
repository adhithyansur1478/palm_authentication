import cv2
import mediapipe as mp
import time
import os
import math
import json
import argparse
import numpy as np
import onnxruntime as ort
from datetime import datetime
from pathlib import Path

# ----------------- Args -----------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["register", "identify"], required=True,
                   help="register: enroll new user | identify: find who it is")
    p.add_argument("--user", type=str, default=None,
                   help="username to register (required for --mode register)")
    p.add_argument("--samples", type=int, default=5,
                   help="How many frames to average (both modes)")
    p.add_argument("--threshold", type=float, default=0.85,
                   help="Cosine similarity threshold for identify")
    p.add_argument("--onnx", type=str, default="palmnet_embedder.onnx",
                   help="Path to ONNX embedding model")
    p.add_argument("--db", type=str, default="users.json",
                   help="Path to JSON DB file")
    p.add_argument("--save_crops", action="store_true",
                   help="(Optional) also save each crop to disk for debugging")
    return p.parse_args()

# ----------------- Config -----------------
PALM_LANDMARKS = [0, 1, 5, 9, 13, 17]  # wrist + MCPs
CLOSE_MIN = 300     # minimum palm pixel width to trigger
CLOSE_MAX = 350     # maximum palm pixel width to trigger
SAVE_COOLDOWN_SEC = 0.6  # time between accepted captures
OUTPUT_DIR = "palm_shots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_SIZE = (224, 224)
BOX_SCALE = 1.8

# ----------------- DB helpers -----------------
def init_db(db_path):
    if not os.path.exists(db_path):
        with open(db_path, "w") as f:
            json.dump({}, f)

def load_db(db_path):
    init_db(db_path)
    with open(db_path, "r") as f:
        return json.load(f)

def save_db(db_path, db):
    with open(db_path, "w") as f:
        json.dump(db, f, indent=2)

def save_user_embedding(db_path, username, emb):
    db = load_db(db_path)
    db[username] = emb.tolist()
    save_db(db_path, db)

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

# ----------------- ONNX embedder -----------------
class Embedder:
    def __init__(self, onnx_path):
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    def preprocess_from_gray224(self, gray_224):
        rgb = np.repeat(gray_224[..., None], 3, axis=2).astype(np.float32) / 255.0
        rgb = (rgb - 0.5) / 0.5
        rgb = np.transpose(rgb, (2, 0, 1))[None, ...].astype(np.float32)
        return rgb

    def get_embedding_from_roi_gray(self, roi_gray224):
        x = self.preprocess_from_gray224(roi_gray224)
        emb = self.sess.run(["embedding"], {"input": x})[0][0]
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        return emb

# ----------------- MediaPipe Setup -----------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def main():
    args = get_args()
    if args.mode == "register" and not args.user:
        raise ValueError("--user is required in register mode")

    # Load ONNX
    embedder = Embedder(args.onnx)

    # Prepare DB
    init_db(args.db)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    prev_time = 0.0
    last_saved_time = 0.0
    collected_embs = []

    status_text = "Searching..."
    dist = 0.0

    print(f"Mode: {args.mode} | Samples to collect: {args.samples}")
    if args.mode == "identify":
        print(f"Threshold: {args.threshold}")

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

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            # Palm landmarks
            palm_points = []
            for idx in PALM_LANDMARKS:
                lx = int(hand_landmarks.landmark[idx].x * w)
                ly = int(hand_landmarks.landmark[idx].y * h)
                palm_points.append((lx, ly))

            x0, y0 = palm_points[0]    # wrist (bottom)
            x1_l, y1_l = palm_points[1]  # left side
            x9, y9 = palm_points[3]    # top
            x17, y17 = palm_points[5]  # right
            x5, y5 = palm_points[2]    # MCP index for width

            palm_pixel_width = math.dist((x5, y5), (x17, y17))
            dist = palm_pixel_width

            # Rect bounds
            x_left   = x1_l
            x_right  = x17
            y_top    = y9
            y_bottom = y0

            if x_right < x_left:
                x_left, x_right = x_right, x_left
            if y_bottom < y_top:
                y_top, y_bottom = y_bottom, y_top

            x_left   = max(x_left, 0)
            y_top    = max(y_top, 0)
            x_right  = min(x_right, w - 1)
            y_bottom = min(y_bottom, h - 1)

            if y_bottom > y_top and x_right > x_left:
                palm_crop = frame[y_top:y_bottom, x_left:x_right].copy()

            # Draw debug
            cv2.rectangle(frame, (x_left, y_top), (x_right, y_bottom), (0, 255, 0), 2)
            cv2.circle(frame, (x0, y0), 6, (0, 0, 255), -1)
            cv2.circle(frame, (x1_l, y1_l), 6, (0, 255, 255), -1)
            cv2.circle(frame, (x9, y9), 6, (255, 0, 255), -1)
            cv2.circle(frame, (x17, y17), 6, (255, 0, 0), -1)
            cv2.putText(frame, "0 (bottom)", (x0 + 5, y0 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, "1 (left)", (x1_l + 5, y1_l - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, "9 (top)", (x9 + 5, y9 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            cv2.putText(frame, "17 (right)", (x17 + 5, y17 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # Distance gate
            if CLOSE_MIN < palm_pixel_width < CLOSE_MAX:
                status_text = f"READY - Capturing at {palm_pixel_width:.2f}px"
                now = time.time()
                if now - last_saved_time >= SAVE_COOLDOWN_SEC:
                    if palm_crop is not None and palm_crop.size > 0:
                        # Resize, gray, embed
                        roi_gray = cv2.cvtColor(palm_crop, cv2.COLOR_BGR2GRAY)
                        roi_gray224 = cv2.resize(roi_gray, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)
                        emb = embedder.get_embedding_from_roi_gray(roi_gray224)
                        collected_embs.append(emb)

                        if args.save_crops:
                            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                            out_path = os.path.join(OUTPUT_DIR, f"{args.mode}_{ts}.png")
                            cv2.imwrite(out_path, palm_crop)

                        last_saved_time = now
                        print(f"[{len(collected_embs)}/{args.samples}] Captured at distance {palm_pixel_width:.2f}px")
                    should_save = True
            else:
                if palm_pixel_width <= CLOSE_MIN:
                    status_text = f"Too Far ({dist:.2f}px)"
                else:
                    status_text = f"Too Close ({dist:.2f}px)"
        else:
            status_text = "Searching..."

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

        # Stop if enough collected
        if len(collected_embs) >= args.samples:
            break

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            hands.close()
            cap.release()
            cv2.destroyAllWindows()
            print("Cancelled.")
            return

    hands.close()
    cap.release()
    cv2.destroyAllWindows()

    # ---------- After loop: register / identify ----------
    if not collected_embs:
        print("No samples collected. Exiting.")
        return

    test_emb = np.mean(collected_embs, axis=0)
    test_emb /= (np.linalg.norm(test_emb) + 1e-12)

    if args.mode == "register":
        save_user_embedding(args.db, args.user, test_emb)
        print(f"✅ Registered '{args.user}' with {len(collected_embs)} samples.")
        return

    # identify
    db = load_db(args.db)
    if not db:
        print("DB is empty. Register someone first.")
        return

    best_user, best_sim = None, -1.0
    for user, emb_list in db.items():
        emb_user = np.array(emb_list, dtype=np.float32)
        sim = cosine_sim(test_emb, emb_user)
        if sim > best_sim:
            best_user, best_sim = user, sim

    print(f"Best match: {best_user} (cosine={best_sim:.4f})")
    if best_sim >= args.threshold:
        print(f"✅ IDENTIFIED as '{best_user}'")
    else:
        print(f"❌ UNKNOWN (below threshold {args.threshold})")

if __name__ == "__main__":
    main()
