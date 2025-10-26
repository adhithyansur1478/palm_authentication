import sys, traceback

print(">>> Starting final_integration.py")

try:
    import os
    import cv2
    import auth
    import testtt
    import model
    import numpy as np
    print(">>> All imports successful")

except Exception as e:
    print("❌ Import error:", e)
    traceback.print_exc(file=sys.stdout)  # force full error output
    input("\nPress Enter to exit...")
    sys.exit(1)


REGISTERED_DIR = "registered_palm"
os.makedirs(REGISTERED_DIR, exist_ok=True)


# ----------- Helper: Ensure Correct Embedding Shape -----------
def ensure_vector(emb):
    """
    Ensure embedding is a 1D numpy array and normalized.
    Handles cases like (1,512).
    """
    emb = np.array(emb, dtype=np.float32).flatten()
    emb /= (np.linalg.norm(emb) + 1e-12)
    return emb

def get_next_user_id():
    """Finds the next available numeric user ID."""
    existing = [int(d.split("_")[0]) for d in os.listdir(REGISTERED_DIR) if os.path.isdir(os.path.join(REGISTERED_DIR, d)) and d.split("_")[0].isdigit()]
    return max(existing) + 1 if existing else 1

def register_user(username, num_samples=6, debug=False):
    print(f"Collecting {num_samples} palm samples for {username}...")

    # Orientation-based embedding storage
    orientation_embs = {"Left": [], "Right": [], "Straight": []}

    # Assign user ID and folder
    user_id = get_next_user_id()
    user_folder = os.path.join(REGISTERED_DIR, f"{user_id:03d}_P_{username}")
    os.makedirs(user_folder, exist_ok=True)

    for i in range(num_samples):
        print(f"\n📸 Capture {i + 1}/{num_samples}")
        emb, img = testtt.liv_vdo(1, debug=debug)

        # Get last detected orientation
        orientation = testtt.get_last_orientation()

        # Ignore uncertain detections
        if orientation == "Unknown":
            print("⚠️ Orientation unclear — skipping this frame.")
            continue

        # Merge tilted orientations into main categories
        if "Left" in orientation:
            orientation_key = "Left"
        elif "Right" in orientation:
            orientation_key = "Right"
        else:
            orientation_key = "Straight"

        emb = ensure_vector(emb)
        orientation_embs[orientation_key].append(emb)

        # Save palm image
        img_path = os.path.join(
            user_folder, f"{user_id:03d}_P_{username}_{orientation_key}_{i + 1}.png"
        )
        cv2.imwrite(img_path, img)
        print(f"📁 Saved image ({orientation_key}): {img_path}")

    # ---------- Compute orientation averages ----------
    avg_data = {}
    orientation_avgs = []

    for orient, embs in orientation_embs.items():
        if embs:  # only if data exists
            avg_emb = ensure_vector(np.mean(embs, axis=0))
            avg_data[orient] = avg_emb.tolist()
            orientation_avgs.append(avg_emb)
            print(f"✅ {orient}: {len(embs)} samples saved.")
        else:
            avg_data[orient] = None
            print(f"⚠️ No samples found for {orient} orientation.")

    # ---------- Compute overall average ----------
    if orientation_avgs:
        overall_avg = ensure_vector(np.mean(orientation_avgs, axis=0))
        avg_data["Average"] = overall_avg.tolist()
        print("✅ Overall average embedding computed.")
    else:
        avg_data["Average"] = None
        print("⚠️ No valid embeddings found — check camera/orientation settings.")

    # ---------- Save structured data ----------
    auth.save_user_embedding(username, avg_data)
    print(f"\n✅ Registration completed for {username} with orientation-based storage.")


# ----------- Step 2: Match User -----------
def cosine_similarity(vec1, vec2):
    vec1 = ensure_vector(vec1)
    vec2 = ensure_vector(vec2)
    return np.dot(vec1, vec2)


def match_user(num_samples=1, threshold=0.85, debug=False):
    print("Capturing palm for authentication...")
    emb, img = testtt.liv_vdo(num_samples, debug=debug)

    # Handle multiple embeddings (average)
    if isinstance(emb, list):
        emb = np.median([ensure_vector(e) for e in emb], axis=0)
    emb = ensure_vector(emb)

    # Get detected orientation from the last frame
    orientation = testtt.get_last_orientation()
    if "Left" in orientation:
        curr_orient = "Left"
    elif "Right" in orientation:
        curr_orient = "Right"
    else:
        curr_orient = "Straight"

    print(f"🖐️ Detected Orientation: {curr_orient}")

    db = auth.load_db()
    best_match = None
    best_score = -1
    best_type = None  # "Orientation" or "Average"

    for username, data in db.items():
        # Skip if user has no embeddings
        if not isinstance(data, dict):
            continue

        orient_emb = data.get(curr_orient)
        avg_emb = data.get("Average")

        score_orient = cosine_similarity(emb, orient_emb) if orient_emb is not None else -1
        score_avg = cosine_similarity(emb, avg_emb) if avg_emb is not None else -1

        # Hybrid logic: prefer orientation match if confident enough
        if score_orient >= threshold:
            score = score_orient
            match_type = "Orientation"
        elif score_avg >= threshold:
            score = score_avg
            match_type = "Average"
        else:
            # pick best among the two (even if below threshold)
            if score_orient >= score_avg:
                score = score_orient
                match_type = "Orientation"
            else:
                score = score_avg
                match_type = "Average"

        # Track best overall
        if score > best_score:
            best_score = score
            best_match = username
            best_type = match_type

        print(f"[{username}] {curr_orient}={score_orient:.3f}, Avg={score_avg:.3f}")

    # Final decision
    print(f"\nBest match: {best_match} (score={best_score:.4f}, type={best_type})")
    if best_score >= threshold:
        print(f"✅ Authentication successful for {best_match} using {best_type} match!")
        return best_match
    else:
        print(f"❌ No confident match found (best={best_score:.3f})")
        return None



# ----------- Test -----------

#register_user("chuppy", 10,debug=False)
#match_user(5,debug=False)


# ----------- Entry Point (Menu) -----------
if __name__ == "__main__":
    try:
        print("Palm Recognition System")
        print("1 = Register User")
        print("2 = Match User")
        choice = input("Enter choice: ")

        if choice == "1":
            username = input("Enter username: ")
            samples = int(input("Number of samples (default=3): ") or 10)
            register_user(username, samples, debug=False)
        elif choice == "2":
            samples = int(input("Number of samples (default=1): ") or 5)
            match_user(samples)
        else:
            print("❌ Invalid choice")

    except Exception as e:
        import traceback
        print("\n❌ An error occurred:",e)
        traceback.print_exc(file=sys.stdout)  # force full error details

    input("\nPress Enter to exit...")

