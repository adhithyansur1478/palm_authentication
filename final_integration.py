import os

import cv2

import auth
import testtt
import model
import numpy as np

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


# ----------- Step 1: Register User -----------
def register_user(username, num_samples=3,debug=False):
    print(f"Collecting {num_samples} palm samples for {username}...")
    embeddings = []

    # Assign a user ID
    user_id = get_next_user_id()
    user_folder = os.path.join(REGISTERED_DIR, f"{user_id:03d}_P_{username}")
    os.makedirs(user_folder, exist_ok=True)

    for i in range(num_samples):
        emb,img = testtt.liv_vdo(1,debug=debug)  # capture 1 embedding per shot
        embeddings.append(ensure_vector(emb))

        # Save palm image
        img_path = os.path.join(user_folder, f"{user_id:03d}_P_{username}_{i + 1}.png")
        cv2.imwrite(img_path, img)
        print(f"📸 Saved image: {img_path}")

    # Step 2: Average + normalize embedding
    avg_emb = ensure_vector(np.mean(embeddings, axis=0))

    # Save embedding to DB
    auth.save_user_embedding(username, avg_emb)
    print(f"✅ Registration completed for {username}.")


# ----------- Step 2: Match User -----------
def cosine_similarity(vec1, vec2):
    vec1 = ensure_vector(vec1)
    vec2 = ensure_vector(vec2)
    return np.dot(vec1, vec2)


def match_user(num_samples=1, threshold=0.85,debug=False):
    print("Capturing palm for authentication...")
    emb,img = testtt.liv_vdo(num_samples,debug=debug)

    # If multiple embeddings captured, average them
    if isinstance(emb, list):
        emb = np.mean([ensure_vector(e) for e in emb], axis=0)

    emb = ensure_vector(emb)

    db = auth.load_db()
    best_match = None
    best_score = -1

    for username, stored_emb in db.items():
        score = cosine_similarity(emb, stored_emb)
        if score > best_score:
            best_score = score
            best_match = username

    print(f"Best match: {best_match} (cosine={best_score:.4f})")
    if best_score >= threshold:
        print(f"Authentication successful: {best_match} (score={best_score:.3f})")
        return best_match
    else:
        print(f"No match found. Best score={best_score:.3f}")
        return None


# ----------- Test -----------

#register_user("adhii", 10,debug=False)
match_user(5,debug=False)
