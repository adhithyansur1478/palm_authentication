import json
import os
import numpy as np

DB_PATH = "users.json"

# Initialize JSON if it doesn't exist
if not os.path.exists(DB_PATH):
    with open(DB_PATH, "w") as f:
        json.dump({}, f)

def load_db():
    with open(DB_PATH, "r") as f:
        return json.load(f)

def save_db(db):
    with open(DB_PATH, "w") as f:
        json.dump(db, f, indent=4)

def save_user_embedding(username, emb):
    db = load_db()
    db[username] = emb.tolist()  # store as list for JSON compatibility
    save_db(db)
    print(f"User '{username}' registered.")
