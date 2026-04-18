"""
One-time migration: converts the old single-owner face_db layout to the
new multi-person layout used by MultiPersonRecognizer + enrollment_server.

Old layout:
    face_db/
        owner_embeddings.npy
        owner_mean_embedding.npy   (optional)
        facenet.onnx               (cache — re-generated automatically)

New layout:
    face_db/
        people.json
        embeddings/
            owner_1.npy

Run once:
    python migrate_owner_db.py [--person-id owner_1] [--name "Ahmad"] [--role owner]
"""

import argparse
import json
import os
import shutil
import time

import numpy as np

_BASE           = os.path.dirname(os.path.abspath(__file__))
_DB_DIR         = os.path.join(_BASE, "face_db")
_OLD_EMB        = os.path.join(_DB_DIR, "owner_embeddings.npy")
_REGISTRY_PATH  = os.path.join(_DB_DIR, "people.json")
_EMBEDDINGS_DIR = os.path.join(_DB_DIR, "embeddings")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--person-id", default="owner_1",
                        help="New person_id for the existing owner (default: owner_1)")
    parser.add_argument("--name", default="Owner",
                        help="Display name (default: Owner)")
    parser.add_argument("--role", default="owner", choices=["owner", "family_member"])
    args = parser.parse_args()

    if not os.path.exists(_OLD_EMB):
        print(f"[migrate] Nothing to migrate — {_OLD_EMB} not found.")
        print("[migrate] Your database is already in the new format, or has never been enrolled.")
        return

    os.makedirs(_EMBEDDINGS_DIR, exist_ok=True)

    # Copy embeddings to new location
    dest = os.path.join(_EMBEDDINGS_DIR, f"{args.person_id}.npy")
    shutil.copy2(_OLD_EMB, dest)
    count = len(np.load(dest))
    print(f"[migrate] Copied {count} embeddings  →  {dest}")

    # Build / update registry
    registry = []
    if os.path.exists(_REGISTRY_PATH):
        with open(_REGISTRY_PATH) as f:
            registry = json.load(f)

    # Remove any existing entry with the same person_id
    registry = [p for p in registry if p["person_id"] != args.person_id]
    registry.insert(0, {
        "person_id":      args.person_id,
        "name":           args.name,
        "role":           args.role,
        "embedding_file": f"embeddings/{args.person_id}.npy",
        "enrolled_at":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })

    with open(_REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"[migrate] Registry written  →  {_REGISTRY_PATH}")

    # Rename old files so they are not picked up again
    os.rename(_OLD_EMB, _OLD_EMB + ".bak")
    old_mean = os.path.join(_DB_DIR, "owner_mean_embedding.npy")
    if os.path.exists(old_mean):
        os.rename(old_mean, old_mean + ".bak")

    print("\n[migrate] Done.  Old files renamed to *.bak (safe to delete).")
    print(f"[migrate] Enrolled person_id='{args.person_id}'  name='{args.name}'  "
          f"role='{args.role}'  samples={count}")


if __name__ == "__main__":
    main()
