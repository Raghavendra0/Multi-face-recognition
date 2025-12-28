import os
import numpy as np
from src.face_detector import detect_faces
from src.face_embedding import get_embedding
from src.similarity import similarity_percentage

def build_face_database(base_dir):
    database = {}

    for person in os.listdir(base_dir):
        person_dir = os.path.join(base_dir, person)
        if not os.path.isdir(person_dir):
            continue

        embeddings = []

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)

            try:
                img, faces = detect_faces(img_path)
                if len(faces) == 0:
                    continue
                x, y, w, h = faces[0]
                face = img[y:y+h, x:x+w]
                emb = get_embedding(face)
                embeddings.append(emb)

            except Exception as e:
                print(f" Skipping {img_path}: {e}")
                continue

        if len(embeddings) > 0:
            database[person] = np.mean(embeddings, axis=0)

    return database

def identify_face(query_emb, database, threshold=75.0):
    best_identity = "Unknown"
    best_score = 0.0

    for identity, db_emb in database.items():
        score = similarity_percentage(query_emb, db_emb)
        if score > best_score:
            best_score = score
            best_identity = identity

    if best_score < threshold:
        return {
            "identity": "Unknown",
            "similarity": best_score,
            "status": "No Match"
        }

    return {
        "identity": best_identity,
        "similarity": best_score,
        "status": "Known Person"
    }

