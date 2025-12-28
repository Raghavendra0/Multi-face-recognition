from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import os
import shutil

from src.face_detector import detect_faces
from src.face_embedding import get_embedding
from src.face_database import build_face_database, identify_face

app = FastAPI(title="Multi-Face Recognition API")

# Load DB once
face_db = build_face_database("data/known_faces")

@app.post("/recognize")
async def recognize_faces(file: UploadFile = File(...)):
    temp_path = f"temp/{file.filename}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    img, faces = detect_faces(temp_path)

    results = []

    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        if face.shape[0] < 80 or face.shape[1] < 80:
            continue

        emb = get_embedding(face)
        result = identify_face(emb, face_db)

        results.append({
            "identity": result["identity"],
            "similarity": round(result["similarity"], 2),
            "status": result["status"],
            "box": [int(x), int(y), int(w), int(h)]
        })

    return {
        "faces_detected": len(results),
        "results": results
    }
