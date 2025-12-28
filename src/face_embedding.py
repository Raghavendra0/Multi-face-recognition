import cv2
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1

# ---------- DEVICE ----------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- MODEL ----------
model = InceptionResnetV1(pretrained="vggface2").to(device).eval()


def preprocess_face(face_img):
    """Convert face image to tensor"""
    face = cv2.resize(face_img, (160, 160))
    face = face[:, :, ::-1].copy()  # BGR → RGB
    face = torch.tensor(face).permute(2, 0, 1).float() / 255.0
    return face


def get_embedding(face_img):
    """
    Get embedding for a single face
    """
    if face_img is None or not isinstance(face_img, np.ndarray) or face_img.size == 0:
        raise ValueError("Invalid face image")

    face_tensor = preprocess_face(face_img).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(face_tensor)

    embedding = embedding.cpu().numpy()[0]
    embedding = embedding / np.linalg.norm(embedding)

    return embedding


def get_embeddings_batch(face_images):
    """
    Get embeddings for multiple faces (FASTER)
    """
    tensors = []

    for face in face_images:
        if face is None or face.size == 0:
            continue
        tensors.append(preprocess_face(face))

    if len(tensors) == 0:
        return np.array([])

    batch = torch.stack(tensors).to(device)

    with torch.no_grad():
        embeddings = model(batch)

    embeddings = embeddings.cpu().numpy()
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    return embeddings
