import numpy as np
from numpy.linalg import norm

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    sim = np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))
    return float(sim)

def similarity_percentage(vec1: np.ndarray, vec2: np.ndarray) -> float:
    sim = cosine_similarity(vec1, vec2)
    return round(sim * 100, 2)

def verify_identity(emb1, emb2, threshold: float = 75.0):
    score = similarity_percentage(emb1, emb2)

    if score >= threshold:
        return {
            "similarity": score,
            "match": True,
            "message": "Same Person"
        }
    else:
        return {
            "similarity": score,
            "match": False,
            "message": "Different Person"
        }