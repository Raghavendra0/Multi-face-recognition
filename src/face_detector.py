import cv2
import numpy as np
from facenet_pytorch import MTCNN
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

mtcnn = MTCNN(
    image_size=160,
    margin=20,
    keep_all=True,
    device=device
)

def detect_faces(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Cannot read image")

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    boxes, _ = mtcnn.detect(rgb)

    if boxes is None:
        return img, []

    faces = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        faces.append((x1, y1, x2 - x1, y2 - y1))

    return img, faces
