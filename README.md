#  Multi-Face Recognition System

A deep learning–based **multi-face recognition application** that detects multiple faces in an image, identifies known individuals, and reports similarity confidence using **FaceNet (VGGFace2)** embeddings.

 **Live Demo:** (https://agents-multi-face-recognition.streamlit.app/)  
 **GitHub Repository:** (https://github.com/Raghavendra0/Multi-face-recognition.git)

---

##  Features

- Multi-face detection using **MTCNN**
- Face embedding extraction using **FaceNet (InceptionResnetV1)**
- Face search against a known face database
- Similarity score displayed in **percentage**
- Confidence-based identity classification
- Bounding boxes with identity labels on detected faces
- Interactive **Streamlit UI**
- Deployable on **Hugging Face Spaces**

---

##  How It Works

1. Upload an image containing one or more faces  
2. Detect all faces in the image  
3. Crop each detected face  
4. Generate 512-D FaceNet embeddings  
5. Compare embeddings with known faces using cosine similarity  
6. Display identity, similarity percentage, confidence level, and bounding boxes  

---

##  Tech Stack

- **Python**
- **PyTorch**
- **FaceNet (VGGFace2)**
- **MTCNN**
- **OpenCV**
- **Streamlit**
- **Hugging Face Spaces**

---
