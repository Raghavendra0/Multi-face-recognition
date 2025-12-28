import os
import cv2
import streamlit as st

from src.face_detector import detect_faces
from src.face_embedding import get_embedding
from src.face_database import build_face_database, identify_face
from src.confidence import confidence_label


# -------------------------------
# App UI
# -------------------------------
st.set_page_config(page_title="Multi-Face Recognition", layout="wide")
st.title("Multi-Face Recognition System")
st.write("Upload an image containing one or more faces")

# -------------------------------
# Load face database once
# -------------------------------
@st.cache_resource
def load_face_db():
    return build_face_database("data/known_faces")

face_db = load_face_db()

# -------------------------------
# File uploader
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------
# Main logic
# -------------------------------
if uploaded_file:
    os.makedirs("temp", exist_ok=True)
    image_path = f"temp/{uploaded_file.name}"

    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Detect faces
    img, faces = detect_faces(image_path)

    if len(faces) == 0:
        st.error(" No faces detected")
    else:
        st.success(f" Detected {len(faces)} face(s)")

        # Process each face
        for i, (x, y, w, h) in enumerate(faces):
            face = img[y:y+h, x:x+w]

            if face.shape[0] < 80 or face.shape[1] < 80:
                continue

            emb = get_embedding(face)
            result = identify_face(emb, face_db)

            similarity = result["similarity"]
            label, color = confidence_label(similarity)

            # Draw rectangle
            box_color = (0, 255, 0) if result["status"] == "Known Person" else (0, 0, 255)
            cv2.rectangle(img, (x, y), (x+w, y+h), box_color, 2)

            # Draw label
            text = f"{result['identity']} ({similarity:.1f}%)"
            cv2.putText(
                img,
                text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                box_color,
                2
            )

            # Info card per face
            st.markdown(f"###  Face {i+1}")
            st.markdown(
                f"""
                <div style="padding:12px;border-radius:10px;border:2px solid {color}">
                <b>Identity:</b> {result['identity']}<br>
                <b>Similarity:</b> {similarity:.2f}%<br>
                <b>Status:</b> {label}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(min(int(similarity), 100))
            st.divider()

        # -------------------------------
        # SHOW FINAL IMAGE WITH BOXES
        # -------------------------------
        st.subheader("Final Result")
        st.image(img, channels="BGR", use_column_width=True)
