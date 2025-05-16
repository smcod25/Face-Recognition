import streamlit as st
import cv2
import numpy as np
import os
import torch
import pickle
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from scipy.spatial.distance import cosine

# Initialize FaceNet model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Paths
EMBEDDINGS_PATH = "embeddings.pkl"
DATASET_DIR = "dataset"

# Load existing embeddings or create empty dict
if os.path.exists(EMBEDDINGS_PATH):
    with open(EMBEDDINGS_PATH, "rb") as f:
        embeddings_db = pickle.load(f)
else:
    embeddings_db = {}

def detect_faces(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return faces

def extract_embedding(face_img):
    img = face_img.resize((160, 160))
    img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).float()
    img_tensor = (img_tensor - 127.5) / 128.0
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        embedding = model(img_tensor).cpu().numpy()
    return embedding[0]

def save_embeddings():
    os.makedirs(DATASET_DIR, exist_ok=True)
    # Save embeddings_db to file
    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(embeddings_db, f)

def add_new_person(name, images):
    if not name:
        st.error("Please enter a name.")
        return
    person_dir = os.path.join(DATASET_DIR, name)
    os.makedirs(person_dir, exist_ok=True)
    person_embeddings = []

    for i, img in enumerate(images):
        faces = detect_faces(img)
        if len(faces) == 0:
            st.warning(f"No face detected in image {i+1}. Skipping.")
            continue
        x, y, w, h = faces[0]  # Take the first face detected
        face_img = img.crop((x, y, x+w, y+h))
        emb = extract_embedding(face_img)
        person_embeddings.append(emb)
        # Save face crop for reference
        face_img.save(os.path.join(person_dir, f"face_{i+1}.jpg"))

    if len(person_embeddings) == 0:
        st.error("No valid faces found in the images.")
        return

    # Average embedding for the person
    avg_emb = np.mean(person_embeddings, axis=0)
    embeddings_db[name] = avg_emb
    save_embeddings()
    st.success(f"Saved embeddings for {name} with {len(person_embeddings)} face(s).")

def recognize_faces_in_image(image):
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    faces = detect_faces(image)

    if len(faces) == 0:
        st.warning("No faces detected.")
        return image

    for (x, y, w, h) in faces:
        face_img = image.crop((x, y, x+w, y+h))
        emb = extract_embedding(face_img)

        # Compare with known embeddings
        min_dist = float("inf")
        identity = "Unknown"
        for person, db_emb in embeddings_db.items():
            dist = cosine(emb, db_emb)
            if dist < min_dist and dist < 0.6:  # threshold
                min_dist = dist
                identity = person

        # Draw rectangle and label
        cv2.rectangle(img_cv, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img_cv, identity, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    return cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

# Streamlit UI
st.title("Face Recognition System")

tab1, tab2 = st.tabs(["Add New Person", "Recognize Faces"])

with tab1:
    st.header("Add New Person and Save Embeddings")
    person_name = st.text_input("Enter person's name")
    uploaded_images = st.file_uploader("Upload images (multiple)", accept_multiple_files=True, type=["jpg","jpeg","png"])

    if st.button("Save Embeddings"):
        if uploaded_images and person_name:
            pil_images = [Image.open(img).convert("RGB") for img in uploaded_images]
            add_new_person(person_name, pil_images)
        else:
            st.error("Please enter a name and upload at least one image.")

with tab2:
    st.header("Face Recognition")
    uploaded_img = st.file_uploader("Upload an image for recognition", type=["jpg","jpeg","png"])
    if uploaded_img:
        img = Image.open(uploaded_img).convert("RGB")
        result_img = recognize_faces_in_image(img)
        st.image(result_img, caption="Recognition Result", use_container_width=True)
