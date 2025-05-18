# Face-Recognition
Face Recognition using OpenCV, Python.

---

# 🧠 Face Recognition System

A simple and efficient web-based *Face Recognition System* built using *Streamlit, **FaceNet (InceptionResnetV1), and **OpenCV*. This app allows you to:

* Register new individuals with multiple face images.
* Generate and store facial embeddings using a pretrained deep learning model.
* Recognize faces in uploaded images by comparing with saved identities.

---

## 🔧 Features

* 🔍 *Face Detection* using OpenCV Haar Cascades.
* 🤖 *Face Embedding* generation with facenet-pytorch (pretrained on VGGFace2).
* 🧠 *Similarity Matching* via cosine distance.
* 📁 *Persistent Storage* of embeddings using pickle.
* 🖼 *Web UI* with Streamlit for adding users and recognizing faces.
* ✅ Fixed label font size for consistent recognition results across all image sizes.

---

## 🛠 Tech Stack

* *Python 3.7+*
* [Streamlit](https://streamlit.io/) — Web app UI
* [OpenCV](https://opencv.org/) — Face detection
* [facenet-pytorch](https://github.com/timesler/facenet-pytorch) — Deep face recognition model
* [Torch](https://pytorch.org/) — Model backend
* [Pillow](https://python-pillow.org/) — Image handling
* [Scipy](https://www.scipy.org/) — Cosine distance for matching

---

## 📦 Installation

1. *Clone the repository*:

   bash
   git clone https://github.com/your-username/face-recognition-streamlit.git
   cd face-recognition-streamlit
   

2. *Create a virtual environment* (recommended):

   bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   

3. *Install dependencies*:

   bash
   pip install -r requirements.txt
   

   If requirements.txt is not created yet, here are the required packages:

   bash
   pip install streamlit opencv-python pillow facenet-pytorch torch torchvision scipy
   

---

## 🚀 Run the Application

bash
streamlit run app.py


This will launch the app in your default web browser.

---

## 📂 Project Structure


face-recognition-streamlit/
│
├── app.py                  # Main Streamlit application
├── embeddings.pkl          # Saved facial embeddings (auto-generated)
├── dataset/                # Directory to store cropped face images
├── requirements.txt        # Python dependencies (optional)
└── README.md               # Project documentation


---

## 🧪 How it Works

### 1. Add New Person

* Upload one or more images of a person.
* App detects faces in each image.
* Generates embeddings using InceptionResnetV1.
* Stores the *average embedding* of the person's faces.

### 2. Recognize Faces

* Upload an image.
* Faces are detected and embeddings are extracted.
* Embeddings are compared to known identities using *cosine similarity*.
* Labels are drawn on recognized faces.

---

## ⚙ Configuration

* *Recognition Threshold*: The cosine distance threshold is set to 0.3 in recognize_faces_in_image(). You can adjust it based on accuracy needs.

---

## 🛡 Notes

* All images and embeddings are stored locally.
* The model runs inference on GPU (if available) or CPU.
* Ensure clear and frontal face images for best results.

---


## 👨‍💻 Author

*Shubham Mishra*

---




