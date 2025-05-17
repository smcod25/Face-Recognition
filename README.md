# Face-Recognition
Face Recognition using OpenCV, Python
This is a face recognition system built using Streamlit for the user interface, OpenCV for face detection, and FaceNet (via facenet-pytorch) for face embedding and recognition. It enables users to add new individuals to a face database and recognize people from uploaded images based on stored embeddings.

How It Works
1. Adding a New Person
Upload one or more clear images of a person.

The system detects faces, extracts embeddings using FaceNet, and saves the average embedding.

2. Recognizing Faces
Upload an image containing one or more faces.

The system compares detected face embeddings against stored ones using cosine similarity.


