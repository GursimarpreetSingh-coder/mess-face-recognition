import cv2
import numpy as np
from insightface.app import FaceAnalysis


print("Loading InsightFace...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

print("Model loaded!")

# Read image
image = cv2.imread("student.jpg")

if image is None:
    print("ERROR: Could not read student.jpg")
    exit()

print("Image loaded!")

# Detect faces
faces = app.get(image)

print(f"Faces detected: {len(faces)}")

if len(faces) == 0:
    print("No face detected.")
    exit()

for i, face in enumerate(faces):

    print(f"\nFace {i + 1}")

    print("Detection confidence:", face.det_score)

    print("Bounding box:", face.bbox)

    embedding = face.embedding

    print("Embedding shape:", embedding.shape)

    # Normalize embedding
    embedding = embedding / np.linalg.norm(embedding)

    print("Normalized embedding shape:", embedding.shape)

    print("First 10 embedding values:")
    print(embedding[:10])

    # Save embedding
    np.save(f"student_embedding_{i + 1}.npy", embedding)

    print(f"Saved: student_embedding_{i + 1}.npy")


print("\nFace processing complete!")