import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis


# ==========================================
# CONFIG
# ==========================================

EMBEDDINGS_DIR = "embeddings"
TEST_IMAGE = "student_test.jpg"

MODEL_NAME = "buffalo_l"

# Recognition threshold
THRESHOLD = 0.45


# ==========================================
# LOAD INSIGHTFACE
# ==========================================

print("Loading InsightFace...")

app = FaceAnalysis(
    name=MODEL_NAME,
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

print("InsightFace loaded!\n")


# ==========================================
# COSINE SIMILARITY
# ==========================================

def cosine_similarity(a, b):

    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    return np.dot(a, b)


# ==========================================
# LOAD TEST IMAGE
# ==========================================

image = cv2.imread(TEST_IMAGE)

if image is None:

    print(f"ERROR: Could not read {TEST_IMAGE}")
    exit()

print(f"Test image loaded: {TEST_IMAGE}")


# ==========================================
# DETECT FACE
# ==========================================

faces = app.get(image)

print(f"Faces detected: {len(faces)}")


if len(faces) == 0:

    print("❌ No face detected.")
    exit()


if len(faces) > 1:

    print("❌ Multiple faces detected.")
    print("Please use an image containing one person.")
    exit()


# ==========================================
# GET QUERY EMBEDDING
# ==========================================

face = faces[0]

query_embedding = face.embedding

query_embedding = (
    query_embedding /
    np.linalg.norm(query_embedding)
)

print("Query embedding generated.")
print("Embedding shape:", query_embedding.shape)


# ==========================================
# SEARCH ALL STUDENTS
# ==========================================

best_student = None
best_score = -1


for filename in os.listdir(EMBEDDINGS_DIR):

    if not filename.endswith(".npy"):
        continue

    student_id = filename[:-4]

    path = os.path.join(
        EMBEDDINGS_DIR,
        filename
    )

    stored_embeddings = np.load(path)

    print(
        f"\nComparing with {student_id} "
        f"({len(stored_embeddings)} embeddings)"
    )


    # Compare query against every enrollment image

    scores = []

    for stored_embedding in stored_embeddings:

        score = cosine_similarity(
            query_embedding,
            stored_embedding
        )

        scores.append(score)


    # Best score for this student

    student_best_score = max(scores)

    print(
        f"Best similarity: "
        f"{student_best_score:.4f}"
    )


    if student_best_score > best_score:

        best_score = student_best_score
        best_student = student_id


# ==========================================
# FINAL DECISION
# ==========================================

print("\n")
print("=" * 50)
print("RECOGNITION RESULT")
print("=" * 50)

print(f"Best student : {best_student}")
print(f"Similarity   : {best_score:.4f}")
print(f"Threshold    : {THRESHOLD:.4f}")


if best_score >= THRESHOLD:

    print("\n✅ FACE RECOGNIZED")

    print(
        f"Student ID: {best_student}"
    )

else:

    print("\n❌ UNKNOWN PERSON")

print("=" * 50)