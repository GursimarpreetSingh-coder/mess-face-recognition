import cv2
import os
import numpy as np
from insightface.app import FaceAnalysis


# ==========================================
# CONFIGURATION
# ==========================================

EMBEDDINGS_DIR = "embeddings"
THRESHOLD = 0.45

MODEL_NAME = "buffalo_l"


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

print("InsightFace loaded!")


# ==========================================
# LOAD STUDENT EMBEDDINGS
# ==========================================

database = {}

for filename in os.listdir(EMBEDDINGS_DIR):

    if not filename.endswith(".npy"):
        continue

    student_id = filename[:-4]

    path = os.path.join(
        EMBEDDINGS_DIR,
        filename
    )

    embeddings = np.load(path)

    database[student_id] = embeddings

    print(
        f"Loaded {student_id}: "
        f"{embeddings.shape}"
    )


print(
    f"\nTotal students loaded: "
    f"{len(database)}"
)


# ==========================================
# COSINE SIMILARITY
# ==========================================

def cosine_similarity(a, b):

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm == 0 or b_norm == 0:
        return 0

    return np.dot(a, b) / (
        a_norm * b_norm
    )


# ==========================================
# FIND BEST STUDENT
# ==========================================

def recognize_face(query_embedding):

    best_student = "Unknown"
    best_score = -1

    for student_id, embeddings in database.items():

        for stored_embedding in embeddings:

            score = cosine_similarity(
                query_embedding,
                stored_embedding
            )

            if score > best_score:

                best_score = score
                best_student = student_id

    if best_score >= THRESHOLD:

        return best_student, best_score

    return "Unknown", best_score


# ==========================================
# OPEN WEBCAM
# ==========================================

print("\nOpening webcam...")

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("❌ Could not open webcam.")
    exit()


print("Webcam started!")
print("Press Q to quit.")


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    ret, frame = camera.read()

    if not ret:

        print("❌ Could not read frame.")
        break


    # --------------------------------------
    # Detect faces
    # --------------------------------------

    faces = app.get(frame)


    # --------------------------------------
    # Process each face
    # --------------------------------------

    for face in faces:

        bbox = face.bbox.astype(int)

        x1, y1, x2, y2 = bbox

        # Get embedding
        embedding = face.embedding

        # Normalize
        embedding = embedding / np.linalg.norm(
            embedding
        )


        # Recognize
        student_id, score = recognize_face(
            embedding
        )


        # ----------------------------------
        # Display information
        # ----------------------------------

        if student_id != "Unknown":

            label = (
                f"{student_id} "
                f"{score:.2f}"
            )

        else:

            label = (
                f"Unknown "
                f"{score:.2f}"
            )


        # ----------------------------------
        # Draw bounding box
        # ----------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2
        )


        # ----------------------------------
        # Draw label background
        # ----------------------------------

        cv2.rectangle(
            frame,
            (x1, y1 - 35),
            (x2, y1),
            (0, 0, 0),
            -1
        )


        # ----------------------------------
        # Draw text
        # ----------------------------------

        cv2.putText(
            frame,
            label,
            (x1 + 5, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


    # ======================================
    # DISPLAY CAMERA
    # ======================================

    cv2.imshow(
        "Mess Face Recognition",
        frame
    )


    # ======================================
    # QUIT WITH Q
    # ======================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# ==========================================
# CLEANUP
# ==========================================

camera.release()

cv2.destroyAllWindows()

print("\nWebcam stopped.")