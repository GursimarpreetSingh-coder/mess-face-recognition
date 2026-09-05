import cv2
import os
import time
import numpy as np
from datetime import datetime

from insightface.app import FaceAnalysis
import database


# ==========================================
# CONFIGURATION
# ==========================================

EMBEDDINGS_DIR = "embeddings"

# Start with this threshold.
# We will calibrate it later using real data.
THRESHOLD = 0.45

# Minimum seconds before processing the same
# student again.
COOLDOWN_SECONDS = 5

MODEL_NAME = "buffalo_l"


# ==========================================
# MEAL CONFIGURATION
# ==========================================

def get_current_meal():

    hour = datetime.now().hour

    if 7 <= hour < 10:
        return "breakfast"

    elif 12 <= hour < 15:
        return "lunch"

    elif 19 <= hour < 22:
        return "dinner"

    return None
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
# LOAD DATABASE
# ==========================================

database.initialize_database()


# ==========================================
# LOAD EMBEDDINGS
# ==========================================

student_database = {}

for filename in os.listdir(EMBEDDINGS_DIR):

    if not filename.endswith(".npy"):
        continue

    student_id = filename[:-4]

    path = os.path.join(
        EMBEDDINGS_DIR,
        filename
    )

    embeddings = np.load(path)

    student_database[student_id] = embeddings

    print(
        f"Loaded {student_id}: "
        f"{embeddings.shape}"
    )


print(
    f"\nTotal students: "
    f"{len(student_database)}"
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
# RECOGNITION
# ==========================================

def recognize_face(query_embedding):

    best_student = None
    best_score = -1

    for student_id, embeddings in student_database.items():

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

    return None, best_score


# ==========================================
# COOLDOWN TRACKING
# ==========================================

last_processed = {}


# ==========================================
# OPEN CAMERA
# ==========================================

print("\nOpening webcam...")

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("❌ Could not open webcam.")
    exit()


print("Webcam started.")
print("Press Q to quit.")


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    ret, frame = camera.read()

    if not ret:

        print("❌ Camera frame error.")
        break


    # --------------------------------------
    # Current meal
    # --------------------------------------

    current_meal = get_current_meal()


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

        norm = np.linalg.norm(embedding)

        if norm == 0:
            continue

        embedding = embedding / norm


        # Recognize

        student_id, score = recognize_face(
            embedding
        )
student_info = None

if student_id is not None:
    student_info = database.get_student(student_id)

        # ==================================
        # UNKNOWN PERSON
        # ==================================

        if student_id is None:

            label = f"Unknown | {score:.2f}"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2
            )

        else:

            label = (
                f"{student_id} | "
                f"{score:.2f}"
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # ==================================
            # ATTENDANCE
            # ==================================

            if current_meal is not None:

                current_time = time.time()

                last_time = last_processed.get(
                    student_id,
                    0
                )

                # Only process after cooldown

                if (
                    current_time - last_time
                    >= COOLDOWN_SECONDS
                ):

                    last_processed[
                        student_id
                    ] = current_time


                    # Check attendance

                    already = database.already_marked(
                        student_id,
                        current_meal
                    )


                    if already:

                        print(
                            f"⚠ {student_id} "
                            f"already marked for "
                            f"{current_meal}"
                        )

                    else:

                        success = database.mark_attendance(
                            student_id,
                            current_meal,
                            float(score)
                        )

                        if success:

                            print(
                                f"✅ {student_id} "
                                f"→ {current_meal} "
                                f"attendance marked"
                            )


            else:

                print(
                    f"Recognized {student_id}, "
                    f"but no active meal."
                )


        # ==================================
        # DRAW LABEL
        # ==================================

        cv2.rectangle(
            frame,
            (x1, y1 - 35),
            (x2, y1),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            label,
            (x1 + 5, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


    # ======================================
    # TOP STATUS
    # ======================================

    if current_meal:

        meal_text = (
            f"CURRENT MEAL: "
            f"{current_meal.upper()}"
        )

    else:

        meal_text = "NO ACTIVE MEAL"


    cv2.putText(
        frame,
        meal_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )


    # ======================================
    # SHOW CAMERA
    # ======================================

    cv2.imshow(
        "Mess Face Recognition",
        frame
    )


    # ======================================
    # QUIT
    # ======================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# ==========================================
# CLEANUP
# ==========================================

camera.release()

cv2.destroyAllWindows()

print("\nCamera stopped.")