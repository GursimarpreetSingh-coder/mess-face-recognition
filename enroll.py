import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis


# ==========================================
# CONFIGURATION
# ==========================================

DATASET_DIR = "dataset"
EMBEDDINGS_DIR = "embeddings"

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

print("InsightFace loaded!\n")


# ==========================================
# CREATE OUTPUT DIRECTORY
# ==========================================

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)


# ==========================================
# NORMALIZE EMBEDDING
# ==========================================

def normalize(embedding):

    norm = np.linalg.norm(embedding)

    if norm == 0:
        return embedding

    return embedding / norm


# ==========================================
# PROCESS STUDENTS
# ==========================================

student_folders = os.listdir(DATASET_DIR)

total_students = 0
total_images = 0
successful_images = 0
failed_images = 0


for student_id in student_folders:

    student_path = os.path.join(
        DATASET_DIR,
        student_id
    )

    # Ignore files
    if not os.path.isdir(student_path):
        continue

    print("=" * 50)
    print(f"Processing student: {student_id}")
    print("=" * 50)

    student_embeddings = []

    image_files = os.listdir(student_path)

    for image_name in image_files:

        image_path = os.path.join(
            student_path,
            image_name
        )

        # Check image extension
        if not image_name.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        ):
            continue

        total_images += 1

        print(f"Processing: {image_name}")

        # ------------------------------------------
        # Read image
        # ------------------------------------------

        image = cv2.imread(image_path)

        if image is None:

            print("  ❌ Could not read image")
            failed_images += 1
            continue

        # ------------------------------------------
        # Detect faces
        # ------------------------------------------

        try:

            faces = app.get(image)

        except Exception as e:

            print(f"  ❌ Processing error: {e}")
            failed_images += 1
            continue

        # ------------------------------------------
        # Validate face count
        # ------------------------------------------

        if len(faces) == 0:

            print("  ❌ No face detected")
            failed_images += 1
            continue

        if len(faces) > 1:

            print("  ❌ Multiple faces detected")
            failed_images += 1
            continue

        # ------------------------------------------
        # Get embedding
        # ------------------------------------------

        face = faces[0]

        embedding = face.embedding

        embedding = normalize(embedding)

        student_embeddings.append(embedding)

        successful_images += 1

        print(
            f"  ✅ Face detected "
            f"(confidence: {face.det_score:.3f})"
        )

    # ==========================================
    # SAVE STUDENT EMBEDDINGS
    # ==========================================

    if len(student_embeddings) == 0:

        print(
            f"\n❌ No valid images for {student_id}"
        )

        continue

    student_embeddings = np.array(
        student_embeddings
    )

    output_path = os.path.join(
        EMBEDDINGS_DIR,
        f"{student_id}.npy"
    )

    np.save(
        output_path,
        student_embeddings
    )

    total_students += 1

    print(
        f"\n✅ Saved {len(student_embeddings)} "
        f"embeddings"
    )

    print(
        f"   File: {output_path}"
    )


# ==========================================
# FINAL REPORT
# ==========================================

print("\n")
print("=" * 60)
print("ENROLLMENT COMPLETE")
print("=" * 60)

print(f"Students processed : {total_students}")
print(f"Images processed   : {total_images}")
print(f"Successful images  : {successful_images}")
print(f"Failed images      : {failed_images}")

if total_images > 0:

    success_rate = (
        successful_images / total_images
    ) * 100

    print(
        f"Success rate       : {success_rate:.2f}%"
    )

print("=" * 60)