print("Starting InsightFace test...")

from insightface.app import FaceAnalysis

print("InsightFace imported successfully!")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

print("Model object created!")

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

print("===================================")
print("InsightFace model loaded successfully!")
print("===================================")