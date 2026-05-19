from flask import Flask, render_template, request
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw
import os
import uuid

app = Flask(__name__)

# ============================================
# CONFIG
# ============================================

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

# ============================================
# ROBOFLOW CLIENT
# ============================================

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.getenv("ORrp8hMD5MsuLnNhQ5ew")
)

# ============================================
# HOME
# ============================================

@app.route("/")
def home():
    return render_template("index.html")

# ============================================
# DETECT
# ============================================

@app.route("/detect", methods=["POST"])
def detect():

    if "image" not in request.files:
        return "No image uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "Empty filename"

    # simpan file upload
    filename = str(uuid.uuid4()) + ".jpg"

    upload_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(upload_path)

    # ============================================
    # RUN ROBOFLOW
    # ============================================

    result = client.run_workflow(
        workspace_name="rafflys-workspace",
        workflow_id="detect-count-and-visualize-2",
        images={
            "image": upload_path
        },
        use_cache=True
    )

    # ============================================
    # AMBIL PREDICTIONS
    # ============================================

    predictions = result[0]["predictions"]["predictions"]

    # ============================================
    # DRAW BOUNDING BOX
    # ============================================

    image = Image.open(upload_path)
    draw = ImageDraw.Draw(image)

    class_counts = {}

    for pred in predictions:

        x = pred["x"]
        y = pred["y"]
        w = pred["width"]
        h = pred["height"]

        label = pred["class"]
        confidence = pred["confidence"]

        # hitung total kelas
        if label not in class_counts:
            class_counts[label] = 0

        class_counts[label] += 1

        # koordinat bbox
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2

        # gambar kotak
        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=4
        )

        # teks
        draw.text(
            (x1, y1 - 20),
            f"{label} {confidence:.2f}",
            fill="red"
        )

    # simpan hasil
    result_filename = "result_" + filename

    result_path = os.path.join(
        app.config["RESULT_FOLDER"],
        result_filename
    )

    image.save(result_path)

    # ============================================
    # RETURN HTML
    # ============================================

    return render_template(
        "result.html",
        image_path=result_path,
        class_counts=class_counts
    )

# ============================================
# MAIN
# ============================================

if __name__ == '__main__':
    # Railway menetapkan port di sini, atau default ke 5000
    port = int(os.environ.get("PORT", 5000))
    # Sangat penting: gunakan host 0.0.0.0
    app.run(host="0.0.0.0", port=port)