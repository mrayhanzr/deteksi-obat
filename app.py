from flask import Flask, render_template, request
from inference_sdk import InferenceHTTPClient
import cv2
import os

app = Flask(__name__)

# Konfigurasi Client - Langsung masukkan API Key agar tidak ada lagi masalah env
# Catatan: Di Railway, pastikan Anda juga tetap mengisi Variables agar aman
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.environ.get("ROBOFLOW_API_KEY", "ORrp8hMD5MsuLnNhQ5ew")
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    file = request.files["image"]
    image_path = "/tmp/test.jpg"
    file.save(image_path)
    frame = cv2.imread(image_path)

    print(f"DEBUG: Workspace={ 'rafflys-workspace' }, WorkflowID={ 'detect-count-and-visualize' }")

    result = client.run_workflow(
        workspace_name="rafflys-workspace",
        workflow_id="detect-count-and-visualize", 
        images={"image": image_path},
        use_cache=True
    )

    predictions = result[0]["predictions"]["predictions"]

    # --- TAMBAHKAN INI (INISIALISASI VARIABEL) ---
    count_nature = 0
    count_neurobion = 0
    # --------------------------------------------

    for obj in predictions:
        class_name = obj["class"]
        # ... (kode loop Anda) ...
        if class_name == "NATUR-E":
            count_nature += 1
            # ...
        else:
            count_neurobion += 1
            # ...

    # --- SEKARANG VARIABEL SUDAH ADA SAAT DI-RENDER ---
    hasil_path = "static/hasil.jpg"
    cv2.imwrite(hasil_path, frame)

    return render_template(
        "index.html", 
        image=hasil_path, 
        natur=count_nature,    # Sekarang ini akan terbaca
        neurobion=count_neurobion # Sekarang ini akan terbaca
    )