from flask import Flask, render_template, request
from inference_sdk import InferenceHTTPClient
import cv2
import os
import base64

app = Flask(__name__)

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="ORrp8hMD5MsuLnNhQ5ew"
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    file = request.files["image"]
    image_path = "/tmp/test.jpg" # Gunakan /tmp agar bisa ditulis
    file.save(image_path)
    frame = cv2.imread(image_path)

    result = client.run_workflow(
        workspace_name="rafflys-workspace",
        workflow_id="detect-count-and-visualize",
        images={"image": image_path},
        use_cache=True
    )

    predictions = result[0]["predictions"]["predictions"]
    count_nature = 0
    count_neurobion = 0

    for obj in predictions:
        class_name = obj["class"]
        conf = obj["confidence"]
        if conf < 0.85: continue
        
        x, y, w, h = int(obj["x"]), int(obj["y"]), int(obj["width"]), int(obj["height"])
        if (w * h) > 40000: continue
        
        x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
        color = (0, 255, 0) if class_name == "NATUR-E" else (0, 0, 255)
        
        if class_name == "NATUR-E": count_nature += 1
        else: count_neurobion += 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Tambah Teks Total ke frame
    cv2.rectangle(frame, (10, 10), (350, 110), (40, 40, 40), -1)
    cv2.putText(frame, f"NATUR-E : {count_nature}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, f"NEUROBION : {count_neurobion}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # KONVERSI KE BASE64 (Agar tidak butuh file static/hasil.jpg)
    _, buffer = cv2.imencode('.jpg', frame)
    img_str = base64.b64encode(buffer).decode('utf-8')
    image_data = f"data:image/jpeg;base64,{img_str}"

    return render_template(
        "index.html",
        image=image_data, # Mengirim data gambar langsung
        natur=count_nature,
        neurobion=count_neurobion
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))