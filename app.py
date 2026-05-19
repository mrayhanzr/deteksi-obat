from flask import Flask, render_template, request
from inference_sdk import InferenceHTTPClient
import cv2
import os

app = Flask(__name__)

# Konfigurasi Client (Sama seperti localhost Anda)
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
    
    # Railway hanya izinkan simpan di /tmp/
    image_path = "/tmp/test.jpg"
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

    # Lakukan loop deteksi (Logic asli localhost Anda)
    for obj in predictions:
        class_name = obj["class"]
        conf = obj["confidence"]

        # KEMBALIKAN KE SETTINGAN AWAL LOKALHOST
        # (Jika sebelumnya 0.85, silakan turunkan ke 0.5 kalau Natur-E sering hilang)
        if conf < 0.85: continue

        x = int(obj["x"])
        y = int(obj["y"])
        w = int(obj["width"])
        h = int(obj["height"])
        area = w * h

        if area > 40000: continue

        if class_name == "NATUR-E":
            count_nature += 1
            color = (0, 255, 0)
        else:
            count_neurobion += 1
            color = (0, 0, 255)

        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        label = f"{class_name} {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Tambahkan teks total (Logic asli localhost Anda)
    cv2.rectangle(frame, (10, 10), (320, 100), (40, 40, 40), -1)
    cv2.putText(frame, f"NATUR-E : {count_nature}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, f"NEUROBION : {count_neurobion}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # SIMPAN KE /tmp/hasil.jpg
    hasil_path = "/tmp/hasil.jpg"
    cv2.imwrite(hasil_path, frame)

    # KARENA DI RAILWAY FOLDER STATIC TIDAK BISA DITULIS,
    # Kita tetap pakai trik kirim base64 agar browser bisa menampilkan gambarnya
    import base64
    with open(hasil_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    return render_template(
        "index.html",
        image=f"data:image/jpeg;base64,{encoded_string}", # Ini membuat HTML Anda otomatis nampil
        natur=count_nature,
        neurobion=count_neurobion
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))