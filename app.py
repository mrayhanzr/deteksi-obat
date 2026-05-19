from flask import Flask, render_template, request
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw, ImageFont
import os
import base64
import io

app = Flask(__name__)

# Konfigurasi Client
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="ORrp8hMD5MsuLnNhQ5ew"
)

@app.route("/")
def home():
    # Ubah bagian return di app.py menjadi ini:
    return render_template(
        "index.html", 
        image=f"data:image/jpeg;base64,{encoded_string}", # Kirim sebagai variabel 'image'
        natur=count_nature, 
        neurobion=count_neurobion
    )

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return "Tidak ada file gambar", 400

    file = request.files["image"]
    image_path = "/tmp/test.jpg"
    file.save(image_path)
    
    # Panggil Workflow
    result = client.run_workflow(
        workspace_name="rafflys-workspace",
        workflow_id="detect-count-and-visualize", 
        images={"image": image_path},
        use_cache=True
    )
    
    predictions = result[0]["predictions"]["predictions"]
    
    # Gambar Bounding Box menggunakan PIL
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    count_nature = 0
    count_neurobion = 0

    for obj in predictions:
        class_name = obj["class"]
        conf = obj["confidence"]
        
        # Filter (sama seperti logika lama Anda)
        if conf < 0.85: continue
        
        x, y, w, h = obj["x"], obj["y"], obj["width"], obj["height"]
        if (w * h) > 40000: continue
        
        x1, y1, x2, y2 = x - w/2, y - h/2, x + w/2, y + h/2
        
        color = "green" if class_name == "NATUR-E" else "red"
        if class_name == "NATUR-E": count_nature += 1
        else: count_neurobion += 1

        draw.rectangle([x1, y1, x2, y2], outline=color, width=5)
        draw.text((x1, y1 - 20), f"{class_name} {conf:.2f}", fill=color)

    # Simpan hasil ke memori agar aman dari masalah file system
    output = io.BytesIO()
    img.save(output, format="JPEG")
    encoded_string = base64.b64encode(output.getvalue()).decode('utf-8')

    return render_template(
        "index.html", 
        image_data=encoded_string, 
        natur=count_nature, 
        neurobion=count_neurobion
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))