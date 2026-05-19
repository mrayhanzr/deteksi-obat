from flask import Flask, render_template, request
from inference_sdk import InferenceHTTPClient
import cv2
import os

app = Flask(__name__)

# ============================================
# ROBOFLOW CONNECTION
# ============================================

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="ORrp8hMD5MsuLnNhQ5ew"
)

# ============================================
# HOME PAGE
# ============================================

@app.route("/")
def home():
    return render_template("index.html")

# ============================================
# DETECTION
# ============================================

@app.route("/detect", methods=["POST"])
def detect():

    file = request.files["image"]

    image_path = "test.jpg"

    file.save(image_path)

    frame = cv2.imread(image_path)

    # ============================================
    # RUN WORKFLOW
    # ============================================

    result = client.run_workflow(
        workspace_name="rafflys-workspace",
        workflow_id="detect-count-and-visualize",
        images={
            "image": image_path
        },
        use_cache=True
    )

    predictions = result[0]["predictions"]["predictions"]

    count_nature = 0
    count_neurobion = 0

    # ============================================
    # DETECTION LOOP
    # ============================================

    for obj in predictions:

        class_name = obj["class"]
        conf = obj["confidence"]

        if conf < 0.85:
            continue

        x = int(obj["x"])
        y = int(obj["y"])

        w = int(obj["width"])
        h = int(obj["height"])

        area = w * h

        if area > 40000:
            continue

        # ============================================
        # COUNTER
        # ============================================

        if class_name == "NATUR-E":
            count_nature += 1
            color = (0, 255, 0)

        else:
            count_neurobion += 1
            color = (0, 0, 255)

        # ============================================
        # BOUNDING BOX
        # ============================================

        x1 = int(x - w / 2)
        y1 = int(y - h / 2)

        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        label = f"{class_name} {conf:.2f}"

        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    # ============================================
    # TOTAL TEXT
    # ============================================

    cv2.rectangle(frame, (10, 10), (320, 100), (40, 40, 40), -1)

    cv2.putText(
        frame,
        f"NATUR-E : {count_nature}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"NEUROBION : {count_neurobion}",
        (20, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2
    )

    # ============================================
    # SAVE RESULT
    # ============================================

    hasil_path = "static/hasil.jpg"

    cv2.imwrite(hasil_path, frame)

    return render_template(
        "index.html",
        image=hasil_path,
        natur=count_nature,
        neurobion=count_neurobion
    )

# ============================================
# RUN APP
# ============================================

if __name__ == "__main__":
    app.run(debug=True)