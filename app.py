import os
import cv2
import numpy as np
import keras
from flask import Flask, Response, send_file, jsonify, request
from cvzone.HandTrackingModule import HandDetector
from googletrans import Translator

translator = Translator()
app = Flask(__name__, template_folder='.')  

# ---- Load model ----
model_path = "ModelV3"
model = keras.layers.TFSMLayer(model_path, call_endpoint="serving_default")
labels = ["Thankyou", "Hello", "I love you"]

detector = HandDetector(maxHands=2)
img_size = 224
latest_prediction = ""  

# Global variable to hold the latest frame bytes sent from the browser
latest_frame = None

@app.route('/')
def index():
    return send_file("index.html")

# This endpoint receives the raw frames from the user's browser camera
@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    global latest_frame
    file = request.files.get('image')
    if file:
        latest_frame = file.read()
    return jsonify({"success": True})

# This feeds the processed video back to the page just like your original code did
def generate_frames():
    global latest_prediction, latest_frame
    while True:
        if latest_frame is None:
            continue
            
        # Convert incoming bytes to OpenCV frame
        file_bytes = np.frombuffer(latest_frame, np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if frame is None:
            continue

        # YOUR ORIGINAL WORKING AI LOGIC HERE
        hands, img_out = detector.findHands(frame, draw=True)
        if hands:
            hand = hands[0] # Properly access the first hand object
            x, y, w, h = hand["bbox"]
            offset = 20
            y1, y2 = max(0, y-offset), min(frame.shape[0], y+h+offset)
            x1, x2 = max(0, x-offset), min(frame.shape[1], x+w+offset)
            crop = frame[y1:y2, x1:x2]

            if crop.size > 0:
                img_resized = cv2.resize(crop, (img_size, img_size))
                img_normalized = img_resized.astype("float32") / 255.0
                img_expanded = np.expand_dims(img_normalized, axis=0)
                try:
                    preds = model(img_expanded, training=False)
                    preds = list(preds.values())[0].numpy().squeeze()
                    class_id = int(np.argmax(preds))
                    confidence = float(np.max(preds))
                    latest_prediction = labels[class_id]

                    cv2.putText(img_out, f"{latest_prediction} ({confidence:.2f})",
                                (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                except:
                    latest_prediction = "Prediction Error"
                    cv2.putText(img_out, "Prediction Error", (50,50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        ret, buffer = cv2.imencode('.jpg', img_out)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_latest_prediction')
def get_latest_prediction():
    return jsonify({"prediction": latest_prediction})

@app.route("/translate", methods=["POST"])
def translate():
    data = request.json
    text = data.get("text")
    target_lang = data.get("lang")
    if not text:
        return jsonify({"translated_text": ""})
    translated = translator.translate(text, dest=target_lang)
    return jsonify({"translated_text": translated.text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)




