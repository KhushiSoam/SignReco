from flask import Flask, Response, send_file, jsonify, request
import cv2
import numpy as np
import keras
from cvzone.HandTrackingModule import HandDetector
from googletrans import Translator
translator = Translator()

app = Flask(__name__, template_folder='.')  # index.html is in the same folder

# ---- Load model ----
model_path = "ModelV3"
model = keras.layers.TFSMLayer(model_path, call_endpoint="serving_default")
labels = ["Thankyou", "Hello", "I love you"]

detector = HandDetector(maxHands=2)
img_size = 224

# ---- Video capture ----
cap = cv2.VideoCapture(0)
latest_prediction = ""  # Store last prediction for TTS

def generate_frames():
    global latest_prediction
    while True:
        success, frame = cap.read()
        if not success:
            break

        hands, img_out = detector.findHands(frame, draw=True)
        if hands:
            hand = hands[0]
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

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', img_out)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ---- Routes ----
@app.route('/')
def index():
    return send_file("index.html")  # index.html in same folder

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
    app.run(debug=True)




