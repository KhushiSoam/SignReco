import cv2
import numpy as np
import keras
from cvzone.HandTrackingModule import HandDetector

# ---- Load Model with TFSMLayer ----
model_path = r"C:\Users\HP\Desktop\Sign Language Detection\ModelV3"
model = keras.layers.TFSMLayer(model_path, call_endpoint="serving_default")

# ---- Labels (make sure they match your training classes) ----
labels = ["Thankyou", "Hello", "I love you"]   # <-- update with your real labels

# ---- Hand Detector ----
detector = HandDetector(maxHands=2)
img_size = 224  # most TeachableMachine models are 224x224

# ---- Camera ----
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("✅ Camera and model loaded. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame")
        break

    hands, img = detector.findHands(frame, draw=True, flipType=False)  # detect hands
    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]

        # crop with safety margins
        offset = 20
        y1, y2 = max(0, y-offset), min(frame.shape[0], y+h+offset)
        x1, x2 = max(0, x-offset), min(frame.shape[1], x+w+offset)
        crop = frame[y1:y2, x1:x2]

        if crop.size > 0:
            # preprocess
            img_resized = cv2.resize(crop, (img_size, img_size))
            img_normalized = img_resized.astype("float32") / 255.0
            img_expanded = np.expand_dims(img_normalized, axis=0)

            # ---- Predict ----
            try:
                preds = model(img_expanded, training=False)
                preds = list(preds.values())[0].numpy().squeeze()


                class_id = int(np.argmax(preds))
                confidence = float(np.max(preds))
                label_text = f"{labels[class_id]} ({confidence:.2f})"

                cv2.putText(frame, label_text, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)

            except Exception as e:
                cv2.putText(frame, f"Prediction Error", (50,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                print("❌ Prediction failed:", e)

    cv2.imshow("Sign Detection", frame)

    # ---- Exit ----
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
