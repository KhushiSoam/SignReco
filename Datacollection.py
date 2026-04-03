import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import os

# Initialize camera and hand detector
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

detector = HandDetector(maxHands=2)
offset = 20
imgSize = 300

# Folder for saving images
folder = "Images/C"
os.makedirs(folder, exist_ok=True)
counter = 0
frameCount = 0

# Saving mode
saving = False
imagesToSave = 80   # number of images to save
saveDelay = 5       # save every 5th frame
saveCount = 0


def create_white_hand_image(img, hand):
    """Crop the hand, resize it, and place on white background."""
    x, y, w, h = hand['bbox']

    # Clamp coordinates
    y1 = max(0, y - offset)
    y2 = min(img.shape[0], y + h + offset)
    x1 = max(0, x - offset)
    x2 = min(img.shape[1], x + w + offset)

    imgCrop = img[y1:y2, x1:x2]
    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

    if imgCrop.size == 0:
        return imgWhite  # in case crop is invalid

    aspectRatio = h / w
    if aspectRatio > 1:
        k = imgSize / h
        wCal = math.ceil(k * w)
        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
        wGap = math.ceil((imgSize - wCal) / 2)
        imgWhite[:, wGap:wCal + wGap] = imgResize
    else:
        k = imgSize / w
        hCal = math.ceil(k * h)
        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
        hGap = math.ceil((imgSize - hCal) / 2)
        imgWhite[hGap:hCal + hGap, :] = imgResize

    return imgWhite


while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    img = cv2.resize(img, (480, 360))
    frameCount += 1

    hands, img = detector.findHands(img)

    if hands:
        imgWhite = create_white_hand_image(img, hands[0])
        cv2.imshow("ImageWhite", imgWhite)

        # Saving mode: save one image every few frames
        if saving and frameCount % saveDelay == 0:
            saveCount += 1
            filename = os.path.join(folder, f"Image_{counter+saveCount}.jpg")
            cv2.imwrite(filename, imgWhite)
            print(f"Saved image {counter+saveCount}")

            if saveCount >= imagesToSave:
                saving = False
                counter += saveCount
                print(f"Finished saving {imagesToSave} images. Total: {counter}")

    cv2.imshow("Camera", img)

    key = cv2.waitKey(10) & 0xFF
    if key == ord("s") and not saving:
        print(f"Starting saving {imagesToSave} images...")
        saving = True
        saveCount = 0
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


