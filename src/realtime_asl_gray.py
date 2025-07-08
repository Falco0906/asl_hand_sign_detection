import cv2
import numpy as np
import tensorflow as tf
import os

# Automatically find the project root directory (where the .py file is located)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths that will work on any machine
TRAIN_DIR = os.path.join(ROOT_DIR, "asl_dataset_gray", "train")
VAL_DIR   = os.path.join(ROOT_DIR, "asl_dataset_gray", "val")
TEST_DIR  = os.path.join(ROOT_DIR, "asl_dataset_gray", "test")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "asl_model.h5")

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Input settings
IMG_SIZE = (64, 64)

# Load class names
class_names = sorted(os.listdir(TRAIN_DIR))

# Start webcam
#cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)  # use index 1 and AVFoundation on macOS
# Try camera indices 0 to 3
cap = None
for i in range(3):
    test_cap = cv2.VideoCapture(i)
    if test_cap.isOpened():
        cap = test_cap
        print(f"✅ Using camera index {i}")
        break

if cap is None:
    raise RuntimeError("❌ No working webcam found.")


while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from webcam.")
        break

    frame = cv2.flip(frame, 1)

    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #ROI
    roi = gray[100:300, 100:300]
    roi_resized = cv2.resize(roi, IMG_SIZE)
    roi_normalized = roi_resized / 255.0
    roi_input = np.expand_dims(roi_normalized, axis=(0, -1))  # shape: (1, 64, 64, 1)

    # Predict
    prediction = model.predict(roi_input, verbose=0)
    predicted_class = np.argmax(prediction)
    label = class_names[predicted_class]

    # Display frame
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 0, 255), 2)
    cv2.putText(frame, f"Prediction: {label}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("ASL Detection (Grayscale)", frame)

    # q for quot
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
