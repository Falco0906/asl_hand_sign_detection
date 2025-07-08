import cv2
import numpy as np
import tensorflow as tf
import os

# Load the model (reverted to .h5)
model = tf.keras.models.load_model("../models/asl_model.h5")

# Define image size
IMG_SIZE = (64, 64)

# Load class names
class_names = sorted(os.listdir("../asl_dataset/asl_alphabet_train"))

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break

    # Region of Interest
    roi = frame[100:300, 100:300]
    roi_resized = cv2.resize(roi, IMG_SIZE)
    roi_normalized = roi_resized / 255.0
    roi_input = np.expand_dims(roi_normalized, axis=0)

    # Predict
    prediction = model.predict(roi_input, verbose=0)
    confidence = np.max(prediction)
    if confidence < 0.6:
        label = "Uncertain"
    else:
        predicted_class = np.argmax(prediction)
        label = class_names[predicted_class]


    # Display
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)
    cv2.putText(frame, f"Prediction: {label}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("ASL Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
