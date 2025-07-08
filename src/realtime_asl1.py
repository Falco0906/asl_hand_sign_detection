import cv2
import numpy as np
import tensorflow as tf
import os

# Load model and class names
model = tf.keras.models.load_model("../models/asl_model.h5")
class_names = sorted(os.listdir("../asl_dataset/asl_alphabet_train"))

IMG_SIZE = (64, 64)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Define and draw region of interest
    x1, y1, x2, y2 = 100, 100, 300, 300
    roi = frame[y1:y2, x1:x2]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Preprocess ROI
    roi_resized = cv2.resize(roi, IMG_SIZE)
    roi_normalized = roi_resized.astype("float32") / 255.0
    roi_input = np.expand_dims(roi_normalized, axis=0)

    # Predict
    predictions = model.predict(roi_input, verbose=0)
    confidence = np.max(predictions)
    predicted_class = np.argmax(predictions)
    label = class_names[predicted_class]

    # Display prediction if confident
    if confidence > 0.7:
        display_text = f"{label} ({confidence:.2f})"
    else:
        display_text = "Uncertain..."

    cv2.putText(frame, f"Prediction: {display_text}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("ASL Detection", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
