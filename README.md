# American Sign Language Detection

A deep learning-based project for detecting American Sign Language (ASL) hand signs using a CNN model and real-time webcam input. Trained on a grayscale ASL dataset to improve accuracy and generalization.

## Folder Structure

- `asl_dataset_gray/`: Contains `train/` , `val/` and `test/`
- `models/`: Contains trained model (`asl_model.h5`)
- Code for training: (`asl_detection.py`) and webcam detection in src:(`realtime_asl.py`)

## How to Run
```bash
### Clone the Repository
git clone https://github.com/your-username/asl_hand_sign_detection.git
cd asl_hand_sign_detection

### Set up a virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On macOS/Linux

### Install Dependencies
pip install -r requirements.txt

### 1. Train the Model
python asl_detection.py

### 2. Real-time Detection
cd src
python realtime_asl.py
```

Make sure your webcam is on and your hand is clearly visible in the red box.

## Dependencies
See `requirements.txt`

## Notes
Real-time prediction is grayscale and uses a lightweight CNN.

For consistent performance, use good lighting and high-contrast backgrounds.

This version uses CPU by default — GPU support can be added with TensorFlow GPU and CUDA.

## Author
Faisal Khan
GitHub: Falco0906
