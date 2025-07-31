# ASL Hand Gesture Recognition

This project implements an American Sign Language (ASL) hand gesture recognizer using deep learning and computer vision. It supports:

- Training a MobileNetV2-based classifier on 28 ASL classes (A-Z, space, delete)
- Real-time webcam inference with open-set 'nothing' detection (via confidence threshold)
- Data collection from webcam for custom training
- Embedding-based hybrid inference and visualization

## Features
- Uses TensorFlow, OpenCV, and MediaPipe for hand detection and recognition
- No explicit 'nothing' class: low-confidence predictions are treated as 'nothing'
- Dataset and model weights are excluded from version control (see `.gitignore`)

## Usage
1. Install requirements: `pip install -r requirements.txt`
2. Collect data: `python asl_pipeline.py --collect`
3. Train model: `python asl_pipeline.py --train`
4. Run inference: `python asl_pipeline.py --inference`

See the code for more options (triplet embedding, hybrid inference, etc).

## Datasets
- Datasets are not included in this repository. Place your data in the appropriate folders as described in the code and `.gitignore`.

## License
MIT License
