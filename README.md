# Simple Face Recognition Application

A simple face recognition system using Python 3.14 that can detect and recognize faces from webcam input.

## Features

- Real-time face detection from webcam
- Face enrollment system to register known faces
- Face recognition to identify enrolled individuals
- Simple command-line interface

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

> **Note**: If you encounter compatibility issues with Python 3.14, consider using Python 3.11 or 3.12 for better library support.

## Usage

### 1. Enroll a New Face

Run the application in enrollment mode to register a new person:

```bash
python face_recognition_app.py --enroll "Person Name"
```

The application will:
- Open your webcam
- Detect your face
- Capture and save your face data
- Press 'q' to quit after enrollment

### 2. Run Face Recognition

Run the application in recognition mode to identify faces:

```bash
python face_recognition_app.py
```

The application will:
- Open your webcam
- Detect faces in real-time
- Display names of recognized individuals
- Show "Unknown" for unrecognized faces
- Press 'q' to quit

## Directory Structure

```
face recognation/
├── face_recognition_app.py    # Main application
├── requirements.txt            # Dependencies
├── known_faces/               # Enrolled face data
│   ├── Person_Name.jpg        # Face images
│   └── encodings.pkl          # Face encodings
├── hello.py                   # OpenCV test script
└── README.md                  # This file
```

## Troubleshooting

- **Camera not working**: Check if another application is using the webcam
- **Import errors**: Ensure all dependencies are installed: `pip install -r requirements.txt`
- **Python 3.14 issues**: Try using Python 3.11 or 3.12 for better compatibility
- **No faces detected**: Ensure good lighting and face the camera directly
