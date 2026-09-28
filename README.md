Fight & Gun Detection System

A computer vision-based surveillance system designed to detect physical fights and weapons from live camera streams and uploaded videos.

The system combines YOLO-based object/pose detection with video processing to analyze frames and identify potentially dangerous activities. It can be used with webcam/CCTV streams as well as pre-recorded video files.

🚀 Features
🎥 Live camera / CCTV stream processing
📹 Uploaded video analysis
🥊 Fight detection
🔫 Gun detection
🧍 Human pose detection
🤖 YOLO-based real-time detection
📊 Frame-by-frame video analysis
⚡ Real-time detection pipeline
🔌 Support for video stream inputs such as RTSP
🖥️ Local server-based execution
🧠 System Overview

The system processes an input video stream frame by frame.

                ┌─────────────────────┐
                │   Video Input       │
                │                     │
                │ • Webcam            │
                │ • CCTV / RTSP       │
                │ • Uploaded Video    │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │   Video Processing  │
                │     & Frame         │
                │     Extraction      │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │    YOLO Detection   │
                │                     │
                │ • Person / Pose     │
                │ • Fight             │
                │ • Weapon / Gun      │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Detection Analysis  │
                │                     │
                │ • Bounding Boxes    │
                │ • Confidence        │
                │ • Pose Information  │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Detection Result    │
                │                     │
                │ Fight / Weapon /    │
                │ Normal Activity     │
                └─────────────────────┘
🛠️ Technologies Used
Python
YOLO
YOLOv8-Pose
Ultralytics
OpenCV
MediaPipe
FastAPI
Uvicorn
NumPy
📁 Project Structure
Fight-Gun-Detection/
│
├── app/
│   ├── ...
│
├── models/
│   ├── fight_model.pt
│   └── gun_model.pt
│
├── utils/
│   └── ...
│
├── uploads/
│   └── ...
│
├── outputs/
│   └── ...
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md

The exact structure may vary depending on the current implementation.

⚙️ Installation
1. Prerequisites

Make sure the following are installed:

Python 3.10+
Git
Anaconda/Miniconda (optional but recommended)
A CUDA-compatible NVIDIA GPU is recommended for faster inference.
2. Clone the Repository
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Fight-Gun-Detection
3. Create a Virtual Environment

Using Conda:

conda create -n fightdet python=3.10.12 pip

Activate the environment:

conda activate fightdet
4. Install Dependencies
pip install -r requirements.txt
▶️ Running the Application

Start the application using:

python main.py

If the project uses Uvicorn/FastAPI directly, run:

uvicorn main:app --reload

The server will normally be available at:

http://127.0.0.1:8000

Open the address in your browser to verify that the server is running.

🎥 Input Sources

The system can process different types of video input.

1. Webcam

A connected webcam can be used as the live video source.

Webcam → Frame Capture → Detection → Result
2. CCTV / RTSP

For CCTV cameras supporting RTSP:

CCTV Camera
     ↓
 RTSP Stream
     ↓
Video Processing
     ↓
YOLO Detection
     ↓
Fight / Gun Detection

Example:

rtsp://username:password@camera-ip:port/stream
3. Uploaded Video

Pre-recorded videos can be processed by providing a local video file.

Supported formats depend on the OpenCV/FFmpeg installation, with common formats including:

.mp4
.avi
.mkv
.mov
🥊 Fight Detection

The fight detection component analyzes video frames to identify activity associated with physical fights.

The detection pipeline can use:

Human detection
Pose estimation
Body keypoints
Spatial relationships between people
Movement/activity information
YOLO-based detection

The objective is to identify potentially violent activity rather than simply detecting the presence of multiple people.

🔫 Gun Detection

A separate YOLO-based detection model is used for weapon detection.

When a weapon is detected, the system can provide:

Bounding box
Detection confidence
Detected class
Frame-level detection information

Example:

Input Frame
     ↓
Gun Detection Model
     ↓
Gun Detected
     ↓
Bounding Box + Confidence

Gun detection should be evaluated using an appropriate test dataset or legally available training/testing material. Detection performance depends heavily on the quality and variety of the trained dataset.

🧍 Pose Detection

The pose component detects human body keypoints to provide information about body position and movement.

Pose information can be useful for analyzing:

Body posture
Human movement
Interaction between people
Motion patterns
Potential fighting activity

YOLO Pose and/or MediaPipe-based processing can be integrated into the overall pipeline depending on the module being used.

🔄 Combined Detection Pipeline

The complete system can operate as a combined surveillance pipeline:

                 VIDEO SOURCE
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
     Live Stream             Uploaded Video
          │                       │
          └───────────┬───────────┘
                      ▼
               Frame Extraction
                      │
                      ▼
              Computer Vision
                  Pipeline
                      │
          ┌───────────┼───────────┐
          │           │           │
          ▼           ▼           ▼
        Fight       Pose        Gun
      Detection   Detection   Detection
          │           │           │
          └───────────┼───────────┘
                      ▼
               Result Analysis
                      │
                      ▼
              Detection Output
📊 Detection Output

Depending on the module, the system can provide information such as:

Class: Gun
Confidence: 0.87
Bounding Box: (x1, y1, x2, y2)

or:

Activity: Fight Detected
Confidence: 0.91

The output can be displayed on the processed video frame or returned through the application interface.

🔧 Configuration

Model paths, confidence thresholds, input sources and other parameters can be configured according to the implementation.

For example:

Model Path
Confidence Threshold
Video Source
RTSP URL
Output Directory

If environment variables are used, create a .env file:

MODEL_PATH=models/fight_model.pt
GUN_MODEL_PATH=models/gun_model.pt
CONFIDENCE_THRESHOLD=0.50
VIDEO_SOURCE=0

Do not commit .env files containing passwords, API keys, CCTV credentials or other private information.

📦 Models

The project uses trained YOLO models for the relevant detection tasks.

Example:

models/
│
├── fight_model.pt
└── gun_model.pt

Large model files may not be included directly in the Git repository.

If the model files are hosted separately, download them and place them inside the models/ directory.

🧪 Testing

The system should be tested using different types of input:

Fight Detection
Normal activity
Multiple people
Walking/running
Physical interaction
Fighting scenarios
Different camera angles
Gun Detection
Different object distances
Different lighting conditions
Partial visibility
Different camera angles
Different backgrounds
Video Sources
Webcam
CCTV/RTSP
Uploaded MP4/video files

Testing with varied conditions helps identify false positives and missed detections.

⚠️ Limitations

Detection performance can be affected by:

Poor lighting
Low-resolution cameras
Occlusion
Camera angle
Motion blur
Small/distant objects
Crowded scenes
Unusual poses
Dataset limitations

A detection result should therefore be treated as an automated alert, not as definitive proof of an incident.

🔐 Security & Privacy

When deploying this system with CCTV or other camera sources:

Protect RTSP credentials.
Do not commit camera passwords to GitHub.
Avoid uploading private surveillance footage.
Store sensitive configuration in environment variables.
Follow applicable privacy and surveillance laws.
🙏 Acknowledgements

This project builds upon open-source computer vision technologies and research, including:

IMSOO Fight Detection Repository
Ultralytics YOLO
YOLO Pose Estimation
OpenCV
MediaPipe

Original references:

IMSOO Fight Detection: https://github.com/imsoo/fight_detection
Ultralytics Pose: https://docs.ultralytics.com/tasks/pose/
📜 License

Add the license applicable to this project.

If this project incorporates code or models from other repositories, review their respective licenses and retain the required attribution and notices.

👨‍💻 Project

Fight & Gun Detection System

A computer vision project for automated detection of potentially violent activity and weapons from video streams.

Developed as part of an AI / Computer Vision project.