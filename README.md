# AI-Based Abnormal Behavior Detection from Surveillance Cameras

This project builds an AI surveillance pipeline for abnormal event detection from CCTV footage. It combines object detection, pose estimation, action recognition, and a lightweight web dashboard to detect three main risky events:

- Fire
- Fall
- Fighting

The repository includes both the training artifacts (`CodeTrain/`) and the demo application (`CodeApp/app/`) used to run inference on uploaded videos.

## Project Overview

The goal is to support automated safety monitoring in public or indoor spaces by analyzing surveillance videos and raising alerts when dangerous behaviors are detected.

Main features:

- Detect people in video frames
- Track individuals across frames
- Estimate human pose (17 keypoints)
- Classify human actions into `normal`, `fall`, or `fighting`
- Detect fire regions
- Export processed videos with visual alerts
- Serve a simple Flask dashboard for uploading and reviewing results

## Repository Structure

```text
.
|-- CodeApp/
|   `-- app/                       # Flask dashboard and inference pipeline
|-- CodeTrain/
|   |-- TrainSlowFast/            # Action recognition training (SlowFast + Pose)
|   |-- TrainYolo11PoseDetection/ # YOLOv11 person/pose detector training
|   |-- TrainYolo11FireDetection/ # YOLOv11 fire detector training
|   `-- TrainCNNVerification/     # CNN-based fire verification model
|-- checkpointsmodel/             # Exported trained checkpoints
|-- AI_Based_Abnormal_Behavior_Detection_from_Surveillance.pdf
|-- Present_Order.pdf
`-- Slide_Presentation.pdf
```

## Inference Pipeline

The runtime pipeline is implemented mainly in `CodeApp/app/detect_track_action.py`.

### 1. Person Detection

- A custom YOLOv11 model detects people in each frame.
- Invalid detections are filtered using bounding-box rules, skin-color heuristics, and pose validation.

### 2. Tracking

- A lightweight IOU + centroid-based tracker keeps a stable ID for each detected person.
- Each tracked person stores a short temporal buffer of recent frames.

### 3. Pose Estimation

- A YOLOv11 pose model (`yolo11m-pose.pt`) extracts 17 body keypoints.
- Pose features are normalized and passed into the action recognition branch.

### 4. Action Recognition

- The action model uses **SlowFast R101 + Pose Encoder**
- Classes: `normal`, `fall`, `fighting`
- The pipeline applies temporal smoothing, persistence logic, posture-aware rules, and fast fall heuristics

### 5. Fire Detection

- A dedicated YOLOv11 fire detector identifies fire regions.
- An optional CNN verifier (`fire_red_cnn.pth`) helps reduce false positives.

### 6. Alert Output

- The app returns processed video with alert boxes, JSON alert logs, and dashboard statistics

## Models Used

### 1. Person / Abnormal Action Stream

- **Custom YOLOv11m** for person detection
- **YOLOv11m Pose** for skeleton keypoints
- **SlowFast R101 + Pose Encoder** for action classification
- Action classes: `normal`, `fall`, `fighting`

### 2. Fire Stream

- **YOLOv11 Fire Detector** for fire localization
- **MobileNetV3-based CNN** for fire / non-fire verification

## Training Summary

### SlowFast + Pose (Action Recognition)

From `CodeTrain/TrainSlowFast/checkpoints_sf_pose7/training_log.json` and `classification_report_pose.txt`:

- Dataset: `DataInHouse2`
- Classes: `normal`, `fall`, `fighting`
- Best epoch: `14`
- Best validation F1: `0.7304`
- Best validation accuracy: `0.7826`

Classification report on the saved evaluation set:

| Class | Precision | Recall | F1-score |
|------|-----------:|-------:|---------:|
| Normal | 0.8696 | 0.6452 | 0.7407 |
| Fall | 0.6000 | 1.0000 | 0.7500 |
| Fighting | 0.7568 | 0.9032 | 0.8235 |

Overall:

- Accuracy: `0.7846`
- Macro F1: `0.7714`
- Weighted F1: `0.7807`

### YOLOv11 Pose / Person Detection

From `CodeTrain/TrainYolo11PoseDetection/yolov11m_posedetection/results.csv`:

- Best epoch (by `mAP50-95`): `33`
- Precision: `0.7827`
- Recall: `0.7355`
- mAP@50: `0.8211`
- mAP@50:95: `0.6208`

### YOLOv11 Fire Detection

From `CodeTrain/TrainYolo11FireDetection/results.csv`:

- Best epoch (by `mAP50-95`): `72`
- Precision: `0.8306`
- Recall: `0.7705`
- mAP@50: `0.7990`
- mAP@50:95: `0.5151`

## Setup and Run

### Project Preparation

If you download this project from Google Drive, the source code may be provided as compressed files (`.zip` / `.rar`) instead of extracted folders. Before running anything, prepare the project in this order:

### 1. Extract all compressed files

Unzip or extract all source packages completely so the project returns to its normal folder structure.

Typical files in the shared package:

- `CodeApp.zip`
- `CodeTrain.zip`
- `checkpointsmodel.zip`

After extraction, you should have folders such as:

```text
CodeApp/
CodeTrain/
checkpointsmodel/
```

If your download is a `.rar` file, extract it first, then continue extracting any `.zip` files inside until all folders are fully restored.

### 2. Download the dataset

This repository does not include the dataset directly. Use the dataset link you provided separately (for example in `LinkDataSet.zip`, Drive link, or project documentation) to download the required dataset files.

After downloading, extract the dataset and place it in the expected local location for training or experimentation.

Recommended structure:

```text
DataTrainFire/
DataTrainSlowFast/
DataTrainYolo/
```

If you only want to run the demo app, the dataset is not required for inference, but it is required if you want to retrain or reproduce training results.

### 3. Confirm project files are ready

Before moving to the run steps below, make sure:

- `CodeApp/` has been extracted
- `CodeTrain/` has been extracted
- checkpoint files are available after extraction
- the dataset has been downloaded separately if needed
- you are working from the extracted folders, not from inside a compressed archive

### Run the Project

#### Requirements

- Python 3.9+ recommended
- `pip`
- Optional: NVIDIA GPU for faster inference/training

### 1. Go to the app folder

```bash
cd CodeApp/app
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

On Windows:

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Check model files

The application expects these checkpoints inside `CodeApp/app/checkpoints/`:

- `best_model_pose.pth`
- `customyolov11m.pt`
- `best_model_fire.pt`
- `fire_red_cnn.pth`

The repo already contains these files in:

- `CodeApp/app/checkpoints/`

### 6. Run the Flask app

```bash
python app.py
```

### 7. Open the dashboard

Visit:

```text
http://localhost:5000
```

## Demo

After the app is running, users can upload a surveillance video, wait for processing, review alerts on the dashboard, and download the processed output video.

Dashboard screenshot can be added after you upload the image file to the repository.

Recommended image path:

- `docs/images/dashboard-demo.png`

Then add this line back under the `Demo` section:

```md
![AI Vision Dashboard](docs/images/dashboard-demo.png)
```
