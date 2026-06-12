# AI-Powered Football Video Analytics Platform

This repository contains the complete codebase for an end-to-end artificial intelligence platform designed to extract automated tactical and positional data from football match videos. Utilizing state-of-the-art computer vision and deep learning techniques, this project provides a professional-grade analysis toolkit accessible on consumer hardware.

## 📋 Project Overview

The project is structured around three core Jupyter Notebooks, each responsible for a crucial phase of the AI pipeline:

1. **`ball_player_detection.ipynb`**: Training pipeline for object detection.
2. **`train_pitch_keypoint_detector.ipynb`**: Training pipeline for pitch structure recognition.
3. **`footbal_analysis_with_markdown_titles.ipynb`**: The main inference and tactical analysis pipeline.

Together, these components process raw broadcast video to generate tactical insights, 2D radar maps, team space control (Voronoi diagrams), and individual player statistics.

---

## 📂 Notebook Descriptions

### 1. Object Detection (`ball_player_detection.ipynb`)
This notebook handles the training of a custom object detection model to identify the dynamic elements on the pitch.
- **Model Architecture**: YOLOv5xu (extra-large, updated anchor-free variant).
- **Classes Detected**: Players, Goalkeepers, Referees, and the Ball.
- **Process**: Downloads datasets from Roboflow, sets up the training environment with Ultralytics, and fine-tunes the pretrained model.
- **Output**: Generates robust weights for identifying small, fast-moving objects (like the ball) alongside players in various lighting conditions.

### 2. Pitch Keypoint Detection (`train_pitch_keypoint_detector.ipynb`)
This notebook focuses on spatial understanding of the playing field, crucial for projecting camera views into top-down tactical maps.
- **Model Architecture**: YOLOv8x-Pose.
- **Process**: Treats pitch landmarks (corners, penalty area vertices, center circle) as a "pose" estimation task with 32 distinct keypoints.
- **Output**: Produces a model capable of recognizing the pitch structure regardless of the camera's zoom or pan, providing the reference points needed for the homography transformation.

### 3. Comprehensive Analysis Pipeline (`footbal_analysis_with_markdown_titles.ipynb`)
The core of the platform where the trained models converge to analyze match footage and generate actionable insights.
- **Tracking**: Implements the **ByteTrack** algorithm to assign persistent IDs to players across frames, gracefully handling occlusions.
- **Team Classification**: Utilizes an unsupervised learning pipeline (**SiGLIP** embeddings + **UMAP** dimensionality reduction + **K-Means** clustering) to automatically distinguish teams based on jersey colors, without requiring manual labels.
- **Homography Transformation**: Maps player coordinates from the camera's perspective onto a standard 2D pitch diagram using the detected pitch keypoints.
- **Visualizations Generated**:
  - Annotated match video with bounding boxes and team colors.
  - 2D Radar View showing real-time player positions.
  - Space control maps (Voronoi Diagrams) with smooth team color blending.
  - Player movement heatmaps (Density KDE).
- **Quantitative Data**: Extracts per-player distance traveled, velocity metrics, and consecutive spatial coordinates.

---

## 🚀 Getting Started

### Prerequisites
To run the notebooks, ensure your environment meets the following requirements:
- Python 3.10+
- CUDA-enabled GPU (e.g., NVIDIA RTX 4060 or Google Colab Tesla T4)
- Required Python Packages:
  ```bash
  pip install torch torchvision ultralytics supervision opencv-python pandas matplotlib scikit-learn umap-learn transformers sports
  ```

### Execution Order
For a complete from-scratch run, follow this sequence:
1. Run `ball_player_detection.ipynb` to train your detection model and save the weights.
2. Run `train_pitch_keypoint_detector.ipynb` to train the field keypoint model and save the weights.
3. Update the model paths in `footbal_analysis_with_markdown_titles.ipynb` to point to your newly trained weights.
4. Execute `footbal_analysis_with_markdown_titles.ipynb` with your target match video to generate the final analytical outputs.

---

## 📊 Outputs & Capabilities
By executing the analysis pipeline, you will generate:
- `output_annotated.mp4`: The original video overlaid with bounding boxes, tracking IDs, and team indicators.
- `radar_view.mp4`: A 2D top-down map tracking all player movements.
- `voronoi_blend_view.mp4`: A tactical view visualizing pitch control and spatial dominance.
- `player_positions_consecutive_frames.csv`: Raw positional data exported for custom statistical analysis.

## 🛠️ Methodology
This project was developed adhering strictly to the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology, ensuring a robust lifecycle from business understanding (football tactical needs) through data preparation (Roboflow), modeling (YOLO architectures), to evaluation and deployment.
