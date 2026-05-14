# DermalScan AI

A deep learning system for facial skin aging detection with EfficientNetB0, OpenCV face detection, and Streamlit frontend.

## Project Scope

This repository implements the PDF requirements for:
- Dataset preparation and class labeling
- Image preprocessing, augmentation, and one-hot encoding
- EfficientNetB0-based model training with validation
- Face detection and percentage prediction pipeline
- Streamlit web app for upload, visualization, annotation, and downloads
- Export of annotated images and CSV prediction logs
- Documentation for usage and deployment

## Repository Structure

- `app.py` — Streamlit UI for image upload, face detection, prediction, annotation, and downloads.
- `dataset_setup.py` — Creates `dataset_split/` and splits `DATASET/` into train/val/test folders.
- `preprocessing.py` — Prepares image generators and preprocesses images for model inference.
- `train_model.py` — Builds and trains an EfficientNetB0-based classifier, saves model, and plots metrics.
- `face_detection_pipeline.py` — Detects faces, predicts labels, and annotates images.
- `export_logs.py` — Exports prediction logs to CSV and supports app-side downloads.
- `DATASET/` — Raw labeled image folders for each class.
- `dataset_split/` — Generated train/val/test splits after running `dataset_setup.py`.
- `logs/` — Generated plots and training logs.
- `models/` — Saved model weights after training.

## Setup

Install dependencies (use `pip3` if `pip` is not available):

```bash
pip3 install -r requirements.txt
```

If `pip3` is not installed, install Python 3 first and retry.

## Dataset Preparation

Place your labeled images in the following folders:

- `DATASET/wrinkles`
- `DATASET/dark spots`
- `DATASET/puffy eyes`
- `DATASET/clear skin`

Then create the dataset split and class distribution plot:

```bash
python3 dataset_setup.py
```

This generates:

- `dataset_split/train/`
- `dataset_split/val/`
- `dataset_split/test/`
- `logs/class_distribution.png`

Visualize augmentation examples with:

```bash
python3 preprocessing.py
```

This generates:

- `logs/augmentation_samples.png`

## Training

Train the model with transfer learning and fine-tuning:

```bash
python3 train_model.py
```

This saves the best model to:

- `models/dermalscan_efficientnetb0.h5`

It also writes training logs and plots to `logs/`.

## Running the App

Launch the Streamlit frontend:

```bash
streamlit run app.py
```

You can then:

- Upload a face image
- Detect faces and annotate bounding boxes
- View predicted aging classes and percentages
- Download annotated image and CSV log

## Notes

- Model inference uses Haar Cascades for face detection.
- Predictions are shown as percentages and the top class is annotated with expected age range.
- Logs are saved to `logs/predictions_log.csv`.

## Contact

Use this repository to complete the final delivery and add any additional documentation or presentation materials as needed.
