# Presentation Guide for DermalScan AI Milestones

**Project:** DermalScan AI - Facial Skin Aging Detection using EfficientNetB0, OpenCV, and Streamlit  
**Date:** 14 May 2026  
**Presenter:** [Your Name]  

This guide outlines how to present each of the 4 milestones to your mentor. For each milestone, include:  
- **Explanation:** What the milestone covers.  
- **How to Present:** Step-by-step demo in VS Code/terminal/browser.  
- **Key Files/Code to Show:** Highlight relevant parts.  
- **Possible Questions & Answers:** Anticipated queries from mentor.  

Use VS Code for code walkthroughs, terminal for commands, and browser for app demos. Total presentation: ~45-60 minutes. Have the app running at http://localhost:8501 for live demos.

---

## Milestone 1: Dataset Preparation and Class Labeling

### Explanation
This milestone involves organizing raw images into labeled folders, splitting them into train/val/test sets, and preparing for model training. It ensures balanced data with one-hot encoding for 4 classes: wrinkles, dark spots, puffy eyes, clear skin. Images are preprocessed with augmentation (flips, rotations, brightness) to improve model generalization.

### How to Present
1. Open VS Code and show the `DATASET/` folder with subfolders (e.g., `wrinkles/`, `dark spots/`).
2. Explain image counts: "Total ~1200 images, manually labeled by skin condition."
3. Run `python3 dataset_setup.py` to generate `dataset_split/` (if not done).
4. Show `dataset_split/train/`, `val/`, `test/` with ~842/180/181 images.
5. Highlight `preprocessing.py`: Show `ImageDataGenerator` with augmentations and `CLASSES` list.
6. Demo: "Data is split 70/15/15 for training/validation/testing."

**Key Files:** `DATASET/`, `dataset_setup.py`, `preprocessing.py` (lines 15-35).

### Possible Questions & Answers
- **Q: How did you ensure data balance?**  
  A: Checked class distributions manually; augmentation helps with minority classes.
- **Q: Why one-hot encoding?**  
  A: For multi-class classification; TensorFlow/Keras handles it via `categorical` mode.
- **Q: Any data cleaning?**  
  A: Assumed clean; in real-world, would remove blurs/noises.
- **Q: Dataset size?**  
  A: ~1200 images; sufficient for demo, but larger datasets improve accuracy.

---

## Milestone 2: Model Training and Validation

### Explanation
Trains an EfficientNetB0 model with transfer learning: base frozen initially, then fine-tuned. Uses Adam optimizer, callbacks (early stopping, LR reduction, checkpoint). Achieves ~88% test accuracy. Includes GPU acceleration on Mac (tensorflow-metal).

### How to Present
1. Show `train_model.py`: Explain architecture (GlobalAveragePooling, Dense layers, Dropout).
2. Run `python3 train_model.py` (or show logs if trained).
3. Display `logs/training_log.csv`: "Epochs with accuracy/loss; best val_acc: 82.78%."
4. Show `logs/training_curves.png`: "Accuracy/loss plots over epochs."
5. Run test eval: `python3 -c "from preprocessing import build_generators; from face_detection_pipeline import load_model; model=load_model(); _,_,test_gen=build_generators(); loss,acc=model.evaluate(test_gen); print(f'Test Acc: {acc*100:.2f}%')"`
6. Highlight: "Phase 1: Head training; Phase 2: Fine-tuning; GPU sped up from hours to minutes."

**Key Files:** `train_model.py`, `logs/training_log.csv`, `logs/training_curves.png`, `models/dermalscan_efficientnetb0.h5`.

### Possible Questions & Answers
- **Q: Why EfficientNetB0?**  
  A: Lightweight, good accuracy; pre-trained on ImageNet for transfer learning.
- **Q: Training time?**  
  A: ~10-15 mins per phase with GPU; callbacks prevent overfitting.
- **Q: Validation strategy?**  
  A: 15% val set; early stopping on val_accuracy with patience=8.
- **Q: Accuracy?**  
  A: 87.85% test; good for 4 classes; could improve with more data/augmentation.
- **Q: GPU usage?**  
  A: tensorflow-metal enables Metal on Apple M1; logs show GPU activation.

---

## Milestone 3: Face Detection and Prediction Pipeline

### Explanation
Integrates OpenCV Haar cascades for face detection in full images. Predicts skin conditions with probabilities, annotates with colors/age ranges. Handles face crops if no detection (for dataset images).

### How to Present
1. Show `face_detection_pipeline.py`: Explain `detect_faces` (Haar cascade), `predict_face` (model inference), `annotate_image` (bounding boxes, labels).
2. Demo detection: `python3 -c "import cv2; from face_detection_pipeline import detect_faces; img=cv2.imread('path/to/image.jpg'); print(detect_faces(img))"`
3. Show prediction: Run on a sample face crop; display probabilities (e.g., "Puffy Eyes: 45.2%").
4. Explain annotations: Colors for classes, age estimates (e.g., puffy eyes: 30-50).
5. Highlight: "Pipeline works for full images or crops; uses local Haar cascade for reliability."

**Key Files:** `face_detection_pipeline.py`, `cascades/haarcascade_frontalface_default.xml`.

### Possible Questions & Answers
- **Q: Why Haar cascades?**  
  A: Fast, lightweight for real-time detection; OpenCV built-in.
- **Q: Detection accuracy?**  
  A: Works on clear faces; parameters tuned for sensitivity (scaleFactor=1.05, minNeighbors=3).
- **Q: Prediction output?**  
  A: Top class with confidence; sorted probabilities for all classes.
- **Q: Age ranges?**  
  A: Estimated based on conditions (e.g., wrinkles: 50-70); not ML-predicted, rule-based.
- **Q: Handling crops?**  
  A: If no face detected, assumes whole image is face (for dataset compatibility).

---

## Milestone 4: Streamlit Web App and Export Features

### Explanation
Web app for image upload, analysis, visualization, and downloads. Uses Streamlit for UI, integrates detection/prediction. Exports annotated images and CSV logs. Runs locally at http://localhost:8501. Includes export_logs.py (Module 8) for logging predictions to CSV.

### How to Present
1. Open http://localhost:8501 in browser; show UI (upload, analyze button).
2. Upload a face image (e.g., from `DATASET/puffy eyes/`); click "Analyse".
3. Demo results: Annotated image with predictions, age, confidence.
4. Show downloads: Annotated image (PNG) and CSV (predictions log via export_logs.py).
5. Explain code: `app.py` (UI logic, spinner, success messages); `export_logs.py` (CSV export function).
6. Highlight: "Headless mode for demo; integrates all pipelines. Module 8 handles logging for records."

**Key Files:** `app.py`, `export_logs.py`.

### Possible Questions & Answers
- **Q: Why Streamlit?**  
  A: Easy web UI for Python; no frontend skills needed.
- **Q: Deployment?**  
  A: Local for demo; can deploy to Streamlit Cloud or server.
- **Q: Export features (Module 8)?**  
  A: export_logs.py generates CSV with predictions, timestamps, and classes for records/download.
- **Q: Performance?**  
  A: Fast inference (~1-2s); GPU helps.
- **Q: User experience?**  
  A: Simple upload-analyze flow; handles errors (e.g., no model).

---

## Overall Tips for Presentation
- **Flow:** Start with data (Milestone 1), build to model (2), pipeline (3), app (4).
- **Demos:** Use live runs; have backups if issues.
- **Time:** 10-15 mins per milestone.
- **Backup:** If app fails, show code and logs.
- **Questions:** Be ready for technical depth; reference code.

**Final Demo:** Upload an image in the app and show full pipeline in action.

Convert this Markdown to PDF using tools like Pandoc or online converters for submission.