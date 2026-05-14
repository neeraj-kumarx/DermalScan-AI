import os, cv2, numpy as np, tensorflow as tf
from pathlib import Path
from preprocessing import preprocess_array, CLASSES

MODEL_PATH   = "models/dermalscan_efficientnetb0.h5"
CASCADE_PATH = "cascades/haarcascade_frontalface_default.xml"
CLASS_COLORS = {
    "wrinkles":   (0, 180, 255),
    "dark spots": (60, 20, 220),
    "puffy eyes": (255, 180, 0),
    "clear skin": (50, 200, 50),
}
AGE_RANGES = {
    "wrinkles": "50–70", "dark spots": "40–60",
    "puffy eyes": "30–50", "clear skin": "20–35"
}

_model = None
def load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("Model not found. Run train_model.py first.")
        _model = tf.keras.models.load_model(MODEL_PATH)
        print("[✓] Model loaded")
    return _model

def detect_faces(image):
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    gray    = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    faces   = cascade.detectMultiScale(gray, 1.05, 3, minSize=(50, 50))
    return faces.tolist() if len(faces) else []

def predict_face(model, face_crop):
    probs = model.predict(preprocess_array(face_crop), verbose=0)[0]
    return dict(sorted(
        {cls: round(float(p)*100, 1) for cls, p in zip(CLASSES, probs)}.items(),
        key=lambda x: x[1], reverse=True))

def annotate_image(image, faces, predictions):
    out = image.copy()
    for (x, y, w, h), pred in zip(faces, predictions):
        if not pred: continue
        top_cls  = list(pred.keys())[0]
        top_conf = list(pred.values())[0]
        color    = CLASS_COLORS.get(top_cls, (200,200,200))
        cv2.rectangle(out, (x,y), (x+w, y+h), color, 2)
        label = f"{top_cls.title()}: {top_conf:.1f}% | Age: {AGE_RANGES.get(top_cls,'N/A')}"
        cv2.rectangle(out, (x, y-30), (x + len(label)*9, y), color, -1)
        cv2.putText(out, label, (x+4, y-8),
                    cv2.FONT_HERSHEY_DUPLEX, 0.52, (255,255,255), 1)
    return out

def process_image(image_path):
    image = cv2.imread(image_path)
    model = load_model()
    faces = detect_faces(image)
    if not faces:
        print("[!] No face detected")
        return {"annotated": image, "faces": 0, "predictions": []}
    preds = [predict_face(model, image[y:y+h, x:x+w]) for (x,y,w,h) in faces]
    ann   = annotate_image(image, faces, preds)
    os.makedirs("outputs", exist_ok=True)
    out   = f"outputs/{Path(image_path).stem}_result.jpg"
    cv2.imwrite(out, ann)
    print(f"[✓] Saved → {out}")
    return {"annotated": ann, "faces": len(faces), "predictions": preds}