import os, csv, io, cv2
from datetime import datetime
from pathlib import Path

os.makedirs("outputs", exist_ok=True)
os.makedirs("logs",    exist_ok=True)
LOG_FILE = "logs/predictions_log.csv"
HEADERS  = ["timestamp","image","face","x","y","w","h",
            "top_class","top_%","wrinkles_%","dark_spots_%","puffy_eyes_%","clear_skin_%"]

def export_csv(faces, predictions, image_name="image", return_string=False):
    rows = []
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for i, ((x,y,w,h), pred) in enumerate(zip(faces, predictions)):
        rows.append({
            "timestamp": ts, "image": image_name, "face": i+1,
            "x":x, "y":y, "w":w, "h":h,
            "top_class":     list(pred.keys())[0]   if pred else "",
            "top_%":         list(pred.values())[0]  if pred else 0,
            "wrinkles_%":    pred.get("wrinkles",   0),
            "dark_spots_%":  pred.get("dark spots",  0),
            "puffy_eyes_%":  pred.get("puffy eyes",  0),
            "clear_skin_%":  pred.get("clear skin",  0),
        })
    exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADERS)
        if not exists: w.writeheader()
        w.writerows(rows)
    if return_string:
        buf = io.StringIO()
        w2  = csv.DictWriter(buf, fieldnames=HEADERS)
        w2.writeheader(); w2.writerows(rows)
        return buf.getvalue()
    return rows

def export_annotated_image(img_bgr, name="image"):
    path = f"outputs/{Path(name).stem}_{datetime.now().strftime('%H%M%S')}.jpg"
    cv2.imwrite(path, img_bgr)
    return path
