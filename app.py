import os, cv2, time, numpy as np, streamlit as st
from PIL import Image
from datetime import datetime
from face_detection_pipeline import load_model, detect_faces, predict_face, annotate_image
from export_logs import export_csv

st.set_page_config(page_title="DermalScan AI", page_icon="🔬", layout="wide")

st.markdown("""
<style>
  .title { font-size:2.5rem; font-weight:700;
    background:linear-gradient(135deg,#38BDF8,#818CF8,#34D399);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
</style>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🔬 DermalScan")
    st.markdown("**Model:** EfficientNetB0")
    st.markdown("**Classes:**")
    st.markdown("- 🟠 Wrinkles\n- 🔴 Dark Spots\n- 🔵 Puffy Eyes\n- 🟢 Clear Skin")
    st.caption("Infosys Springboard 6.0 · Batch 13")

st.markdown('<p class="title">DermalScan AI</p>', unsafe_allow_html=True)
st.markdown("*AI-powered facial skin aging detection*")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("Upload Image")
    uploaded = st.file_uploader("Choose a face image", type=["jpg","jpeg","png"])
    if uploaded:
        img_pil = Image.open(uploaded).convert("RGB")
        st.image(img_pil, use_column_width=True)
        analyze = st.button("🔍 Analyse", use_container_width=True)
    else:
        analyze = False

with col2:
    st.subheader("Results")
    if uploaded and analyze:
        with st.spinner("Analysing..."):
            start   = time.time()
            img_np  = np.array(img_pil)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            try:
                model = load_model()
            except FileNotFoundError:
                st.error("Model not found! Run train_model.py first.")
                st.stop()
            faces   = detect_faces(img_bgr)
            elapsed = time.time() - start

            if not faces:
                # Assume whole image is a face crop
                preds = [predict_face(model, img_bgr)]
                ann = annotate_image(img_bgr, [(0, 0, img_bgr.shape[1], img_bgr.shape[0])], preds)
            else:
                preds = [predict_face(model, img_bgr[y:y+h, x:x+w]) for (x,y,w,h) in faces]
                ann = annotate_image(img_bgr, faces, preds)
            
            ann_rgb = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
            st.image(ann_rgb, use_column_width=True)
            st.success(f"✅ Done in {elapsed:.2f}s | Faces: {len(faces) if faces else 1}")

            for i, pred in enumerate(preds, 1):
                st.markdown(f"**Face {i} Predictions:**")
                for cls, pct in pred.items():
                    st.markdown(f"- {cls.title()}: **{pct}%**")
                    st.progress(int(pct))

            st.markdown("---")
            c1, c2 = st.columns(2)
            _, img_bytes = cv2.imencode(".jpg", ann)
            c1.download_button("⬇ Download Image", img_bytes.tobytes(),
                               f"dermalscan_{datetime.now().strftime('%H%M%S')}.jpg",
                               mime="image/jpeg", use_container_width=True)
            csv_str = export_csv(faces, preds, uploaded.name, return_string=True)
            c2.download_button("⬇ Download CSV", csv_str,
                               f"results_{datetime.now().strftime('%H%M%S')}.csv",
                               mime="text/csv", use_container_width=True)
    elif not uploaded:
        st.info("Upload an image to begin.")
        