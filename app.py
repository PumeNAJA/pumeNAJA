# app.py
import streamlit as st
from fastai.learner import load_learner
from PIL import Image
import pathlib
import fastai
from fastai.learner import load_learner


learn = load_learner('model.pkl')

st.title("🖼️ ทำนายแมวนะจ๊ะ")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Predict
    pred, pred_idx, probs = learn.predict(img)
    
    st.write(f"### 🏷️ Prediction: `{pred}`")
    st.write(f"### 📈 Confidence: `{probs[pred_idx]:.4f}`")
