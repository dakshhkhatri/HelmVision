import streamlit as st
import torch
import cv2
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import torch.nn as nn

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = r"C:\Users\khatr\Downloads\HelmVision_Project\models\Helmvision_updated.pth"
THRESHOLD = 0.77

# -----------------------------
# MODEL LOAD
# -----------------------------
weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=weights)
model.classifier[1] = nn.Linear(1280, 2)

model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

transform = weights.transforms()

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="HelmVision AI", layout="centered")

# -----------------------------
# HEADER (UPGRADED UI ONLY)
# -----------------------------
st.markdown("""
    <div style="
        text-align: center;
        padding: 25px;
        border-radius: 14px;
        background: linear-gradient(135deg, #0f172a, #1e293b);
        color: white;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
    ">
        <h1 style="margin-bottom:5px;">🪖 HelmVision AI</h1>
        <p style="font-size:16px; color:#cbd5e1;">
            AI-powered Road Safety Helmet Detection
        </p>
    </div>
""", unsafe_allow_html=True)

st.write("")

# -----------------------------
# STATUS BAR
# -----------------------------
st.markdown("""
    <div style="
        padding:10px;
        border-radius:10px;
        background-color:#111827;
        color:#22c55e;
        text-align:center;
        margin-bottom:15px;
        font-weight:bold;
    ">
        🟢 SYSTEM STATUS: ACTIVE
    </div>
""", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR INFO
# --------------------------

# -----------------------------
# MODE SELECTION
# -----------------------------
mode = st.radio("Select Input Mode", ["Image Upload", "Webcam"])

# =====================================================
# IMAGE MODE
# =====================================================
if mode == "Image Upload":

    st.markdown("### 📷 Image Analysis")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")
        img = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(img)
            probs = torch.softmax(output, dim=1)[0]

        prob_helmet = probs[0].item()

        if prob_helmet > THRESHOLD:
            label = "Helmet Detected"
            color = "green"
        else:
            label = "No Helmet Detected"
            color = "red"

        st.image(image, use_container_width=True)
        st.markdown(f"<h3 style='color:{color}; text-align:center'>{label}</h3>", unsafe_allow_html=True)

# =====================================================
# WEBCAM MODE
# =====================================================
elif mode == "Webcam":

    st.markdown("### 🎥 Live Camera Feed")

    run = st.checkbox("Start Webcam")

    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    while run:

        ret, frame = cap.read()
        if not ret:
            st.error("Camera not found")
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img)
            probs = torch.softmax(output, dim=1)[0]

        prob_helmet = probs[0].item()

        if prob_helmet > THRESHOLD:
            label = "Helmet Detected"
            color = (0, 255, 0)
        else:
            label = "No Helmet Detected"
            color = (0, 0, 255)

        cv2.putText(frame,
                    label,
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

    cap.release()