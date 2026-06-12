import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import torch.nn as nn

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "C:\\Users\\khatr\\Downloads\\HelmVision_Project\\models\\Helmvision_updated.pth"
THRESHOLD = 0.77

class_names = ["Helmet", "No Helmet"]

# -----------------------------
# MODEL SETUP
# -----------------------------
weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=weights)

model.classifier[1] = nn.Linear(1280, 2)

model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# -----------------------------
# TRANSFORM
# -----------------------------
transform = weights.transforms()

# -----------------------------
# UI FUNCTION
# -----------------------------
def draw_ui(frame, text, safe=True):
    color = (0, 200, 0) if safe else (0, 0, 220)

    x, y = 20, 20
    w, h = 520, 80

    overlay = frame.copy()

    # transparent card
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # border
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # top accent bar
    cv2.rectangle(frame, (x, y), (x + w, y + 6), color, -1)

    # text
    cv2.putText(frame,
                text,
                (x + 20, y + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA)

# -----------------------------
# CAMERA
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera not found")
    exit()

print("Webcam started... Press Q to quit")

# -----------------------------
# LOOP
# -----------------------------
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # preprocess
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = transform(img)
    img = img.unsqueeze(0)

    # prediction
    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)[0]

    prob_helmet = probs[0].item()
    prob_nohelmet = probs[1].item()

    # threshold logic (UNCHANGED)
    if prob_helmet > THRESHOLD:
        label = "Helmet Detected"
        safe = True
    else:
        label = "No Helmet Detected"
        safe = False

    # UI display (CLEAN)
    draw_ui(frame, label, safe)

    cv2.imshow("HelmVision AI", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()