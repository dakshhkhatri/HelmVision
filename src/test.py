# import torch
# import torchvision
# import cv2
# from PIL import Image

# print("Everything works!")
# from torchvision import datasets
# from torchvision.models import MobileNet_V2_Weights

# weights = MobileNet_V2_Weights.DEFAULT

# train_data = datasets.ImageFolder(
#     "dataset/train",
#     transform=weights.transforms()
# )

# print(train_data.class_to_idx)


from PIL import Image
import torch

from torchvision.models import (
    mobilenet_v2,
    MobileNet_V2_Weights
)

model = mobilenet_v2()

model.classifier[1] = torch.nn.Linear(
    1280,
    2
)

model.load_state_dict(
    torch.load(
        "models/Helmvision.saved_model.pth",
        map_location="cpu"
    )
)

model.eval()

weights = MobileNet_V2_Weights.DEFAULT
transform = weights.transforms()

img = Image.open(
    r"C:\Users\khatr\Downloads\HelmVision_Project\dataset\test\helmet\helmet_90.jpg"
).convert("RGB")

img = transform(img)
img = img.unsqueeze(0)

with torch.no_grad():
    output = model(img)

print("Raw output =", output)

probs = torch.softmax(output, dim=1)

print("Helmet:", probs[0][0].item())
print("No Helmet:", probs[0][1].item())
