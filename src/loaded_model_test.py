import torch
from torchvision.models import mobilenet_v2

model = mobilenet_v2()

model.classifier[1] = torch.nn.Linear(
    in_features=1280,
    out_features=2
)

model.load_state_dict(
    torch.load(
        "models/Helmvision.saved_model.pth",
        map_location="cpu"
    )
)

model.eval()

print("Model loaded successfully!")