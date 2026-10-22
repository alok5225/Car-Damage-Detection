from pathlib import Path

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

# Works on Windows, Linux and Mac (no hardcoded backslashes)
MODEL_PATH = Path(__file__).parent / "model" / "saved_model.pth"

# Same order as ImageFolder / dataset.classes (alphabetical folder names)
CLASS_NAMES = ['Front Breakage', 'Front Crushed', 'Front Normal',
               'Rear Breakage', 'Rear Crushed', 'Rear Normal']

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Built once, not on every prediction
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes=6, dropout_rate=0.2):
        super().__init__()
        # weights=None: no 100MB download, real weights come from saved_model.pth
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.model.fc.in_features, num_classes),
        )

    def forward(self, x):
        return self.model(x)


_model = None


def load_model():
    """Load the model once and reuse it for every prediction."""
    global _model
    if _model is None:
        model = CarClassifierResNet(num_classes=len(CLASS_NAMES))
        # map_location lets GPU-trained weights load on a CPU-only machine
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE).eval()
        _model = model
    return _model


def predict(image):
    """Accepts a PIL image, an uploaded file object, or a path.
    Returns (class_name, confidence between 0 and 1)."""
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert("RGB")

    x = TRANSFORM(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(load_model()(x), dim=1)[0]
    idx = int(probs.argmax())
    return CLASS_NAMES[idx], float(probs[idx])