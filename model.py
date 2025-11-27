import os
import numpy as np
from typing import Tuple

# Your 9-class labels EXACTLY in correct order
CLASS_NAMES = [
    'actinic keratosis',
    'basal cell carcinoma',
    'dermatofibroma',
    'melanoma',
    'nevus',
    'pigmented benign keratosis',
    'seborrheic keratosis',
    'squamous cell carcinoma',
    'vascular lesion'
]

def load_model():
    import torch
    import torch.nn as nn
    from torchvision import models

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "best_skin_model.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    # Recreate EfficientNet-B0 architecture EXACTLY like training
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(CLASS_NAMES))

    # Load weights
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)

    model.eval()
    return model


def predict(model, image_path: str) -> Tuple[str, float]:
    import torch
    from PIL import Image

    # Load & preprocess
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))

    arr = np.asarray(img).astype("float32") / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    std = np.array([0.229, 0.224, 0.225], dtype="float32")

    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)   # HWC → CHW
    tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, 3, 224, 224)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)

        score, idx = probs.max(dim=1)
        score = float(score.item())
        idx = int(idx.item())

    label = CLASS_NAMES[idx]
    return label, score
