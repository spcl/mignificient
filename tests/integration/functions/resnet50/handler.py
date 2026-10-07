import json
import os

import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50

import mignificient

model = None
HERE = os.path.dirname(os.path.realpath(__file__))


def handler(obj):
    global model
    if model is None:  # weights come from $TORCH_HOME (inside the package); no download
        model = resnet50(pretrained=True).eval().to("cuda")
        torch.cuda.get_device_properties("cuda")

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    batch = preprocess(Image.open(os.path.join(HERE, "dog.jpg")).convert("RGB")).unsqueeze(0).to("cuda")
    with torch.no_grad():
        probs = torch.nn.functional.softmax(model(batch)[0], dim=0)
    top_prob, top_catid = torch.topk(probs, 1)

    writer = mignificient.BufferStringWriter(obj.result)
    writer.write(json.dumps({"result": top_catid[0].item(), "probability": top_prob[0].item()}))
    return writer.buffer.size
