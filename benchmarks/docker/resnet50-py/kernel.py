# kernel function for resnet50
# import os
import sys
import torch
import json
import base64
from PIL import Image
from torchvision import transforms
from timeit import default_timer as timer


def resnet50py(payload: str):
    # only handling input preprocessing, and call the kernal function
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights='ResNet50_Weights.DEFAULT')
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext101_32x8d', pretrained=True)
    # model = torch.load('resnext50_32x4d.pt')

    # before = timer()
    model.eval()
    model.to('cuda')
    torch.cuda.get_device_properties('cuda')
    # after = timer()

    start = timer()
    
    payload_obj = json.loads(payload)
    print("after parsing", type(base64.b64decode(payload_obj["image_data"])))
    
    input_image = Image.open(base64.b64decode(payload_obj["image_data"]))
    
    preprocess = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    input_batch = input_batch.to('cuda')


    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_catid = torch.topk(probabilities, 1)
    top_catid = top_catid[0].item()
    top_prob = top_prob[0].item()

    end = timer()
    print("local finished")
    print(end - start)
