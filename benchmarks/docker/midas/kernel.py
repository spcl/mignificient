import cv2
import torch
import json
import base64
import numpy as np

from timeit import default_timer as timer

def midas(payload):
    model_type = "MiDaS_small"
    
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    before = timer()
    midas.eval()
    device = torch.device("cuda")
    midas.to(device)
    after = timer()

    print('model eval time:')
    print(after - before)
    
    start = timer()
    payload_obj = json.loads(payload)
    
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform
        
    payload_obj = json.loads(payload)

    jpg_original = base64.b64decode(payload_obj["image_data"])
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    img = cv2.imdecode(jpg_as_np, flags=1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()
    print("prediction output", output)

    # plt.imshow(output)
    # plt.show()
    # plt.savefig('out.pdf')

    end = timer()
    print(end - start) 