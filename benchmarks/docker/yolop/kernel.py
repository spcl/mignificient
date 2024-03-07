import torch
import json
import sys

from timeit import default_timer as timer

def yolop(payload):
    json_obj = json.loads(payload)
    
    # avoid copying the whole nested lists.
    img = torch.as_tensor(json_obj["image_data"])
    print("kernel shape", len(img), len(img[0]), len(img[0][0]), len(img[0][0][0]))
    
    # load model
    model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)
    before = timer()
    model.eval()
    model.to('cuda')
    after = timer()

    start = timer()

    #inference
    img = torch.randn(1,3,640,640).to('cuda')
    det_out, da_seg_out, ll_seg_out = model(img)

    d1 = det_out[0]
    d2 = da_seg_out[0]
    d3 = ll_seg_out[0]

    print(det_out[0], file=sys.stderr)
    print(da_seg_out[0], file=sys.stderr)
    print(ll_seg_out[0], file=sys.stderr)

    end = timer()
    print(end - start)
    
    # further processing of det_out, da_seg_out and ll_seg_out is needed.