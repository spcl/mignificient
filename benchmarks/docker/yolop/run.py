import sys
import json
import random

import kernel

def main(image_path):
    mu = 0 # mean
    sigma = 1 # sigma
    img_shape = (1,3,640,640)
    img = []
    def generate_one_channel():
        normal_distribution_list = [random.gauss(mu, sigma) for _ in range(img_shape[2] * img_shape[3])]
        # reshape
        normal_distribution_array = [normal_distribution_list[i:i+img_shape[3]] for i in range(0, len(normal_distribution_list), img_shape[2])]
        return normal_distribution_array
    
    for i in range(img_shape[1]): # 3
        img.append(generate_one_channel())
    
    img = tuple([img]) # 1 * 3 * 640 * 640
    
    json_obj = {
        "image_data": img
    }
    payload = json.dumps(json_obj)
    print("shape", len(img), len(img[0]), len(img[0][0]), len(img[0][0][0]))
    
    kernel.yolop(payload)

if __name__ == "__main__":
    # image_path = sys.argv[1]
    main(None)
