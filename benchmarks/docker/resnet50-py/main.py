import sys
import base64
import json
import kernel

def main(image_path):
    # since Image argument 
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
    
    # turn the image bytes into Base64 string
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    json_obj = {
        "image_data": image_base64,
    }
    
    payload = json.dumps(json_obj)
    
    ret = kernel.resnet50py(payload)


if __name__ == "__main__":
    image_path = sys.argv[1]
    main(image_path)
