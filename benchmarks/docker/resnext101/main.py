import sys
import base64
import json
import kernel

def main(image_path):
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
        
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    json_obj = {
        "image_data": image_base64,
    }
    
    payload = json.dumps(json_obj)
    
    ret = kernel.resnext101(payload)
    print("get ret category id", ret)
    
if __name__ == "__main__":
    image_path = sys.argv[1]
    main(image_path)
    