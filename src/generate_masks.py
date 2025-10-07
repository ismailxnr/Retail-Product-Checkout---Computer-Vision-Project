import os
import json
import numpy as np
import cv2
from tqdm import tqdm
from pycocotools import mask as maskUtils


json_path = "annotations/instances_train2019_with_sam.json"
images_dir = "images/train2019"
output_mask_dir = "masks/train2019"


os.makedirs(output_mask_dir, exist_ok=True)


with open(json_path, 'r') as f:
    coco = json.load(f)


from collections import defaultdict
image_to_anns = defaultdict(list)
for ann in coco['annotations']:
    image_to_anns[ann['image_id']].append(ann)


image_id_to_filename = {img['id']: img['file_name'] for img in coco['synthetic_v1']}


for image_id, anns in tqdm(image_to_anns.items(), desc="Generating masks"):
    
    width = coco['synthetic_v1'][image_id]['width']
    height = coco['synthetic_v1'][image_id]['height']
    
    mask = np.zeros((height, width), dtype=np.uint8)

    for ann in anns:
        category_id = ann['category_id']

        if isinstance(ann['segmentation'], list): 
            rles = maskUtils.frPyObjects(ann['segmentation'], height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(ann['segmentation'], dict):  
            rle = ann['segmentation']
        else:
            continue
        
        m = maskUtils.decode(rle)
        mask[m == 1] = category_id  

    
    out_path = os.path.join(output_mask_dir, f"{image_id_to_filename[image_id].split('.')[0]}.png")
    cv2.imwrite(out_path, mask)
