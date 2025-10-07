from tqdm import tqdm
import json
import numpy as np
import cv2
import os
from segment_anything import sam_model_registry, SamPredictor


coco_json_path = "/annotations/instances_train2019.json"
images_dir = "/images/train2019"
output_json_path = "/annotations/instances_train2019_with_sam.json"
sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"


device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)
predictor = SamPredictor(sam)


with open(coco_json_path, 'r') as f:
    coco = json.load(f)


image_id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}


annotations = coco["annotations"]
annotations_to_process = [ann for ann in annotations if ann["segmentation"] == [[]]]
total = len(annotations_to_process)

print(f"Total annotations: {total}")


for i, ann in enumerate(tqdm(annotations_to_process, total=total, desc="Segmenting with SAM", unit="ann")):
    bbox = ann["bbox"]
    x, y, w, h = bbox
    input_box = np.array([x, y, x + w, y + h])

    image_id = ann["image_id"]
    filename = image_id_to_filename[image_id]
    image_path = os.path.join(images_dir, filename)

    image = cv2.imread(image_path)
    if image is None:
        print(f"Not found: {image_path}")
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)


    masks, scores, _ = predictor.predict(box=input_box, multimask_output=True)
    best_mask = masks[np.argmax(scores)]


    contours, _ = cv2.findContours(best_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea).squeeze()
        if contour.ndim == 2:
            segmentation = contour.flatten().tolist()
            ann["segmentation"] = [segmentation]


with open(output_json_path, 'w') as f:
    json.dump(coco, f)

print(f"✅ : {output_json_path}")