import json
import os

train_json_path = "annotations/instances_train2019_with_sam.json"
synthetic_json_path = "annotations/instances_synthetic.json"
output_json_path = "annotations/instances_train2019_with_synthetic.json"

with open(train_json_path, 'r') as f:
    train_data = json.load(f)

with open(synthetic_json_path, 'r') as f:
    synth_data = json.load(f)

combined = {
    "synthetic_v1": [],
    "annotations": [],
    "categories": train_data["categories"]
}

max_image_id = max(img["id"] for img in train_data["synthetic_v1"])
max_ann_id = max(ann["id"] for ann in train_data["annotations"])

image_id_offset = max_image_id + 1
annotation_id_offset = max_ann_id + 1

for img in train_data["synthetic_v1"]:
    combined["synthetic_v1"].append(img)

for ann in train_data["annotations"]:
    combined["annotations"].append(ann)

for img in synth_data["synthetic_v1"]:
    new_img = img.copy()
    new_img["id"] += image_id_offset
    combined["synthetic_v1"].append(new_img)

for ann in synth_data["annotations"]:
    new_ann = ann.copy()
    new_ann["id"] += annotation_id_offset
    new_ann["image_id"] += image_id_offset
    combined["annotations"].append(new_ann)

with open(output_json_path, 'w') as f:
    json.dump(combined, f)
print(f"Combined JSON saved to {output_json_path}")