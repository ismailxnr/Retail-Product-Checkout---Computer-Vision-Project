import json
import copy


file1_path = "annotations_w_extra/instances_synthetic_extra_merged_fixed.json"
file2_path = "annotations/instances_train_merged_v4.json"
output_path = "annotations_w_extra/instances_merged_final.json"



def load_and_normalize(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    if "images" not in data and "synthetic_v1" in data:
        data["images"] = data.pop("synthetic_v1")  # normalize et
    elif "images" not in data:
        raise KeyError(f"{json_path} dosyasında 'images' veya 'synthetic_v1' bulunamadı.")

    return data


coco1 = load_and_normalize(file1_path)
coco2 = load_and_normalize(file2_path)


max_img_id = max(img["id"] for img in coco1["images"])
max_ann_id = max(ann["id"] for ann in coco1["annotations"])


img_id_mapping = {}
new_images = []
new_annotations = []

for img in coco2["images"]:
    new_img = copy.deepcopy(img)
    old_id = new_img["id"]
    new_img["id"] = old_id + max_img_id + 1
    img_id_mapping[old_id] = new_img["id"]
    new_images.append(new_img)

for ann in coco2["annotations"]:
    new_ann = copy.deepcopy(ann)
    new_ann["id"] = ann["id"] + max_ann_id + 1
    new_ann["image_id"] = img_id_mapping[ann["image_id"]]
    new_annotations.append(new_ann)


category_map = {cat["id"]: cat for cat in coco1["categories"]}
for cat in coco2["categories"]:
    if cat["id"] not in category_map:
        category_map[cat["id"]] = cat


merged = {
    "info": coco1.get("info", {}),
    "licenses": coco1.get("licenses", []),
    "images": coco1["images"] + new_images,
    "annotations": coco1["annotations"] + new_annotations,
    "categories": list(category_map.values())
}


with open(output_path, 'w') as f:
    json.dump(merged, f)

print(f"✅ COCO files merged: {output_path}")
