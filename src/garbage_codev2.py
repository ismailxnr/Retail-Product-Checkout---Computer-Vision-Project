import os
import json
import cv2
from tqdm import tqdm


image_dir = "images/synthetic_v1"
json_path = "annotations/instances_synthetic_full_comprassed.json"
output_vis_dir = "visualizations_fast"
os.makedirs(output_vis_dir, exist_ok=True)


with open(json_path, 'r') as f:
    coco = json.load(f)


image_dict = {img["id"]: img for img in coco["synthetic_v1"][:5]}  # İlk 5 görsel
anns_by_image = {}
for ann in coco["annotations"]:
    anns_by_image.setdefault(ann["image_id"], []).append(ann)

category_map = {cat["id"]: cat["name"] for cat in coco["categories"]}


for img_id, img_info in tqdm(image_dict.items(), desc="🔍 Görselleştirme (Hızlı)"):
    file_path = os.path.join(image_dir, img_info["file_name"])
    if not os.path.exists(file_path):
        continue

    img = cv2.imread(file_path)
    if img is None:
        continue

    for ann in anns_by_image.get(img_id, []):
        x, y, w, h = map(int, ann["bbox"])
        cat = category_map.get(ann["category_id"], "Unknown")
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, cat, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    out_path = os.path.join(output_vis_dir, f"vis_{img_info['file_name']}")
    cv2.imwrite(out_path, img)
    del img  # Bellek için

print(f"\n Visiulations ✅: {output_vis_dir}")
