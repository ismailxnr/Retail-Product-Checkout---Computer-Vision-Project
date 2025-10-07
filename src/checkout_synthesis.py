import os
import json
import datetime
from PIL import Image, ImageEnhance
import numpy as np
import random
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from cv2 import findContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE


segment_dir = "segments/train2019"
background_dir = "backgrounds"
output_image_dir = "images/synthetic_v1"
output_json_base = "annotations/instances_synthetic"
original_json_path = "annotations/instances_train2019_with_sam.json"


canvas_size = (2048, 2048)
images_to_generate = 49000
checkpoint_step = 7000
min_objects = 6
max_objects = 14
allow_overlap_probability = 0.25
max_allowed_iou = 0.12
max_workers = 8

os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(os.path.dirname(output_json_base), exist_ok=True)

segment_files = [f for f in os.listdir(segment_dir) if f.endswith('.png')]
background_files = [f for f in os.listdir(background_dir) if f.lower().endswith(('.jpg', '.png'))]
existing_files = set(f for f in os.listdir(output_image_dir) if f.startswith("synthetic_") and f.endswith(".jpg"))


product_to_category = {}
with open(original_json_path, 'r') as f:
    original_data = json.load(f)
    image_id_to_file = {img['id']: img['file_name'] for img in original_data['synthetic_v1']}
    for ann in original_data['annotations']:
        img_file = image_id_to_file.get(ann['image_id'], '')
        product_id = img_file.split('_')[0]
        product_to_category[product_id] = ann['category_id']


last_ckpt = max([int(f.split('_')[-1].replace('.json', '')) for f in os.listdir("annotations") if f.startswith("instances_synthetic_part_")], default=0)
start_idx = last_ckpt


for f in existing_files:
    idx = int(f.split('_')[-1].replace('.jpg', ''))
    if idx < start_idx:
        continue
    os.remove(os.path.join(output_image_dir, f))

print(f"🟢 {start_idx} image already exists. Continuing...")

def compute_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter + 1e-6)

def compute_point_xy(bbox):
    x, y, w, h = bbox
    return [round(x + w / 2.0, 2), round(y + h / 2.0, 2)]

def generate_one_scene(idx):
    file_name = f"synthetic_{idx:05}.jpg"
    img_path = os.path.join(output_image_dir, file_name)
    if os.path.exists(img_path):
        return None

    try:
        bg = Image.open(os.path.join(background_dir, random.choice(background_files))).convert("RGBA").resize(canvas_size)
    except:
        return None

    anns, placed_boxes = [], []
    try:
        selected_segments = random.sample(segment_files, random.randint(min_objects, max_objects))
    except:
        return None

    for seg_file in selected_segments:
        try:
            seg = Image.open(os.path.join(segment_dir, seg_file)).convert("RGBA")
        except:
            continue

        scale = random.uniform(0.8, 1.1)
        seg = seg.resize((max(1, int(seg.width * scale)), max(1, int(seg.height * scale))))
        max_x = canvas_size[0] - seg.width
        max_y = canvas_size[1] - seg.height
        if max_x <= 0 or max_y <= 0:
            continue

        for _ in range(30):
            x, y = random.randint(0, max_x), random.randint(0, max_y)
            box = [x, y, x + seg.width, y + seg.height]
            if max([compute_iou(box, b) for b in placed_boxes], default=0) > max_allowed_iou:
                continue
            if any(not (box[2] <= b[0] or box[0] >= b[2] or box[3] <= b[1] or box[1] >= b[3]) for b in placed_boxes) and random.random() > allow_overlap_probability:
                continue

            placed_boxes.append(box)
            seg = ImageEnhance.Brightness(seg).enhance(random.uniform(0.7, 1.3))
            seg = ImageEnhance.Color(seg).enhance(random.uniform(0.7, 1.3))
            seg = Image.fromarray(cv2.GaussianBlur(np.array(seg), (5, 5), 0))
            bg.paste(seg, (x, y), seg)

            mask_np = (np.array(seg.split()[-1]) > 0).astype(np.uint8)
            if np.count_nonzero(mask_np) == 0:
                continue

            contours, _ = findContours(mask_np, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
            segmentation = []
            for contour in contours:
                contour = contour.squeeze()
                if contour.ndim != 2:
                    continue
                contour[:, 0] += x
                contour[:, 1] += y
                segmentation.append([float(pt) for pt in contour.flatten()])

            y1, x1 = np.where(mask_np)[0].min(), np.where(mask_np)[1].min()
            y2, x2 = np.where(mask_np)[0].max(), np.where(mask_np)[1].max()
            w, h = x2 - x1 + 1, y2 - y1 + 1
            bbox = [float(x + x1), float(y + y1), float(w), float(h)]
            point_xy = compute_point_xy(bbox)

            product_id = seg_file.split('_')[0]
            cat_id = product_to_category.get(product_id, 0)

            anns.append({
                "id": idx * 100 + len(anns) + 1,
                "image_id": idx + 1,
                "category_id": cat_id,
                "bbox": bbox,
                "point_xy": point_xy,
                "segmentation": [segmentation],
                "area": float(w * h),
                "iscrowd": 0
            })
            break

    if not anns:
        return None

    bg.convert("RGB").save(img_path, quality=95)
    return {
        "image": {
            "id": idx + 1,
            "file_name": file_name,
            "width": canvas_size[0],
            "height": canvas_size[1]
        },
        "annotations": anns
    }


for chunk_start in range(start_idx, images_to_generate, checkpoint_step):
    chunk_end = min(chunk_start + checkpoint_step, images_to_generate)
    chunk_images, chunk_annotations, chunk_cats = [], [], set()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(generate_one_scene, i): i for i in range(chunk_start, chunk_end)}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Generating {chunk_start}-{chunk_end}"):
            result = future.result()
            if result:
                chunk_images.append(result["image"])
                chunk_annotations.extend(result["annotations"])
                for ann in result["annotations"]:
                    chunk_cats.add(ann["category_id"])

    coco_chunk = {
        "info": {"description": f"Chunk {chunk_start}-{chunk_end}"},
        "synthetic_v1": chunk_images,
        "annotations": chunk_annotations,
        "categories": [
            {"id": cid, "name": str(cid), "supercategory": "product"} for cid in sorted(chunk_cats)
        ]
    }
    ckpt_path = f"{output_json_base}_part_{chunk_end}.json"
    with open(ckpt_path, 'w') as f:
        json.dump(coco_chunk, f)
    print(f"📂 Checkpoint kaydedildi: {ckpt_path}")


final_images, final_anns, final_cats = [], [], set()
for ckpt_file in sorted([f for f in os.listdir("annotations") if f.startswith("instances_synthetic_part_")]):
    with open(os.path.join("annotations", ckpt_file), 'r') as f:
        data = json.load(f)
        final_images.extend(data["synthetic_v1"])
        final_anns.extend(data["annotations"])
        for cat in data["categories"]:
            final_cats.add(cat["id"])

final_json = {
    "info": {
        "description": "Full Synthetic RPC Dataset",
        "version": "1.0",
        "year": 2025,
        "date_created": datetime.datetime.now().isoformat()
    },
    "synthetic_v1": final_images,
    "annotations": final_anns,
    "categories": [
        {"id": cid, "name": str(cid), "supercategory": "product"} for cid in sorted(final_cats)
    ]
}

with open(f"{output_json_base}_full.json", 'w') as f:
    json.dump(final_json, f)
print(f" Final COCO JSON Saved: {output_json_base}_full.json")