import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing


image_dir = "synthetic_extra"
json_path = "annotations/instances_synthetic_extra.json"
output_dir = "synthetic_extra_rendered"
os.makedirs(output_dir, exist_ok=True)


color_temperature_kelvin = 6000
noise_std = 1.0
jpeg_quality = 90
shadow_alpha = 0.2
shadow_blur_size = (21, 21)


def adjust_color_temperature(image, kelvin=6000):
    kelvin_table = {
        6000: (255, 243, 239),
        7000: (245, 243, 255),
        8000: (235, 238, 255),
    }
    r, g, b = kelvin_table.get(kelvin, (255, 243, 239))
    balance = np.array([b, g, r]) / 255.0
    return np.clip(image * balance, 0, 255).astype(np.uint8)


def add_noise_and_compression(image, noise_std=1.0, jpeg_quality=90):
    image = image.astype(np.float32)
    noise = np.random.normal(0, noise_std, image.shape).astype(np.float32)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    _, enc = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    return cv2.imdecode(enc, 1)


def apply_shadow_per_object(image, annotations, alpha=0.2, blur_size=(21, 21)):
    h, w = image.shape[:2]
    shadow_layer = np.zeros((h, w, 3), dtype=np.uint8)

    for ann in annotations:
        x, y, bw, bh = ann['bbox']
        cx = int(x + bw / 2 + bw * 0.2)
        cy = int(y + bh / 2 + bh * 0.2)
        sw = int(bw * 0.6)
        sh = int(bh * 0.15)

        if sw < 1 or sh < 1:
            continue

        tmp = np.zeros_like(image)
        cv2.ellipse(tmp, (cx, cy), (sw // 2, sh // 2), 0, 0, 360, (0, 0, 0), -1)
        tmp = cv2.GaussianBlur(tmp, blur_size, 0)
        shadow_layer = cv2.add(shadow_layer, tmp)

    return cv2.addWeighted(image, 1.0, shadow_layer, alpha, 0)


def process_image(entry):
    image_id, filename, annotations = entry
    input_path = os.path.join(image_dir, filename)
    output_path = os.path.join(output_dir, filename)

    img = cv2.imread(input_path)
    if img is None:
        return

    img = apply_shadow_per_object(img, annotations, alpha=shadow_alpha, blur_size=shadow_blur_size)
    img = adjust_color_temperature(img, kelvin=color_temperature_kelvin)
    img = add_noise_and_compression(img, noise_std=noise_std, jpeg_quality=jpeg_quality)

    cv2.imwrite(output_path, img)

if __name__ == '__main__':
    multiprocessing.freeze_support()


    with open(json_path, 'r') as f:
        coco = json.load(f)

    image_id_to_filename = {img["id"]: img["file_name"] for img in coco["synthetic_v1"]}
    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)


    entries = [(img_id, image_id_to_filename[img_id], anns_by_image.get(img_id, [])) for img_id in image_id_to_filename]


    with ProcessPoolExecutor(max_workers=8) as executor:
        list(tqdm(executor.map(process_image, entries), total=len(entries), desc="İşleniyor"))

