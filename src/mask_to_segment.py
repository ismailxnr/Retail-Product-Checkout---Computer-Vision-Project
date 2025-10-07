import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

masks_dir = "masks/train2019"
images_dir = "images/train2019"
output_dir = "segments/train2019"

os.makedirs(output_dir, exist_ok=True)

for mask_filename in tqdm(os.listdir(masks_dir)):
    if not mask_filename.endswith('.png'):
        continue

    
    base_name = os.path.splitext(mask_filename)[0]
    image_filename = base_name + ".jpg"

    mask_path = os.path.join(masks_dir, mask_filename)
    image_path = os.path.join(images_dir, image_filename)

    if not os.path.exists(image_path):
        print(f"Görsel dosyası yok: {image_path}")
        continue

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image is None or mask is None:
        print(f"Skipped {mask_filename} due to load error.")
        continue

    
    if np.count_nonzero(mask) == 0:
        print(f"Boş maske atlandı: {mask_filename}")
        continue

   
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        cropped_image = image[y:y+h, x:x+w]
        cropped_mask = mask[y:y+h, x:x+w]

        alpha = np.where(cropped_mask > 0, 255, 0).astype(np.uint8)
        rgba = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGBA)
        rgba[:, :, 3] = alpha

        out_file = os.path.join(output_dir, f"{base_name}_{i}.png")
        Image.fromarray(rgba).save(out_file)
