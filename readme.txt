Synthetic_Retail_Checkout_Project/
│
├── annotations/
├── backgrounds/
├── segments/
├── images/
├── synthetic_extra/
├── results/
├── runs/
│
├── checkout_synthesis.py
├── checkout_synthesis_extra.py
├── generate_masks.py
├── generate_segment.py
├── mask_to_segment.py
├── coco_json_combiner.py
├── train_yolo.py
├── test_yolo.py
├── render_images.py
└── data.yaml
 Note:
This directory structure is required for the scripts to run correctly. Each script assumes relative paths to the annotations, images, segments, and other folders. Modifying the folder layout may lead to file-not-found errors or incorrect data loading.
 

# Project Title: Synthetic Retail Checkout Scene Generator & YOLOv8 Training Pipeline

## 📁 Project Structure Overview:
- `checkout_synthesis.py`        → Generates synthetic checkout images using product segments and backgrounds.
- `generate_masks.py`            → Converts annotations or masks into usable formats.
- `generate_segment.py`          → Prepares instance-level product segments for overlay.
- `mask_to_segment.py`           → Converts raw masks into individual object segments.
- `coco_json_combiner.py`        → Merges multiple COCO-format JSON files into one.
- `train_yolo.py`                → Trains a YOLOv8 model using prepared datasets.
- `test_yolo.py`                 → Evaluates trained YOLOv8 model.
- `render_images.py`             → Manual Domain Adaptation
- `data.yaml`                    → Configuration file for YOLOv8 (defines dataset paths and class names).
- `requirements.txt`             → Python dependencies.

## ▶️ How to Run the Project

### 1. 🧪 Setup Environment
Make sure Python 3.8+ is installed. Then create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

2)Install Dependencies

pip install -r requirements.txt

Note: You may need to install ultralytics for YOLOv8:
pip install ultralytics

3)Generate Product Segments

python generate_segment.py

4)Generate Synthetic Checkout Images

python checkout_synthesis.py

5)Generate Object Masks 

python generate_masks.py

6)Merge COCO Annotations (if needed)

python coco_json_combiner.py

7)Train the YOLOv8 Model

python train_yolo.py

8)Evaluate the Trained Model

python test_yolo.py

#Notes
All generated outputs (images, models, logs) are stored in runs/ or defined output directories.

Check and update data.yaml to match your dataset structure.

Garbage codes were used from time to time in areas such as name changes and visualization of photographs during the course of the project.




