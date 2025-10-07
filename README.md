### 🏷️ Multi-Class Product Counting & Recognition for Automated Retail Checkout  

This project proposes a deep-learning–based system for multi-class product detection and counting using **YOLOv8** on the **Retail Product Checkout (RPC)** dataset.  
We implemented a synthetic dataset generation pipeline, domain adaptation (lighting, shadow, blur), and evaluated performance with mAP metrics.

---

### 📄 Project Proposal  
You can view or download the full proposal document below:  
> [📘 **BIM496_Computer_Vision_Term_Project_Proposal.pdf**](BIM496_Computer_Vision_Term_Project_Proposal_Multi_Class_Product_Counting___Recognition_for_Automated_Retail_Checkout.pdf)

<details>
<summary>🔍 Click to preview key sections</summary>

- **Introduction** – Motivation and overview  
- **Related Work** – Comparison with existing methods  
- **Dataset Description** – RPC structure and challenges  
- **Methodology** – SAM-based segmentation and domain adaptation  
- **Evaluation & Results** – YOLOv8 performance metrics  
- **Conclusion** – Observations and future work  

</details>

---

### 🧩 Technologies Used
| Component | Description |
|------------|-------------|
| **YOLOv8** | Object detection & segmentation |
| **Segment Anything Model (SAM)** | Instance mask extraction |
| **Python / PyTorch** | Model training |
| **OpenCV, Albumentations** | Augmentation & visualization |
| **COCO-format JSON** | Dataset structure |

---

### 📈 Key Results
| Metric | Value |
|--------|-------:|
| **Precision** | 0.875 |
| **Recall** | 0.674 |
| **mAP@50** | 0.75 |
| **mAP@50:95** | ↑ steady improvement after epoch 15 |

---

> 🧩 *“Even without GAN-based translation, handcrafted domain adaptation yielded +30–40% performance gains on RPC.”*

---

### 👥 Authors
- **İsmail Çınar**  
- **Şulenur Şahan**  

📅 *BIM496 – Eskişehir Technical University, 2025*  
