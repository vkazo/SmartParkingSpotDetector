# Smart Parking Spot Detector
**Author:** Kaden Glover  
**Course:** ITAI 1378 – Computer Vision  
**Project Tier:** 2 (Fine-tuning YOLOv8 on a parking dataset)

---

## 🧠 Problem Statement
Finding available parking spaces is a common issue in busy garages. Drivers waste time circling, increasing congestion and emissions.  
This project aims to detect open parking spots automatically using computer vision.

---

## 💡 Solution Overview
Uses YOLOv8 to detect cars in parking lot images and infer empty spaces.  
Outputs a visual map and count of available spots.

---

## ⚙️ Technical Approach
- **Technique:** Object Detection  
- **Model:** YOLOv8 (Ultralytics)  
- **Framework:** PyTorch  
- **Dataset:** [Aerial View Car Detection for YOLOv5](https://www.kaggle.com/datasets/braunge/aerial-view-car-detection-for-yolov5)

---

## 📈 Metrics
| Metric | Target |
|--------|---------|
| Detection Accuracy | ≥ 90% |
| Inference Speed | ≤ 1 sec/frame |

---

## 🧰 Resources
| Resource | Notes |
|-----------|-------|
| Compute | Google Colab (Free GPU) |
| Framework | PyTorch + Ultralytics |
| Dataset | Kaggle |
| Cost | $0 |

---

## 🧾 AI Usage Log
| Tool | Purpose |
|------|----------|
| ChatGPT | Project planning & documentation |
| Ultralytics Docs | Model setup reference |
| Kaggle | Dataset source |
