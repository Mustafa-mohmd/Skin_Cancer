# Skin Cancer Lesions Classification using Deep Learning

This repository contains deep learning models developed to detect and classify skin lesions into multiple diagnostic types. By leveraging transfer learning with **MobileNetV2** and combining it with hybrid architectural layers, this project achieves high classification accuracy to aid in early skin cancer detection.

## 🚀 Key Features
- **High Performance:** Attained an optimized classification accuracy of **95.7%** on specific benchmarks (outperforming several standard approaches).
- **Hybrid Architectures:** Integrates **MobileNetV2** feature extraction with custom dense layers (`2dense`) for advanced semantic mapping.
- **Multi-Dataset Versatility:** Includes targeted notebooks for widely recognized dermatological datasets:
  - **HAM10000** (7 classification types)
  - **ISIC 2019** (10,000+ image optimized pipeline)
  - **PAD-UFES-20** (Patient-centric clinical data & images)

## 📁 Repository Structure

```text
├── 10000+HAM+Mobv2+2dense(7type).ipynb   # HAM10000 dataset training pipeline
├── ISIC19+MobV2+2dense+10000.ipynb      # ISIC 2019 dataset training pipeline
├── PAD20+ML+Mobv2+2dense(95_6).ipynb    # PAD-UFES-20 dataset with 95.7% accuracy 
└── README.md                             # Project documentation
