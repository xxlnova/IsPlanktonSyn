# 🌊 IsPlanktonSyn: A Large-Scale Full-Color Synthetic Benchmark for In-Situ Plankton Recognition

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](#) 
[![Dataset Scale](https://img.shields.io/badge/Images-99,141-green)](#dataset) 
[![Categories](https://img.shields.io/badge/Species-117-orange)](#dataset)
[![Field](https://img.shields.io/badge/Field-Ocean_Ecology-blue)](#) 
[![License](https://img.shields.io/badge/License-Academic_Use-lightgrey)](#license)

> **[IsPlanktonSyn: A Large-Scale Full-Color Synthetic Benchmark and Generation Pipeline for In-Situ Plankton Recognition](#)**<br>
> Yunlong Liu, Tao Zhou, Zekai Zhang, Qinghui Chen, Pengfei Zhu, Cong Liu, Dagang Li, Da Chen, Jinglin Zhang*, Jianping Li<br>

This is the official repository for **IsPlanktonSyn**, a large-scale, biologically plausible full-color benchmark designed to break the modality bottleneck in in-situ plankton recognition. 

---

## 📖 Overview

In-situ plankton imaging is critical for marine ecosystem monitoring. To avoid perturbing natural plankton behavior (phototaxis aggregation), modern underwater imaging systems typically employ near-infrared (NIR) illumination. However, this produces strictly **grayscale observations**, discarding crucial species-specific chromatic cues needed for fine-grained visual categorization (FGVC).

**Our Solution:** 
1. **IsColor (Generation Pipeline):** We propose a training-free multimodal diffusion colorization method. Given a single reference color image, IsColor injects biologically plausible color priors into grayscale in-situ shots while rigorously preserving morphological structures.
2. **IsPlanktonSyn (Dataset):** Using IsColor, we constructed a massive synthetic full-color dataset. Extensive experiments demonstrate that models trained on IsPlanktonSyn consistently and significantly outperform their grayscale counterparts across all mainstream CNN and Transformer architectures.

---

## 📊 Dataset Highlights

*   **Massive Scale:** **99,141** high-quality, full-color plankton images.
*   **Broad Taxonomic Coverage:** **117** categories encompassing both phytoplankton and complex zooplankton.
*   **High Visual Fidelity:** Colors are synthesized based on real biological priors without distorting the native morphological features or in-situ backgrounds.
*   **Standardized Splits:** Pre-configured Training/Validation/Testing splits to ensure fair and reproducible benchmark evaluations.
*   **Proven Effectiveness:** Significantly boosts the fine-grained recognition performance of downstream tasks and supports synthetic-to-real real-world deployment.

---

## 📥 Download IsPlanktonSyn

The complete dataset, including full labeled images, category split files, and metadata, is currently hosted on Baidu Netdisk.

*   **Platform:** Baidu Netdisk (百度网盘)
*   **Link:** [https://pan.baidu.com/s/1SzQtxgz5eaSizKLonr6xEA](https://pan.baidu.com/s/1SzQtxgz5eaSizKLonr6xEA)
*   **Extraction Code:** `c6np`

> **Note:** If you are accessing the dataset from outside of China and encounter issues with Baidu Netdisk, please open an [Issue](#) and we will provide alternative download links (e.g., Google Drive / HuggingFace) in the future.

---

## 🚀 Getting Started

### 1. Data Preparation
After downloading and extracting the dataset, the folder structure is organized intuitively by plankton species ID for seamless integration with PyTorch's `ImageFolder` or custom dataloaders.

```text
IsPlanktonSyn/
├── train/
│   ├── class_001/
│   ├── class_002/
│   └── ...
├── val/
├── test/
└── meta_data.json  # Label mapping and extra metadata
