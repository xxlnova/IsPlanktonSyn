# 🌊 IsPlanktonSyn: A Large-Scale Full-Color Synthetic Benchmark for In-Situ Plankton Recognition

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](#) 
[![Dataset Scale](https://img.shields.io/badge/Images-89,712-green)](#dataset) 
[![Categories](https://img.shields.io/badge/Species-116-orange)](#dataset)
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

*   **Massive Scale:** **89,712** high-quality, full-color plankton images.
*   **Broad Taxonomic Coverage:** **116** categories encompassing 31 phytoplankton and 85 zooplankton taxa.
*   **High Visual Fidelity:** Colors are synthesized based on real biological priors without distorting the native morphological features or in-situ backgrounds.
*   **Standardized Prompts:** Includes a standardized, biologically plausible prompt construction template (powered by Qwen3-VL) to strictly guide the multimodal diffusion process.
*   **Standardized Splits:** Pre-configured Training/Validation/Testing splits (7:1:2 ratio) to ensure fair and reproducible benchmark evaluations.
*   **Proven Effectiveness:** Significantly boosts the fine-grained recognition performance of downstream tasks and supports synthetic-to-real real-world deployment.

---

## 📥 Download IsPlanktonSyn

The complete dataset, including full labeled images, category split files, and metadata, is available on **Kaggle** and **Baidu Netdisk**.

### Option 1: Kaggle (Recommended for Global Users)
*   **Link:** [Kaggle Dataset: IsPlanktonSyn](https://www.kaggle.com/datasets/xlnovax/isplanktonsyn)

### Option 2: Baidu Netdisk (百度网盘)
*   **Link:** [https://pan.baidu.com/s/1SzQtxgz5eaSizKLonr6xEA](https://pan.baidu.com/s/1SzQtxgz5eaSizKLonr6xEA)
*   **Extraction Code:** `c6np`

---

## ⚙️ Environment & Dependencies

To run the **IsColor** generation pipeline from scratch, you will need the following dependencies and hardware setup:

*   **Core Models:** Stable Diffusion v1.5, pre-trained Canny and Tile ControlNets, and IP-Instruct encoder.
*   **Prompt Engine:** Qwen3-VL (for automated species-specific prompt generation).
*   **Hardware:** The pipeline and downstream baselines were evaluated and executed on a single server equipped with an NVIDIA A100 GPU.

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
└── test/
