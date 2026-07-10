# IsPlanktonSyn: Large-Scale Full-Color Plankton Recognition Dataset
[![Ocean Vision](https://img.shields.io/badge/Field-Ocean%20Ecology-blue)]
[![Dataset Scale](https://img.shields.io/badge/Images-99141-green)]
[![Categories](https://img.shields.io/badge/Species-117-orange)]
[![License](https://img.shields.io/badge/License-Academic%20Use-lightgrey)]

## Overview
In-situ plankton imaging and accurate species recognition are fundamental to ocean-ecosystem monitoring. However, underwater imaging platforms using red-to-near-infrared illumination only output grayscale images, losing critical color features for classification, which leads to severe abundance bias caused by phototaxis aggregation.

To solve this problem, we propose a training-free multimodal diffusion colorization method and construct **IsPlanktonSyn**, a large-scale full-color plankton dataset. With one single reference color image of the target plankton, our diffusion model can restore high-fidelity color information from grayscale underwater shots.

### Dataset Core Information
- Dataset Name: IsPlanktonSyn
- Total Images: 99,141 high-quality color plankton images
- Species Coverage: 117 categories (zooplankton + phytoplankton)
- Core Value: Compensates the lack of species diversity and rich color annotations in existing plankton datasets; significantly boosts performance of all mainstream recognition algorithms.

This dataset provides high-quality labeled full-color visual data for training plankton recognition models, and builds new data & technical support for underwater image enhancement and intelligent ocean ecological monitoring.

## Dataset Download
The complete dataset is shared via Baidu Netdisk.
> File Name: IsPlanktonsyn
- Link: https://pan.baidu.com/s/1SzQtxgz5eaSizKLonr6xEA
- Extraction Code: `c6np`

### Download Tips
1. Copy the link to your browser and open Baidu Netdisk web/client;
2. Input the extraction code `c6np` to access the compressed dataset package;
3. The archive contains full labeled images, category split files and readme metadata.

## Key Contributions
1. Built the first large-scale full-color plankton dataset covering 117 plankton taxa;
2. Solves the color loss problem of in-situ near-infrared underwater imaging;
3. Validated consistent performance gain of multiple recognition backbones on IsPlanktonSyn;
4. Supports downstream tasks: plankton classification, underwater grayscale image colorization, marine ecological quantitative monitoring.

## Usage Instructions
1. Unzip the downloaded dataset file;
2. The folder structure is organized by plankton species ID with unified label mapping files;
3. Train your classification/colorization model directly on the annotated full-color images;
4. For grayscale colorization experiments, match each color sample with corresponding grayscale input as described in our paper.

## Citation
If you use the IsPlanktonSyn dataset in your research, please cite our work:
