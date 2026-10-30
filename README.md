#  Classification performance analysis using LiDAR, RADAR, and LiDAR+RADAR fusion approach on the TruckScenes dataset (Technologies for Autonomous Vehicles)

A comprehensive data-driven comparative evaluation and sensor fusion benchmarking pipeline for 3D multi-class object detection and classification on the commercial vehicle dataset **TruckScenes**, developed for the **Technologies for Autonomous Vehicles** course at Politecnico di Torino.

## Table of contents
1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Architecture](#architecture)
   - [Sensor Ingestion & Spatial Transformation](#1-sensor-ingestion--spatial-transformation)
   - [Feature Extraction Pipeline](#2-feature-extraction-pipeline)
   - [Regularization & Data Normalization](#3-regularization--data-normalization)
   - [Classifier Training & Cross-Validation Protocol](#4-classifier-training--cross-validation-protocol)
4. [Specifications](#specifications)
5. [Project structure](#project-structure)
6. [Results](#results)
   - [Cross-Validation Benchmark](#cross-validation-benchmark)
   - [Operational Scaling Analysis](#operational-scaling-analysis)
7. [Usage](#usage)
   - [Dataset Preparation](#dataset-preparation)
   - [Execution](#execution)
   - [Diagnostic Outputs & Visualization](#diagnostic-outputs--visualization)

## Overview

In autonomous vehicle perception architectures, reliable classification of surrounding obstacles across dynamic and adverse operational domains represents a critical design challenge. While LiDAR sensors yield dense point clouds with millimeter-level geometric boundary resolution, their optical wavelength makes them susceptible to environmental scattering (e.g., fog, rain, snow). Conversely, millimeter-wave RADAR systems operate reliably across harsh atmospheric conditions and directly capture relative Doppler target velocities, yet provide considerably sparser point distributions with limited spatial resolution.

This project investigates and benchmarks the object classification performance across three sensing configurations:
* **LiDAR Only:** Surface roughness, point density, and elevation variance descriptors.
* **RADAR Only:** Low-density penetration and spatial spread descriptors.
* **LiDAR + RADAR Fusion:** 27-dimensional early/feature-level fusion combining geometric bounds, modality-specific descriptors, spatial correlation metrics, and detection balance scores.

The end-to-end pipeline is evaluated over the commercial vehicle **TruckScenes** dataset (`v1.0-mini`) using Stratified 5-Fold Cross-Validation and an ensemble Random Forest classifier.

## Requirements

* **Python Version:** 3.8 to 3.11 recommended (tested on Python 3.9+).
* **Required Libraries:**
  * `numpy` (Numerical linear algebra and point cloud manipulation)
  * `pandas` (Tabular metrics structuring and serialization)
  * `scikit-learn` (Random Forest, Stratified K-Fold, scaling, and classification metrics)
  * `open3d` (PCD file parsing and point cloud vector conversion)
  * `truckscenes` (TruckScenes autonomous vehicle benchmark API)
  * `matplotlib` & `seaborn` (ROC curves, confusion matrices, and distribution plotting)
  * `tqdm` (Progress tracking)

Install all project dependencies via `pip`:
```bash
pip install numpy pandas scikit-learn open3d matplotlib seaborn tqdm truckscenes
```

## Architecture

### 1. Sensor Ingestion & Spatial Transformation
* **Multi-Sensor Channel Aggregation:** Concurrently reads Point Cloud Data (`.pcd`) from 6 synchronized LiDAR channels (top, front, left, right) and 6 surround RADAR sensors.
* **Coordinate Discretization & Local Alignment:** Normalizes UTM spatial offsets ($x_0 = 571700\text{ m}$, $y_0 = 5367800\text{ m}$, $z_0 = 594\text{ m}$) into local vehicle reference frames.
* **Bounding Box Spatial Querying:** Isolates returns belonging to individual object annotations using oriented 3D bounding box boundaries:
  $$\mathbf{p} \in \left[ \mathbf{c}_{\text{norm}} - \frac{\mathbf{s}}{2}, \; \mathbf{c}_{\text{norm}} + \frac{\mathbf{s}}{2} \right]$$

### 2. Feature Extraction Pipeline

#### A. Invariant Geometric Features ($9\text{ Dimensions}$)
Independent of sensor returns, extracted directly from object annotations:
* Oriented bounds: $\text{length} = \max(l, w)$, $\text{width} = \min(l, w)$, $\text{height} = h$.
* Scale-invariant ratios: aspect ratio $\frac{l}{w + \epsilon}$, height-to-width ratio $\frac{h}{w + \epsilon}$.
* Logarithmic volume compression: $\ln(l \cdot w \cdot h + 1)$.
* Spatial discretization buckets ($50\text{ m}$ horizontal, $2\text{ m}$ vertical resolution) preventing overfitting to geographical coordinates.

#### B. Modality-Specific Descriptors
* **LiDAR Descriptors ($6\text{ Dimensions}$):** Evaluates point density $\rho = \frac{N_{\text{pts}}}{V_{\text{obj}}}$ (capped at $N_{\max} = 50$), centroid dispersion $\sigma_{\text{dist}}$, elevation variance $\text{Var}(z)$, and normalized surface roughness:
  $$\text{Roughness} = \frac{\sigma_{\text{dist}}}{\mu_{\text{dist}} + \epsilon}$$
* **RADAR Descriptors ($5\text{ Dimensions}$):** Measures electromagnetic penetration capacity $\frac{N_{\text{pts}}}{V_{\text{obj}} + 1}$ and sparse boundary spread metrics.

#### C. Cross-Modal Fusion Features ($7\text{ Dimensions}$)
* **Detection Balance Score:**
  $$\text{Balance} = \frac{\min(r_{\text{LiDAR}}, \, r_{\text{RADAR}})}{\max(r_{\text{LiDAR}}, \, r_{\text{RADAR}}) + \epsilon}$$
* **Inter-Sensor Spatial Correlation:**
  $$\text{Correlation} = \frac{1}{1 + \Vert \mathbf{c}_{\text{LiDAR}} - \mathbf{c}_{\text{RADAR}} \Vert}$$
* **Dimensional Complementarity:** Absolute deviation between modality-specific point spread vectors:
  $$\Delta_{\text{spread}} = \text{mean}(\vert \mathbf{\sigma}_{\text{LiDAR}} - \mathbf{\sigma}_{\text{RADAR}} \vert)$$
* **Fusion Confidence Metric:** Composite score calculated as $\text{Balance} \times \text{Correlation}$.

### 3. Regularization & Data Normalization
* **Gaussian Perturbation:** Injects zero-mean noise $\mathcal{N}(0, \sigma_{\text{noise}}^2)$ with $\sigma_{\text{noise}} = 0.03$ to regularize feature vectors against rigid annotation bounds.
* **$Z$-score Normalization:** Standardizes inputs based strictly on training-fold distributions:
  $$x_{\text{std}} = \frac{x - \mu}{\sigma}$$

### 4. Classifier Training & Cross-Validation Protocol
* **Stratified $k$-Fold Cross-Validation:** $k = 5$ folds with fixed random seed ($42$) ensuring balanced multi-class proportions. Classes with fewer than 5 total samples are filtered out to prevent split leakage.
* **Ensemble Architecture:**
  * *Single-Sensor Setup:* 50 estimators, max depth 10, min samples split 10, min samples leaf 5.
  * *Sensor Fusion Setup:* 100 estimators, max depth 15, min samples split 8, min samples leaf 3 (`max_features='sqrt'`).

## Specifications

| Parameter | Identifier / Symbol | Value | Unit / Description |
| :--- | :---: | :---: | :---: |
| Dataset Version | `DATASET_VERSION` | `'v1.0-mini'` | Commercial TruckScenes release |
| Cross-Validation Folds | `CROSS_VALIDATION_FOLDS` | $5$ | Stratified partitions |
| Deterministic Random Seed | `RANDOM_SEED` | $42$ | Reproducible split state |
| Subsampling Point Threshold | $N_{\max}$ | $50$ | Points / bounding box cap |
| Gaussian Regularization Level | $\sigma_{\text{noise}}$ | $0.03$ | Perturbation standard deviation |
| Minimum Class Support | `valid_classes` | $\ge 5$ | Minimum samples per category |
| LiDAR Feature Dimensionality | $D_{\text{LiDAR}}$ | $15$ | $9\text{ Geom} + 6\text{ LiDAR}$ |
| RADAR Feature Dimensionality | $D_{\text{RADAR}}$ | $14$ | $9\text{ Geom} + 5\text{ RADAR}$ |
| Fusion Feature Dimensionality | $D_{\text{Fusion}}$ | $27$ | $9\text{ Geom} + 6\text{ L} + 5\text{ R} + 7\text{ Fus}$ |
| Single Modality Estimators | $T_{\text{single}}$ | $50$ | Random Forest decision trees |
| Fusion Modality Estimators | $T_{\text{fusion}}$ | $100$ | Random Forest decision trees |

## Project structure

```plaintext
.
├── SCRIPT-Classification_performance_analysis.py   # Core feature extraction, fusion, and evaluation script
├── REPORT-Classification_performance_analysis.pdf  # Comprehensive technical report and experimental paper
└── README-Classification_performance_analysis.md   # Project documentation
```

## Results

### Cross-Validation Benchmark

The quantitative performance across 5 stratified folds on TruckScenes is reported below as $\text{mean} \pm \text{standard deviation}$:

| Sensor Modality | Accuracy | Precision | Recall | F1-Score | Balanced Accuracy | Specificity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LiDAR Only** | $0.980 \pm 0.002$ | $0.979 \pm 0.002$ | $0.980 \pm 0.002$ | $0.979 \pm 0.002$ | $0.894 \pm 0.011$ | $0.999 \pm 0.000$ |
| **RADAR Only** | $0.984 \pm 0.002$ | $0.982 \pm 0.002$ | $0.984 \pm 0.002$ | $0.983 \pm 0.002$ | $0.910 \pm 0.008$ | $0.999 \pm 0.000$ |
| **LiDAR + RADAR Fusion** | $\mathbf{0.991 \pm 0.002}$ | $\mathbf{0.991 \pm 0.002}$ | $\mathbf{0.991 \pm 0.002}$ | $\mathbf{0.991 \pm 0.002}$ | $\mathbf{0.985 \pm 0.005}$ | $\mathbf{0.999 \pm 0.000}$ |

* **Balanced Accuracy Enhancement:** While overall accuracy increases from $98.0\%$ to $99.1\%$, **Balanced Accuracy improves by $+9.1\%$** (from $89.4\%$ up to $98.5\%$). This demonstrates that early sensor fusion resolves minority class misclassifications (e.g., pedestrians, two-wheelers, debris) that are otherwise overlooked due to single-sensor sparsity or occlusion.
* **ROC-AUC Convergence:** The fused model achieves near-unitary Area Under Curve ($AUC \approx 1.00$) across all target classes under One-vs-Rest evaluation.

### Operational Scaling Analysis

In autonomous driving platforms, perception modules operate at high update rates where small percentage variations significantly affect safety:

1. **Perception Inference Frequency:** $f_s = 36\text{ Hz}$ ($36\text{ classifications/second}$).
2. **Error Rate under Fusion ($99.1\% \approx 99\%$ Accuracy):**
   $$\text{Error rate} = (1 - 0.99) \times 36 = 0.36\text{ errors/second}$$
3. **Cumulative Misclassifications per Operating Hour ($3600\text{ s}$):**
   $$\text{Errors/hour}_{\text{Fusion}} = 0.36 \times 3600 = 1296\text{ incorrect detections/hour}$$
4. **Comparison against Single-Sensor Baseline ($98\%$ Accuracy):**
   $$\text{Errors/hour}_{\text{Single}} = (1 - 0.98) \times 36 \times 3600 = 2592\text{ incorrect detections/hour}$$

Multi-sensor fusion **halves the frequency of classification errors**, eliminating over $1290$ perception failures per vehicle operating hour.

## Usage

### Dataset Preparation
1. Download the [TruckScenes Dataset](https://truckscenes.org) (`v1.0-mini`).
2. Extract the archive into your project directory. The root folder should adhere to the following schema:
   ```plaintext
   man-truckscenes/
   ├── samples/                    # Sensor captures (.pcd point clouds and images)
   ├── sweeps/                     # Intermediate sensor sweeps
   └── v1.0-mini/                 # JSON schema metadata tables
   ```
3. Open `SCRIPT-Classification_performance_analysis.py` and set `DATASET_ROOT_PATH` to your local path:
   ```python
   DATASET_ROOT_PATH = r'/path/to/your/man-truckscenes'
   ```

### Execution
Run the full benchmarking and diagnostic script from your terminal:
```bash
python3 SCRIPT-Classification_performance_analysis.py
```

### Diagnostic Outputs & Visualization
Upon completion, the pipeline automatically compiles and exports the following figures to the project root:
* `sensor_comparison_visualization.png`: Top-down bird's-eye view (BEV) point cloud projections comparing LiDAR, RADAR, and combined spatial densities.
* `LIDAR_confusion_matrix.png`, `RADAR_confusion_matrix.png`, `LIDAR_RADAR_confusion_matrix.png`: Absolute count and normalized percentage confusion matrices with readable category mapping.
* `LIDAR_roc_curves.png`, `RADAR_roc_curves.png`, `LIDAR_RADAR_roc_curves.png`: Multi-class ROC curves displaying micro, macro, and per-class AUC performance.
* `performance_variability_scatter_plots.png`: 6-panel metric distribution plots illustrating cross-validation variance across folds and sensor modalities.
