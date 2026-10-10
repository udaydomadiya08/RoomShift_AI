# RoomShift AI — Evaluation Report & Empirical Benchmark

## Evaluation Separation Taxonomy

To adhere to rigorous computer vision and robotics benchmarking standards, RoomShift AI maintains a strict three-tier evaluation separation:

```
+-------------------------------------------------------------------------------+
| Tier A: Synthetic Unit & Regression Tests                                      |
| - Verifies algorithmic correctness, matrix math, coordinate conversions,     |
|   IO safety, and pipeline orchestration.                                      |
| - Result: 50 tests passing.                                                   |
+-------------------------------------------------------------------------------+
                                      │
                                      ▼
+-------------------------------------------------------------------------------+
| Tier B: Real-Dataset Spatial-Change Evaluation (3RScan Benchmark)            |
| - Evaluates spatial change classification and displacement quantification      |
|   on real human environments from the official 3RScan dataset.                 |
| - Ground-truth annotations from TUM CAMPAR (Wald et al., ICCV 2019).          |
| - Result: 25 validation scenes evaluated, F1: 0.318, Displacement MAE: 0.84m. |
+-------------------------------------------------------------------------------+
                                      │
                                      ▼
+-------------------------------------------------------------------------------+
| Tier C: End-to-End Real Phone Video Reconstruction                            |
| - Full camera SfM pose recovery, neural depth estimation, and registration    |
|   directly from uncalibrated casual consumer phone recordings.                |
| - Status: Pipeline operational; real-phone validation pending authorized data.|
+-------------------------------------------------------------------------------+
```

*Note: Success in Tier B validates the 3D spatial reasoning engine against genuine human room reconfigurations, but does not substitute for Tier C's optical challenge of monocular reconstruction from casual phone video.*

---

## 1. Real-Dataset Benchmark: 3RScan Validation

We evaluated RoomShift AI's spatial change detection engine on 25 validation scene pairs from 3RScan featuring verified multi-object reconfigurations.

### Benchmark Setup
- **Dataset**: 3RScan (TUM CAMPAR, ICCV 2019)
- **Split**: Validation (47 scenes, 110 pairs)
- **Sample Evaluated**: 25 diverse indoor pairs (living rooms, bedrooms, offices, kitchens)
- **Change Threshold**: $\tau_{\text{moved}} = 0.25\text{ m}$
- **Matching Distance Horizon**: $\tau_{\text{match}} = 2.50\text{ m}$
- **Command to Reproduce**:
  ```bash
  python -m roomshift.cli evaluate-3rscan --data-dir data/external/3rscan --max-scenes 25 --output outputs/3rscan_evaluation.json
  ```

---

## 2. Quantitative Results

### 2.1 Overall Performance

| Metric | Measured Value | Description |
|--------|----------------|-------------|
| **Scenes Evaluated** | 25 | Independent temporal pairs |
| **Ground-Truth Changes** | 167 | Total physical events annotated |
| **True Positives (TP)** | 104 | Correctly classified change events |
| **False Positives (FP)** | 384 | Predicted changes not in ground truth |
| **False Negatives (FN)** | 63 | Missed ground-truth changes |
| **Precision** | **0.213** (21.3%) | Ratio of correct predictions to total |
| **Recall** | **0.623** (62.3%) | Ratio of detected changes to total GT |
| **F1 Score** | **0.318** (31.8%) | Harmonic mean of Precision and Recall |
| **Displacement MAE** | **0.844 m** | Mean Absolute Error on moved objects |

### 2.2 Per-Class Breakdown

| Category | Precision | Recall | F1 Score | TP | FP | FN | Notes |
|:---------|:---------:|:------:|:--------:|:--:|:--:|:--:|:------|
| **MOVED** | **0.491** | **0.520** | **0.505** | 52 | 54 | 48 | Displacement estimation within 84 cm average error across whole rooms |
| **ADDED** | **0.470** | **0.780** | **0.587** | 39 | 44 | 11 | High recall in identifying newly introduced objects |
| **REMOVED** | **0.043** | **0.765** | **0.082** | 13 | 286 | 4 | Low precision due to occluded objects counted as removed |

---

## 3. Defect Analysis & Engineering Fixes

### 3.1 Defect 1: Under-estimation of Indoor Object Move Displacements
- **Symptom**: In the initial evaluation run, `MOVED` recall was only 0.050 (TP=5, FN=95).
- **Root Cause**: `match_objects_across_recordings` hardcoded a maximum matching threshold of 0.5 meters. In 3RScan, the empirical median displacement of moved objects is **1.545 meters**. Objects that moved 1.5–2.0 meters across an office or room were rejected by the matching stage and falsely split into separate `REMOVED` and `ADDED` events.
- **Resolution**: Parameterized `match_distance_threshold` to accommodate real-room spatial scales (2.5 meters). This increased `MOVED` recall tenfold (from 5% to 52%) and boosted overall F1 from 0.197 to 0.318.

### 3.2 Defect 2: High False-Positive Rate on `REMOVED` Objects
- **Symptom**: High FP count on `REMOVED` (286 false positives).
- **Root Cause**: In 3RScan rescans, cameras do not always sweep all corners of the room that were captured in the initial reference scan. An object unobserved in the rescan can be misclassified as removed if the point-cloud visibility query threshold is too loose.
- **Actionable Robotic Recommendation**: Integrate camera trajectory frustum ray-casting to distinguish between *"actively observed empty space"* and *"unvisited camera blindspots"*.

---

## 4. End-to-End Real Phone Video Status

### Current Capability
RoomShift AI provides an operational end-to-end processing pipeline:
```bash
python -m roomshift.cli compare --before video_before.mp4 --after video_after.mp4 --output outputs/demo
```
The pipeline automatically handles:
1. Video validation and frame decimation.
2. SIFT feature matching and 5-point RANSAC camera pose estimation.
3. MiDaS monocular depth estimation (`midas_small`).
4. Point cloud generation and voxel downsampling.
5. Cross-recording feature matching and point-to-plane ICP registration.
6. Multi-frame YOLOv8 object tracking and centroid estimation.
7. Change detection classification and interactive 3D Plotly report generation.

### Validation Status
- **Synthetic Videos**: Verified (5 test scenarios in `examples/`).
- **Real 3RScan 3D Scenes**: Verified (25 validation scene pairs evaluated).
- **Casual Consumer Phone Videos**: Functional pipeline in place; full real-world validation on paired raw phone videos remains pending authorized access to raw uncompressed video streams or user-recorded room video pairs.
