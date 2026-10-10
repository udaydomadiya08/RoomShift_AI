# RoomShift AI — Change Detection Methodology

## Overview

RoomShift AI is designed for 3D spatial change detection in indoor robotics environments. Given two monocular video recordings of a scene captured at different times, the system determines what physical changes occurred in the environment while rejecting false positives caused by camera trajectory differences, perspective projection, lighting variations, and occlusions.

```
+-----------------------------------------------------------------------+
|                         RoomShift AI Pipeline                         |
+-----------------------------------------------------------------------+
      Video A (Before)                         Video B (After)
             │                                        │
             ▼                                        ▼
    Frame Sampling & Blur Filter            Frame Sampling & Blur Filter
             │                                        │
             ▼                                        ▼
    Feature Extraction (SIFT)               Feature Extraction (SIFT)
             │                                        │
             ▼                                        ▼
    Sparse SfM + Poses                      Sparse SfM + Poses
             │                                        │
             ▼                                        ▼
    Monocular Depth (MiDaS)                 Monocular Depth (MiDaS)
             │                                        │
             ▼                                        ▼
    Point Cloud Reconstruction              Point Cloud Reconstruction
             │                                        │
             ▼                                        ▼
    Object Detection (YOLOv8)               Object Detection (YOLOv8)
             │                                        │
             ▼                                        ▼
    Multi-frame Track Aggregation           Multi-frame Track Aggregation
             │                                        │
             └──────────────────┬─────────────────────┘
                                │
                                ▼
                       Scene Registration
                    (Feature RANSAC + ICP)
                                │
                                ▼
                    Change Detection Classifier
       ┌────────────────────────┼────────────────────────┐
       ▼                        ▼                        ▼
  MOVED Object             ADDED Object             REMOVED Object
  (Displacement > delta)   (Observed empty before)  (Observed empty after)
                                │
                                ▼
                         GEOMETRIC CHANGE
                         (Dense KD-tree delta)
                                │
                                ▼
                         UNCERTAIN Change
                         (Low confidence / occluded)
```

---

## 1. Scene Representation and Registration

### 1.1 Sparse to Dense Reconstruction
1. **Pose Recovery**: Successive frames are matched using scale-invariant feature transforms (SIFT) with Lowe's ratio test. Essential matrices ($E$) are estimated via 5-point RANSAC to recover camera rotations $R_i$ and translations $t_i$.
2. **Dense Depth**: Monocular depth maps $D(u,v)$ are estimated per sampled frame using MiDaS.
3. **Backprojection**: 3D points are backprojected using estimated intrinsics:
   $$X = \frac{u - c_x}{f_x} \cdot Z, \quad Y = \frac{v - c_y}{f_y} \cdot Z, \quad Z = D(u, v)$$
4. **Voxel Filtering**: Point clouds are downsampled onto a voxel grid to ensure uniform spatial density and remove high-frequency noise.

### 1.2 Cross-Recording Alignment
- Point clouds from Recording 1 ($S_1$) and Recording 2 ($S_2$) exist in independent coordinate systems with arbitrary global scale.
- We perform feature matching across the two keyframe sets, backprojecting 2D feature correspondences into 3D points.
- Rigid transformation $(R, t)$ and relative scale $s$ are recovered using Horn's method / Umeyama RANSAC, followed by point-to-plane Iterative Closest Point (ICP) refinement:
  $$S_2' = s R S_2 + t$$
- Registration confidence is quantified via ICP inlier ratio, fitness, and root mean square error (RMSE).

---

## 2. Object Tracking and 3D Grounding

### 2.1 2D Object Detection and Tracking
- Objects are detected per frame using YOLOv8 nano / small.
- Consecutive detections are associated into multi-frame tracks using Intersection over Union (IoU) matching and Kalman/centroid tracking.

### 2.2 3D Centroid Estimation
- For each bounding box $[x_1, y_1, x_2, y_2]$, 3D points within the central region of the box are sampled from the backprojected depth map.
- The object centroid is computed as the median of the inlier 3D points in camera coordinates and transformed to the global world frame:
  $$P_{obj} = \text{median}(\{P_{3D}(u,v) \mid (u,v) \in \text{BBox}_{obj}\})$$

---

## 3. Spatial Change Classification Taxonomy

All detected changes are mapped into one of five rigorous states:

### 3.1 MOVED
- **Condition**: An object $O_A$ in Recording 1 is matched to object $O_B$ in Recording 2 (same semantic class, visual appearance consistency), but:
  $$\| P_{obj, 1} - P_{obj, 2}' \|_2 > \tau_{\text{move}}$$
- **Confidence**: Scaled by alignment confidence and object detection confidence.
- **Reporting**: Old 3D coordinates, new 3D coordinates, and displacement magnitude (in relative reconstruction units).

### 3.2 REMOVED
- **Condition**: Object $O_A$ was observed in Recording 1, but no corresponding object is detected near $P_{obj, 1}$ in Recording 2.
- **Visibility Verification**: The system verifies whether $P_{obj, 1}$ was actively observed in Recording 2 (using camera frustum and point cloud density). If the region was visible and is now empty, it is classified as `REMOVED`.

### 3.3 ADDED
- **Condition**: Object $O_B$ is detected in Recording 2, with no corresponding object in Recording 1.
- **Visibility Verification**: The system verifies that the location $P_{obj, 2}'$ was observed in Recording 1 and was unoccupied.

### 3.4 GEOMETRIC_CHANGE
- **Condition**: Significant point cloud residuals identified via KD-tree distance queries that do not correspond to a tracked semantic object (e.g., structural rearrangements, moved boxes without COCO labels, open/closed doors).
- Clustered via spatial density to prevent spurious single-point alerts.

### 3.5 UNCERTAIN
- **Condition**:
  1. Low registration confidence ($\text{confidence} < \tau_{\text{reliable}}$).
  2. The candidate location was occluded or outside the camera trajectory in one of the recordings.
  3. Insufficient detection confidence or contradictory evidence.
- The system explicitly reports `UNCERTAIN` rather than asserting false positives, crucial for safety-critical humanoid navigation.

---

## 4. Humanoid Robotics Relevance

Humanoid robots navigating human environments (domestic homes, offices, warehouses) require up-to-date spatial topological maps:
- **Obstacle Avoidance**: Identifying newly placed obstacles in previously traversable corridors.
- **Manipulation Planning**: Knowing that a target cup or tool has shifted from counter to table.
- **Map Maintenance**: Pruning static maps without full re-exploration when only a few objects have changed.
