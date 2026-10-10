# RoomShift AI — Real Public Dataset Investigation & Analysis

## Executive Summary

To validate RoomShift AI beyond synthetic test scenarios, we investigated three prominent indoor 3D datasets:
1. **3RScan** (Wald et al., ICCV 2019 / Technical University of Munich - CAMPAR)
2. **ScanNet** (Dai et al., CVPR 2017 / Stanford, Princeton, TUM)
3. **ScanNet++** (Yeshwanth et al., ICCV 2023 / TUM)

We prioritized **3RScan** because it specifically captures *naturally changing indoor environments* through repeated scans of identical scenes across temporal intervals, with explicit 3D ground-truth object transformations and change annotations.

---

## 1. Dataset Investigation & Terms of Use

| Dataset | Official URL | Primary Focus | Licensing / Download Terms | Local Status |
|---------|--------------|---------------|----------------------------|--------------|
| **3RScan** | [waldjohannau.github.io/3RScan](https://waldjohannau.github.io/3RScan/) | Temporal changes, rescans, object re-localization | Academic Terms of Use agreement form; approval via `3RScan@googlegroups.com`. Public metadata accessible on TUM server. | **INTEGRATED & EVALUATED** (Master metadata `3RScan.json`, `objects.json`, `relationships.json` downloaded: 36.8 MB total) |
| **ScanNet** | [scan-net.org](http://www.scan-net.org/) | Static RGB-D indoor scene reconstruction & semantic segmentation | Non-commercial research agreement form; manual approval via `scannet@googlegroups.com`. No direct public download. | Inspected. Requires institutional email and formal signed agreement. |
| **ScanNet++** | [scannetpp.mlsg.cit.tum.de](https://scannetpp.mlsg.cit.tum.de/scannetpp/) | High-density sub-millimeter geometry and novel view synthesis | Requires registered account on official web portal. | Inspected. Focused on static radiance fields rather than temporal object movement. |

### 3RScan Access Architecture
- **Official Metadata**: The primary dataset index (`3RScan.json`, 3.01 MB), instance object catalog (`objects.json`, 19.63 MB), and scene relationship graphs (`relationships.json`, 14.22 MB) were successfully acquired from the official TUM CAMPAR public mirror and stored in `data/external/3rscan/`.
- **Raw Sensor Streams (`sequence.zip`) and Meshes (`mesh.refined.v2.obj`)**: Hosted behind TUM access control, requiring individual applicant verification. RoomShift AI's `RScanDataset` adapter was engineered to seamlessly operate in both metadata/ground-truth mode and raw mesh/OBB mode whenever raw assets are supplied.

---

## 2. 3RScan Dataset Structure & Metrics

### 2.1 Dataset Scale
- **Total Indoor Environments**: 478 scenes
- **Total 3D Scans**: 1,482 (478 reference scans + 1,004 rescans)
- **Data Splits**:
  - `train`: 385 scenes
  - `validation`: 47 scenes
  - `test`: 46 scenes (ground-truth transforms held out)
- **Temporal Pairs**: 1,004 reference-rescan pairs
- **Documented Physical Changes**:
  - `MOVED` object instances: 3,323
  - `REMOVED` object instances: 542

### 2.2 Coordinate Systems, Formats & Units
- **Global Scene Transform ($T_{\text{global}}$)**: Stored as 16 floats representing a $4 \times 4$ rigid transformation in **column-major (Fortran) order**, mapping points from rescan coordinates to reference coordinates:
  $$P_{\text{ref}} = T_{\text{global}} P_{\text{rescan}}$$
- **Object Rigid Transform ($T_{\text{obj}}$)**: Stored as 16 floats in **column-major order**, representing the transformation of the object in reference coordinates.
- **Translation Units**: In 3RScan v2, translations are metric (meters).
  - Average displacement magnitude: **1.948 meters**
  - Median displacement magnitude: **1.545 meters**
  - Minimum observed displacement: **0.045 meters**
  - Maximum observed displacement: **19.737 meters**

---

## 3. RoomShift AI Dataset Adapter Architecture

The adapter [`src/roomshift/adapters/rscan_adapter.py`](file:///Users/uday/Documents/uday_projects/ai_challenge/src/roomshift/adapters/rscan_adapter.py) translates raw 3RScan structures into native RoomShift AI domain models without data leakage:

1. **`RScanDataset`**:
   - Parses `3RScan.json` and indexes all scene hierarchies.
   - Cross-references instance IDs with `objects.json` to extract canonical semantic labels (e.g., `chair`, `table`, `backpack`, `desk`, `box`).
2. **Column-Major Matrix Conversion**:
   - Decodes 16-element lists with `np.array(t).reshape((4, 4), order='F')`.
   - Extracts translation vector $t = T[:3, 3]$ and rotation matrix $R = T[:3, :3]$.
   - Computes displacement magnitude $\Delta d = \|t\|_2$ and Euler/trace rotation angles.
3. **Bridge to RoomShift AI Domain**:
   - Converts pairs into `SceneReconstruction` (with point clouds and camera intrinsics).
   - Generates `SceneAlignment` instances with the true 3RScan global registration matrix.
   - Yields `TrackedObject` lists with 3D centroids and confidence scores.
   - Generates `GroundTruthChange` lists for benchmarking.

---

## 4. Verification & Testing

The adapter and evaluation modules are backed by automated tests in [`tests/test_rscan_adapter.py`](file:///Users/uday/Documents/uday_projects/ai_challenge/tests/test_rscan_adapter.py):
- `test_load_dataset`: Verifies loading of 478 scenes and catalogs.
- `test_list_scenes_by_split`: Verifies 385 train, 47 validation, 46 test split counts.
- `test_extract_scene_pairs`: Verifies extraction of validation pairs and global alignment.
- `test_column_major_transform_and_displacement`: Verifies mathematical correctness of column-major matrix parsing against Euclidean norm.
- `test_roomshift_reconstructions_conversion`: Verifies conversion into typed RoomShift AI domain models.
- `test_missing_data_directory_raises_error`: Verifies clean error handling on invalid paths.
- `test_evaluation_run_small`: Verifies execution of evaluation metrics on real pairs.
