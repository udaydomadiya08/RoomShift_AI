# RoomShift AI — Asset Manifest

This document records the provenance, generation method, and authenticity status of all visual demonstration assets stored in `docs/assets/` and referenced in `README.md`.

---

## Asset Provenance & Manifest

| Asset Path | Resolution | Size | Origin / Provenance | Generation Method | Authenticity Status |
|---|---|---|---|---|---|
| `docs/assets/real_chair_comparison.jpg` | 1800 × 1200 | ~191 KB | `before.mp4` & `after.mp4` real smartphone recordings | Extracted keyframes (Frame 18 from `before.mp4` and Frame 46 from `after.mp4`) showing the study desk baseline and wardrobe relocation. Rendered via OpenCV/Matplotlib. | **Authentic Physical Evidence** |
| `docs/assets/reconstruction_3d_overlay.jpg` | 4000 × 2000 | ~575 KB | 3D point clouds and camera trajectories from `outputs/real_room_test_v2/` & `outputs/real_room_test_v3/` | 3D scatter projection of 15,000 reconstructed points per scene, camera trajectories, and chair 3D markers in the aligned bedroom volume. Rendered via Matplotlib 3D toolkit. | **Authentic Reconstructed Geometry** |
| `docs/assets/alignment_projections.jpg` | 3600 × 1200 | ~443 KB | 3D point clouds and camera poses from real video reconstruction | Orthographic 3-view projections (Top-Down $X$-$Z$, Front $X$-$Y$, Side $Z$-$Y$) showing spatial overlap and camera path entry into the room. | **Authentic Reconstructed Geometry** |
| `docs/assets/v3_changes_breakdown.jpg` | 1800 × 810 | ~97 KB | `outputs/real_room_test_v3/change_report.json` | Bar chart rendering the exact distribution of classified changes from the real video validation run. | **Authentic Metric Output** |

---

## Integrity Notes
- **Zero Synthetic Mockups:** No AI-generated or synthetic illustrations are used to represent system performance on real videos.
- **Privacy Preservation:** Full raw smartphone video files (`before.mp4`, `after.mp4`) are excluded via `.gitignore` to protect personal residential privacy. Only low-resolution representative extracted keyframes and point clouds are rendered.
