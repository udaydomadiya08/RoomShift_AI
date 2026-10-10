# RoomShift AI — Implementation Plan

## Environment
- Python 3.9.18, Apple Silicon M1, 8 GB RAM, 34 GB disk
- OpenCV 4.10, PyTorch 2.8 (MPS), Open3D 0.18, Ultralytics 8.4, Plotly 6.7
- FFmpeg 8.0, Flask 3.0, SciPy, sklearn, trimesh
- No COLMAP available — using OpenCV SfM + monocular depth

## Architecture Decision
Since COLMAP is unavailable, reconstruction uses:
1. **OpenCV feature matching** (SIFT/ORB) for camera pose estimation
2. **Monocular depth estimation** (MiDaS via PyTorch) for dense geometry
3. **Point cloud generation** from depth + poses
4. **ICP + feature-based registration** for scene alignment
5. **YOLOv8** for object detection
6. **Geometric + semantic** change detection

## Milestones

### M1: Project scaffold & video ingestion ✅
- Directory structure, configs, typed data structures
- Video validation, metadata, frame extraction with sampling

### M2: 3D reconstruction ✅
- Feature detection & matching (SIFT)
- Essential/Fundamental matrix estimation
- Camera pose recovery
- MiDaS monocular depth
- Point cloud generation from depth maps

### M3: Scene alignment ✅
- Cross-recording feature matching
- RANSAC-based transformation estimation  
- ICP refinement
- Registration quality metrics

### M4: Object detection & tracking ✅
- YOLOv8 detection on sampled frames
- Cross-frame association
- 3D position estimation via depth backprojection
- Cross-recording object matching

### M5: Change detection ✅
- Geometric comparison in shared frame
- Object-level correspondence
- Change classification (moved/added/removed/uncertain)
- Visibility & occlusion analysis
- Confidence estimation

### M6: Reporting & visualization ✅
- JSON schema for change records
- Human-readable reports
- 3D visualization with Plotly
- Web interface with Flask

### M7: Testing & evaluation ✅
- Unit tests for each module
- Integration tests
- Synthetic test scenarios
- Evaluation metrics (50 passing tests)

### M8: Documentation & polish ✅
- README, architecture docs (`docs/CHANGE_DETECTION.md`, `docs/LIMITATIONS.md`)
- CLI verification (`compare` and `evaluate` subcommands)
- Final audit and end-to-end execution verified

## Risks
- MiDaS depth has unknown absolute scale → use relative alignment
- 8GB RAM constraint → process frames incrementally, limit batch sizes
- No real test videos initially → create synthetic fixtures + recording guide
