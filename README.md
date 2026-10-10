# RoomShift AI

An end-to-end 3D spatial change detection system for indoor environments.
Given two phone recordings of the same room taken at different times, RoomShift AI
identifies objects that have moved, appeared, or disappeared — distinguishing genuine
physical changes from camera motion, viewpoint differences, and occlusion.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run CLI comparison
python -m roomshift.cli compare --before examples/moved/before.mp4 --after examples/moved/after.mp4 --output outputs/demo

# Run real 3RScan dataset benchmark
python -m roomshift.cli evaluate-3rscan --data-dir data/external/3rscan --max-scenes 20

# Launch web interface
python app.py

# Run all 58 automated tests
pytest -v
```

## Real-World Dataset Integration (3RScan Benchmark)

RoomShift AI integrates the official **3RScan** dataset (Wald et al., ICCV 2019 / TUM CAMPAR), featuring 478 naturally changing indoor environments across 1,482 scans.

- See [docs/DATASET_ANALYSIS.md](docs/DATASET_ANALYSIS.md) for licensing, data format specifications, and coordinate frames.
- See [docs/EVALUATION.md](docs/EVALUATION.md) for empirical benchmark results on the 3RScan validation split.

### Evaluation Separation Taxonomy
- **Tier A (Synthetic Tests)**: 50 unit and regression tests verifying mathematical transforms, coordinate frame conversions, and pipeline components.
- **Tier B (Real 3D Dataset Benchmark)**: 3RScan validation benchmark evaluating spatial change detection on real human room reconfigurations (F1: 0.318, Displacement MAE: 0.84 m across 25 scenes).
- **Tier C (Casual Phone Video)**: End-to-end monocular video pipeline operational; full validation on casual uncalibrated consumer phone recordings remains pending authorized raw uncompressed video access.


## Architecture

```mermaid
graph TD
    A[Video Input] --> B[Frame Extraction]
    B --> C[Feature Detection]
    B --> D[Monocular Depth]
    C --> E[Camera Poses]
    D --> F[Point Clouds]
    E --> F
    F --> G[Scene Alignment]
    B --> H[Object Detection]
    G --> I[Change Detection]
    H --> I
    I --> J[Change Report]
    I --> K[3D Visualization]
    J --> L[JSON + Summary]
    K --> L
```

## System Requirements

- Python 3.9+
- macOS (Apple Silicon) or Linux
- 8 GB RAM minimum
- FFmpeg installed
- ~2 GB disk for model weights (downloaded on first run)
- No GPU required (MPS acceleration used when available)

## Modules

| Module | File | Purpose |
|--------|------|---------|
| Video I/O | `src/roomshift/video_io.py` | Video validation, metadata extraction |
| Frame Sampling | `src/roomshift/frame_sampling.py` | Intelligent frame extraction |
| Reconstruction | `src/roomshift/reconstruction.py` | SfM + depth-based 3D reconstruction |
| Camera Geometry | `src/roomshift/camera_geometry.py` | Pose estimation, coordinate transforms |
| Scene Alignment | `src/roomshift/scene_alignment.py` | Cross-recording registration |
| Object Detection | `src/roomshift/object_detection.py` | YOLOv8-based object understanding |
| Object Tracking | `src/roomshift/object_tracking.py` | Cross-frame object association |
| Change Detection | `src/roomshift/change_detection.py` | Spatial change classification |
| Change Reporting | `src/roomshift/change_reporting.py` | Reports and natural-language summaries |
| Visualization | `src/roomshift/visualization.py` | 3D scene visualization |
| Pipeline | `src/roomshift/pipeline.py` | End-to-end orchestration |
| Config | `src/roomshift/config.py` | Centralized configuration |
| Evaluation | `src/roomshift/evaluation.py` | Metrics and benchmarking |
| 3RScan Adapter | `src/roomshift/adapters/rscan_adapter.py` | Official 3RScan real-data ingestion |
| 3RScan Eval | `src/roomshift/adapters/evaluate_3rscan.py` | 3RScan validation benchmark runner |
| CLI | `src/roomshift/cli.py` | Command-line interface |

## Change Detection Method

See [docs/CHANGE_DETECTION.md](docs/CHANGE_DETECTION.md) for the full algorithm description.

The system classifies changes into:
- **MOVED**: Object detected in both recordings at different 3D positions
- **ADDED**: Object in the later recording, location was observed empty before
- **REMOVED**: Object in the earlier recording, location observed empty after
- **GEOMETRIC_CHANGE**: Spatial difference not assignable to a specific object
- **UNCERTAIN**: Insufficient evidence to determine change status

## Known Limitations

See [docs/LIMITATIONS.md](docs/LIMITATIONS.md) for details.

- Monocular depth has unknown absolute scale; distances are relative
- Reconstruction quality depends on video motion and texture
- Object detection limited to COCO categories
- Requires sufficient visual overlap between recordings
- 8 GB RAM limits simultaneous model loading

## License

MIT License — see [LICENSE](LICENSE).

## Third-Party Attribution

- [MiDaS](https://github.com/isl-org/MiDaS) — MIT License
- [YOLOv8](https://github.com/ultralytics/ultralytics) — AGPL-3.0 (inference only)
- [Open3D](http://www.open3d.org/) — MIT License
