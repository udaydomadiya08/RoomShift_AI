# RoomShift AI — Known Limitations and Engineering Considerations

## 1. Scale Ambiguity in Monocular SfM & Depth
- **Nature of Limitation**: Monocular cameras and standard neural depth estimators (e.g. MiDaS) produce relative, inverse-depth representations without metric grounding.
- **System Behavior**: RoomShift AI explicitly marks displacements as `relative units` rather than falsely claiming meters.
- **Robotics Path Forward**: In production robotic platforms, metric scale can be resolved via stereo cameras, RGB-D sensors (e.g., RealSense), LiDAR SLAM, or IMU scale integration (VIO).

## 2. Dynamic Illumination and Viewpoint Range
- **Nature of Limitation**: Dramatic illumination shifts (e.g., day vs. night) or severe viewpoint discrepancies reduce 2D feature matching yield.
- **System Behavior**: When alignment confidence drops below the threshold, the system flags the registration as unreliable and falls back to conservative `UNCERTAIN` outputs to prevent navigation failures.
- **Robotics Path Forward**: Deploy learned feature extractors (SuperPoint/LightGlue) and multi-spectral sensors.

## 3. Semantic Category Coverage
- **Nature of Limitation**: Pretrained YOLOv8 models are trained on the 80 COCO classes. Novel objects (unlabeled consumer goods, specific tools) are not semantically identified.
- **System Behavior**: RoomShift AI provides dense point-cloud KD-tree residual analysis (`GEOMETRIC_CHANGE`) so unlabeled physical changes are never missed, even if semantic labels are absent.
- **Robotics Path Forward**: Open-vocabulary vision models (e.g., Grounding DINO, MobileSAM) or robot-specific CAD model matching.

## 4. Hardware Constraints (8 GB RAM / Apple Silicon M1)
- **Nature of Limitation**: Processing high-resolution video streams simultaneously can exhaust memory.
- **System Behavior**: The pipeline implements intelligent frame decimation, sequential model execution, garbage collection, and voxel downsampling.
- **Robotics Path Forward**: Edge optimization with TensorRT / CoreML quantization for real-time onboard execution.
