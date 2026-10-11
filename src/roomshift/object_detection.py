"""
Object detection module — YOLOv8-based object understanding.

Detects objects in selected frames and estimates 3D positions
by back-projecting detection centroids through depth maps.
"""
import logging
import os
from typing import List, Optional, Tuple
import uuid

import cv2
import numpy as np

from .types import (
    ObjectDetection, ExtractedFrame, DepthMap,
    CameraIntrinsics, CameraPose, SceneReconstruction,
)
from .config import DetectionConfig

logger = logging.getLogger(__name__)

# Cache the model
_detector_model = None


def detect_objects_in_frames(
    frames: List[ExtractedFrame],
    reconstruction: SceneReconstruction,
    config: Optional[DetectionConfig] = None,
) -> List[ObjectDetection]:
    """
    Run object detection on sampled frames and estimate 3D positions.
    
    Uses YOLOv8 for detection and depth maps for 3D localization.
    """
    if config is None:
        config = DetectionConfig()
    
    model = _load_model(config)
    if model is None:
        logger.warning("Object detection model unavailable — running in geometry-only mode")
        return []
    
    all_detections = []
    
    for i, frame in enumerate(frames):
        if i % config.detect_every_n != 0:
            continue
        
        detections = _detect_single_frame(frame, model, config)
        
        # Estimate 3D positions
        for det in detections:
            det.position_3d = _estimate_3d_position(
                det, frame, reconstruction
            )
        
        all_detections.extend(detections)
        logger.debug(f"Frame {frame.frame_index}: {len(detections)} detections")
    
    logger.info(f"Total detections across frames: {len(all_detections)}")
    return all_detections


def _load_model(config: DetectionConfig):
    """Load YOLOv8 model."""
    global _detector_model
    
    if _detector_model is not None:
        return _detector_model
    
    try:
        from ultralytics import YOLO
        logger.info(f"Loading detection model: {config.model_name}")
        _detector_model = YOLO(config.model_name)
        return _detector_model
    except Exception as e:
        logger.error(f"Failed to load detection model: {e}")
        return None


def _detect_single_frame(
    frame: ExtractedFrame,
    model,
    config: DetectionConfig,
) -> List[ObjectDetection]:
    """Run detection on one frame."""
    detections = []
    
    try:
        results = model(
            frame.image,
            conf=config.confidence_threshold,
            iou=config.iou_threshold,
            max_det=config.max_detections,
            device=config.device,
            verbose=False,
        )
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            
            for j in range(len(boxes)):
                bbox = boxes.xyxy[j].cpu().numpy().astype(int)
                conf = float(boxes.conf[j].cpu())
                cls_id = int(boxes.cls[j].cpu())
                label = result.names.get(cls_id, f"class_{cls_id}")
                
                detections.append(ObjectDetection(
                    detection_id=str(uuid.uuid4())[:8],
                    frame_index=frame.frame_index,
                    label=label,
                    confidence=round(conf, 3),
                    bbox=tuple(bbox.tolist()),
                ))
    
    except Exception as e:
        logger.warning(f"Detection failed on frame {frame.frame_index}: {e}")
    
    return detections


def _estimate_3d_position(
    detection: ObjectDetection,
    frame: ExtractedFrame,
    reconstruction: SceneReconstruction,
) -> Optional[np.ndarray]:
    """
    Estimate 3D position of a detection by back-projecting its centroid
    through the depth map.
    """
    # Find depth map for this frame
    depth_map = None
    for dm in reconstruction.depth_maps:
        if dm.frame_index == frame.frame_index:
            depth_map = dm
            break
    
    if depth_map is None and reconstruction.depth_maps:
        # Use nearest depth map
        depth_map = min(
            reconstruction.depth_maps,
            key=lambda dm: abs(dm.frame_index - frame.frame_index)
        )
    
    if depth_map is None:
        return None
    
    # Find camera pose
    pose = None
    for p in reconstruction.camera_poses:
        if p.frame_index == frame.frame_index:
            pose = p
            break
    
    if pose is None:
        return None
    
    # Get centroid of bbox
    x1, y1, x2, y2 = detection.bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    
    # Get depth at centroid
    depth = depth_map.depth
    h, w = depth.shape[:2]
    img_h, img_w = frame.image.shape[:2]
    
    if (h, w) != (img_h, img_w):
        depth = cv2.resize(depth, (img_w, img_h))
    
    xi, yi = int(round(cx)), int(round(cy))
    xi = max(0, min(xi, img_w - 1))
    yi = max(0, min(yi, img_h - 1))
    
    # Sample a small region around centroid for robustness
    r = 5
    y_lo, y_hi = max(0, yi - r), min(img_h, yi + r)
    x_lo, x_hi = max(0, xi - r), min(img_w, xi + r)
    region = depth[y_lo:y_hi, x_lo:x_hi]
    
    valid = region[(region > 0) & np.isfinite(region)]
    if len(valid) == 0:
        return None
    
    z = np.median(valid)
    is_metric = getattr(depth_map, 'is_metric', False)
    if is_metric:
        z_norm = z
    else:
        # Invert MiDaS and normalize median depth to 2.0 units, exactly matching reconstruction.backproject_depth
        valid_depth = depth[(depth > 0) & np.isfinite(depth)]
        inv_all = 1.0 / (valid_depth + 1e-6)
        median_inv = np.median(inv_all) if len(inv_all) > 0 else 1.0
        if median_inv <= 0:
            median_inv = 1.0
        z_norm = (1.0 / (z + 1e-6)) / median_inv * 2.0
    
    K = reconstruction.intrinsics
    x3d = (cx - K.cx) * z_norm / K.fx
    y3d = (cy - K.cy) * z_norm / K.fy
    
    # Transform to world coordinates
    pt_cam = np.array([x3d, y3d, z_norm], dtype=np.float64).reshape(3, 1)
    R_inv = pose.rotation.T
    t_inv = -R_inv @ pose.translation.reshape(3, 1)
    pt_world = (R_inv @ pt_cam + t_inv).ravel()
    
    return pt_world

