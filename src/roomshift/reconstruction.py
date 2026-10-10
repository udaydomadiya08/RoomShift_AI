"""
3D Reconstruction module — generates point clouds from video frames.

Combines structure-from-motion camera poses with monocular depth estimation
to produce colored point clouds. Handles unknown absolute scale explicitly.
"""
import logging
import os
from typing import List, Optional

import cv2
import numpy as np
import torch

from .types import (
    ExtractedFrame, CameraIntrinsics, CameraPose,
    PointCloud, DepthMap, SceneReconstruction, FrameManifest,
)
from .camera_geometry import (
    estimate_intrinsics, estimate_poses_sequential,
    detect_features, match_features, triangulate_points,
)
from .config import ReconstructionConfig

logger = logging.getLogger(__name__)

# Global depth model cache to avoid reloading
_depth_model = None
_depth_transform = None


def reconstruct_scene(
    manifest: FrameManifest,
    config: Optional[ReconstructionConfig] = None,
) -> SceneReconstruction:
    """
    Full reconstruction pipeline for one video recording.
    
    Steps:
    1. Estimate camera intrinsics from frame dimensions.
    2. Estimate camera poses via sequential feature matching.
    3. Run monocular depth estimation on selected frames.
    4. Back-project depth maps to 3D using estimated poses.
    5. Merge into a single point cloud.
    
    Scale status: The resulting point cloud has RELATIVE scale — monocular
    depth and SfM translation scale are not metrically calibrated.
    """
    if config is None:
        config = ReconstructionConfig()

    frames = manifest.frames
    if not frames:
        logger.error("No frames to reconstruct")
        return _empty_reconstruction(manifest.video_path)

    # Step 1: Intrinsics
    h, w = frames[0].image.shape[:2]
    intrinsics = estimate_intrinsics(w, h)
    logger.info(f"Estimated intrinsics: fx={intrinsics.fx:.1f}, size={w}x{h}")

    # Step 2: Camera poses
    logger.info("Estimating camera poses...")
    poses = estimate_poses_sequential(frames, intrinsics, config)
    
    if not poses or all(p.confidence < 0.5 for p in poses):
        logger.warning("Camera pose estimation failed or low confidence")

    # Step 3: Monocular depth
    logger.info("Running monocular depth estimation...")
    depth_frames = frames[:config.max_depth_frames]
    depth_maps = estimate_depth_batch(depth_frames, config)

    # Step 4: Back-project to 3D
    logger.info("Generating point cloud from depth maps...")
    all_points = []
    all_colors = []

    for depth_map in depth_maps:
        # Find corresponding pose
        pose = _find_pose(poses, depth_map.frame_index)
        # Find corresponding frame
        frame = _find_frame(frames, depth_map.frame_index)
        
        if pose is None or frame is None:
            continue

        pts, cols = backproject_depth(
            depth_map.depth, frame.image, pose, intrinsics,
            num_points=config.points_per_depth,
        )
        if pts is not None and len(pts) > 0:
            all_points.append(pts)
            all_colors.append(cols)

    # Step 5: Also triangulate sparse points from feature matches
    sparse_pts = _triangulate_sparse(frames, poses, intrinsics, config)
    if sparse_pts is not None and len(sparse_pts) > 0:
        all_points.append(sparse_pts)
        all_colors.append(np.ones((len(sparse_pts), 3)) * 0.7)  # gray

    # Merge
    if all_points:
        merged_points = np.vstack(all_points)
        merged_colors = np.vstack(all_colors)
        
        # Statistical outlier removal
        merged_points, merged_colors = _remove_outliers(merged_points, merged_colors)
        
        point_cloud = PointCloud(
            points=merged_points,
            colors=merged_colors,
        )
        logger.info(f"Reconstruction: {point_cloud.num_points} points")
    else:
        point_cloud = None
        logger.warning("No points generated in reconstruction")

    quality = _estimate_quality(poses, point_cloud)

    return SceneReconstruction(
        video_path=manifest.video_path,
        point_cloud=point_cloud,
        camera_poses=poses,
        intrinsics=intrinsics,
        depth_maps=depth_maps,
        scale_status="relative",
        quality_score=quality,
        coordinate_frame="opencv",
        metadata={
            "num_frames": len(frames),
            "num_poses": len(poses),
            "num_depth_maps": len(depth_maps),
        },
    )


def estimate_depth_batch(
    frames: List[ExtractedFrame],
    config: ReconstructionConfig,
) -> List[DepthMap]:
    """
    Run MiDaS monocular depth estimation on a batch of frames.
    Uses the small model by default for memory efficiency.
    """
    global _depth_model, _depth_transform
    
    depth_maps = []

    try:
        if _depth_model is None:
            logger.info(f"Loading depth model: {config.depth_model}")
            _depth_model, _depth_transform = _load_midas(config.depth_model)
        
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        _depth_model = _depth_model.to(device)
        _depth_model.eval()
        
        for frame in frames:
            try:
                depth = _run_depth_inference(frame.image, _depth_model, _depth_transform, device)
                depth_maps.append(DepthMap(
                    depth=depth,
                    frame_index=frame.frame_index,
                    is_metric=False,
                    model_name=config.depth_model,
                ))
            except Exception as e:
                logger.warning(f"Depth estimation failed for frame {frame.frame_index}: {e}")

    except Exception as e:
        logger.error(f"Failed to load depth model: {e}")
        logger.info("Continuing without depth estimation — will use sparse reconstruction only")

    return depth_maps


def backproject_depth(
    depth: np.ndarray,
    image: np.ndarray,
    pose: CameraPose,
    intrinsics: CameraIntrinsics,
    num_points: int = 5000,
) -> tuple:
    """
    Back-project depth map to 3D points in world coordinates.
    
    Samples num_points from the depth map to keep point count manageable.
    Returns (points, colors) as (N,3) arrays.
    """
    h, w = depth.shape[:2]
    img_h, img_w = image.shape[:2]
    
    # Resize depth to match image if needed
    if (h, w) != (img_h, img_w):
        depth = cv2.resize(depth, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        h, w = img_h, img_w

    # Create pixel grid
    valid_mask = (depth > 0) & np.isfinite(depth)
    valid_indices = np.where(valid_mask)
    
    if len(valid_indices[0]) == 0:
        return None, None
    
    # Sample points
    n_valid = len(valid_indices[0])
    if n_valid > num_points:
        sample_idx = np.random.choice(n_valid, num_points, replace=False)
    else:
        sample_idx = np.arange(n_valid)
    
    ys = valid_indices[0][sample_idx]
    xs = valid_indices[1][sample_idx]
    zs = depth[ys, xs]
    
    # Normalize depth to reasonable range (MiDaS gives inverse depth)
    # Invert and normalize
    zs = 1.0 / (zs + 1e-6)
    zs = zs / np.median(zs) * 2.0  # normalize median depth to 2 units
    
    # Back-project to camera coordinates
    x_cam = (xs - intrinsics.cx) * zs / intrinsics.fx
    y_cam = (ys - intrinsics.cy) * zs / intrinsics.fy
    z_cam = zs
    
    points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)  # (N, 3)
    
    # Transform to world coordinates
    R_inv = pose.rotation.T
    t_inv = -R_inv @ pose.translation.reshape(3, 1)
    points_world = (R_inv @ points_cam.T + t_inv).T  # (N, 3)
    
    # Get colors (BGR -> RGB, normalize)
    if len(image.shape) == 3:
        colors = image[ys, xs, ::-1].astype(np.float64) / 255.0
    else:
        gray = image[ys, xs].astype(np.float64) / 255.0
        colors = np.stack([gray, gray, gray], axis=1)
    
    # Filter unreasonable points
    norms = np.linalg.norm(points_world, axis=1)
    mask = (norms > 0.01) & (norms < 50.0) & np.all(np.isfinite(points_world), axis=1)
    
    return points_world[mask], colors[mask]


def _load_midas(model_name: str):
    """Load MiDaS depth estimation model."""
    # Map our config names to torch hub model names
    model_map = {
        "midas_small": ("MiDaS_small", "small_transform"),
        "midas_large": ("DPT_Large", "dpt_transform"),
        "dpt_hybrid": ("DPT_Hybrid", "dpt_transform"),
    }
    
    hub_name, transform_name = model_map.get(model_name, ("MiDaS_small", "small_transform"))
    
    # Load from torch hub
    model = torch.hub.load("intel-isl/MiDaS", hub_name, trust_repo=True)
    
    # Load transform
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    if transform_name == "small_transform":
        transform = midas_transforms.small_transform
    else:
        transform = midas_transforms.dpt_transform
    
    return model, transform


def _run_depth_inference(image: np.ndarray, model, transform, device) -> np.ndarray:
    """Run depth inference on a single image."""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)
    
    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    return prediction.cpu().numpy()


def _triangulate_sparse(
    frames: List[ExtractedFrame],
    poses: List[CameraPose],
    intrinsics: CameraIntrinsics,
    config: ReconstructionConfig,
) -> Optional[np.ndarray]:
    """Triangulate sparse points from feature matches across frame pairs."""
    from .camera_geometry import detect_features, match_features
    
    all_pts = []
    step = max(1, len(frames) // 5)  # sample a few pairs
    
    for i in range(0, len(frames) - step, step):
        j = i + step
        if j >= len(frames):
            break
            
        kp1, desc1 = detect_features(frames[i].image, config)
        kp2, desc2 = detect_features(frames[j].image, config)
        matches = match_features(desc1, desc2, config)
        
        pose1 = _find_pose(poses, frames[i].frame_index)
        pose2 = _find_pose(poses, frames[j].frame_index)
        
        if pose1 and pose2 and len(matches) >= 15:
            pts = triangulate_points(kp1, kp2, matches, pose1, pose2, intrinsics)
            if pts is not None:
                all_pts.append(pts)
    
    if all_pts:
        return np.vstack(all_pts)
    return None


def _find_pose(poses: List[CameraPose], frame_index: int) -> Optional[CameraPose]:
    """Find pose for a given frame index."""
    for p in poses:
        if p.frame_index == frame_index:
            return p
    return poses[0] if poses else None


def _find_frame(frames: List[ExtractedFrame], frame_index: int) -> Optional[ExtractedFrame]:
    """Find frame by index."""
    for f in frames:
        if f.frame_index == frame_index:
            return f
    return None


def _remove_outliers(points: np.ndarray, colors: np.ndarray) -> tuple:
    """Remove statistical outliers from point cloud."""
    if len(points) < 10:
        return points, colors
    
    # Compute mean distance to k nearest neighbors
    from scipy.spatial import KDTree
    k = min(20, len(points) - 1)
    tree = KDTree(points)
    dists, _ = tree.query(points, k=k + 1)
    mean_dists = np.mean(dists[:, 1:], axis=1)
    
    threshold = np.mean(mean_dists) + 2.0 * np.std(mean_dists)
    mask = mean_dists < threshold
    
    return points[mask], colors[mask]


def _estimate_quality(poses: List[CameraPose], cloud: Optional[PointCloud]) -> float:
    """Estimate reconstruction quality [0, 1]."""
    score = 0.0
    
    if not poses:
        return 0.0
    
    # Pose quality
    good_poses = sum(1 for p in poses if p.confidence > 0.5)
    pose_ratio = good_poses / len(poses) if poses else 0
    score += 0.4 * pose_ratio
    
    # Point cloud density
    if cloud and cloud.num_points > 0:
        density_score = min(cloud.num_points / 10000.0, 1.0)
        score += 0.4 * density_score
    
    # Number of frames used
    frame_score = min(len(poses) / 20.0, 1.0)
    score += 0.2 * frame_score
    
    return round(score, 3)


def _empty_reconstruction(video_path: str) -> SceneReconstruction:
    """Create an empty reconstruction result."""
    return SceneReconstruction(
        video_path=video_path,
        point_cloud=None,
        camera_poses=[],
        intrinsics=CameraIntrinsics(fx=500, fy=500, cx=320, cy=240, width=640, height=480),
        depth_maps=[],
        scale_status="unknown",
        quality_score=0.0,
    )
