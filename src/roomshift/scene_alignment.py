"""
Scene alignment module — registration of two scene reconstructions into a shared coordinate frame.

Uses feature-based cross-recording matching, RANSAC geometric estimation,
and ICP refinement. Handles scale ambiguity from monocular depth explicitly.
"""
import logging
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.spatial import KDTree

from .types import (
    SceneReconstruction, SceneAlignment, PointCloud,
    ExtractedFrame, FrameManifest,
)
from .camera_geometry import detect_features, match_features
from .config import AlignmentConfig, ReconstructionConfig

logger = logging.getLogger(__name__)


def align_scenes(
    recon1: SceneReconstruction,
    recon2: SceneReconstruction,
    frames1: list,
    frames2: list,
    align_config: Optional[AlignmentConfig] = None,
    recon_config: Optional[ReconstructionConfig] = None,
) -> SceneAlignment:
    """
    Align two scene reconstructions into a shared coordinate frame.
    
    Strategy:
    1. Match features across recordings to find correspondences.
    2. Use matched 3D points (from depth back-projection) to estimate rigid transform.
    3. Refine with ICP if both point clouds are available.
    4. Assess registration quality.
    
    The resulting transform maps points from scene2 into scene1's coordinate frame.
    """
    if align_config is None:
        align_config = AlignmentConfig()
    if recon_config is None:
        recon_config = ReconstructionConfig()

    logger.info("Starting scene alignment...")

    # Step 1: Cross-recording feature matching
    transform, confidence, diagnostics = _feature_based_alignment(
        frames1, frames2, recon1, recon2, align_config, recon_config
    )

    # Step 2: ICP refinement if point clouds exist
    if (transform is not None and 
        recon1.point_cloud is not None and recon2.point_cloud is not None and
        recon1.point_cloud.num_points > 100 and recon2.point_cloud.num_points > 100):
        
        transform, icp_fitness, icp_rmse = _icp_refinement(
            recon1.point_cloud, recon2.point_cloud,
            transform, align_config
        )
        diagnostics['icp_fitness'] = icp_fitness
        diagnostics['icp_rmse'] = icp_rmse
        
        # Boost confidence if ICP converged well
        if icp_fitness > 0.3:
            confidence = min(confidence + 0.2, 1.0)
        elif icp_fitness > 0.1:
            confidence = min(confidence + icp_fitness * 0.3, 1.0)
    
    # Step 3: If feature-based failed, try direct ICP
    if transform is None and recon1.point_cloud and recon2.point_cloud:
        logger.info("Feature-based alignment failed, attempting direct ICP...")
        transform, icp_fitness, icp_rmse = _direct_icp(
            recon1.point_cloud, recon2.point_cloud, align_config
        )
        if transform is not None:
            confidence = icp_fitness * 0.5  # lower confidence for unverified ICP
            diagnostics['method'] = 'direct_icp'
            diagnostics['icp_fitness'] = icp_fitness
            diagnostics['icp_rmse'] = icp_rmse

    # Create result
    if transform is None:
        logger.warning("Scene alignment failed — insufficient correspondences or overlap")
        transform = np.eye(4)
        confidence = 0.0

    alignment = SceneAlignment(
        transform=transform,
        scale=diagnostics.get('scale', 1.0),
        inlier_count=diagnostics.get('inlier_count', 0),
        inlier_ratio=diagnostics.get('inlier_ratio', 0.0),
        fitness=diagnostics.get('icp_fitness', 0.0),
        rmse=diagnostics.get('icp_rmse', float('inf')),
        confidence=round(confidence, 3),
        method=diagnostics.get('method', 'feature_based'),
        diagnostics=diagnostics,
    )

    logger.info(
        f"Alignment result: confidence={alignment.confidence:.3f}, "
        f"method={alignment.method}, reliable={alignment.is_reliable}"
    )
    return alignment


def transform_point_cloud(cloud: PointCloud, transform: np.ndarray) -> PointCloud:
    """Apply a 4x4 transformation to a point cloud."""
    if cloud is None or cloud.points is None or len(cloud.points) == 0:
        return cloud
    
    R = transform[:3, :3]
    t = transform[:3, 3]
    
    transformed_points = (cloud.points @ R.T) + t
    
    return PointCloud(
        points=transformed_points,
        colors=cloud.colors.copy() if cloud.colors is not None else None,
        normals=cloud.normals,
        confidence=cloud.confidence,
    )


def _feature_based_alignment(
    frames1: list,
    frames2: list,
    recon1: SceneReconstruction,
    recon2: SceneReconstruction,
    align_config: AlignmentConfig,
    recon_config: ReconstructionConfig,
) -> Tuple[Optional[np.ndarray], float, dict]:
    """
    Estimate alignment using cross-recording feature matches.
    
    Matches features between frames from different recordings,
    finds 3D correspondences in the world coordinate frames, and estimates a rigid/similarity transform.
    """
    diagnostics = {'method': 'feature_based'}
    
    # Sample frames across each recording (up to 8 frames per recording for broader overlap)
    n1 = len(frames1)
    n2 = len(frames2)
    step1 = max(1, n1 // 8)
    step2 = max(1, n2 // 8)
    sample1 = frames1[::step1][:8]
    sample2 = frames2[::step2][:8]

    all_pts1_3d = []
    all_pts2_3d = []

    for f1 in sample1:
        kp1, desc1 = detect_features(f1.image, recon_config)
        if len(kp1) == 0:
            continue
            
        for f2 in sample2:
            kp2, desc2 = detect_features(f2.image, recon_config)
            if len(kp2) == 0:
                continue

            # Match features across recordings
            matches = match_features(desc1, desc2, recon_config)
            if len(matches) < 12:
                continue

            # Get 2D points
            pts1_2d = np.float64([kp1[m.queryIdx].pt for m in matches])
            pts2_2d = np.float64([kp2[m.trainIdx].pt for m in matches])

            # Verify with fundamental matrix
            F, mask = cv2.findFundamentalMat(
                pts1_2d, pts2_2d,
                cv2.FM_RANSAC,
                ransacReprojThreshold=align_config.ransac_threshold,
            )
            if F is None or mask is None:
                continue
            
            inliers = mask.ravel().astype(bool)
            if np.sum(inliers) < 8:
                continue

            # Back-project to 3D world coordinates using depth maps and camera poses
            pts1_3d = _backproject_2d_to_3d(
                pts1_2d[inliers], f1, recon1
            )
            pts2_3d = _backproject_2d_to_3d(
                pts2_2d[inliers], f2, recon2
            )

            if pts1_3d is not None and pts2_3d is not None:
                valid = (
                    np.all(np.isfinite(pts1_3d), axis=1) &
                    np.all(np.isfinite(pts2_3d), axis=1)
                )
                if np.sum(valid) >= 4:
                    all_pts1_3d.append(pts1_3d[valid])
                    all_pts2_3d.append(pts2_3d[valid])

    if not all_pts1_3d or not all_pts2_3d:
        logger.warning("No 3D correspondences found for alignment")
        return None, 0.0, diagnostics

    pts1 = np.vstack(all_pts1_3d)
    pts2 = np.vstack(all_pts2_3d)

    if len(pts1) < 4:
        return None, 0.0, diagnostics

    # Scale-adaptive RANSAC threshold based on coordinate extent
    extent1 = float(np.linalg.norm(np.ptp(pts1, axis=0))) if len(pts1) > 1 else 1.0
    extent2 = float(np.linalg.norm(np.ptp(pts2, axis=0))) if len(pts2) > 1 else 1.0
    ref_extent = max(extent1, extent2, 1.0)
    ransac_thresh = max(align_config.ransac_threshold, ref_extent * 0.05)

    # Estimate rigid/similarity transform with RANSAC
    transform, inlier_count, inlier_ratio, scale = _ransac_rigid_transform(
        pts2, pts1,  # from scene2 to scene1
        iterations=align_config.ransac_iterations,
        threshold=ransac_thresh,
        allow_scale=True,
    )

    diagnostics['num_correspondences'] = len(pts1)
    diagnostics['inlier_count'] = inlier_count
    diagnostics['inlier_ratio'] = inlier_ratio
    diagnostics['scale'] = scale

    confidence = min(inlier_ratio * 2.0, 1.0) * min(inlier_count / 15.0, 1.0)
    
    return transform, confidence, diagnostics


def _backproject_2d_to_3d(
    pts_2d: np.ndarray,
    frame: 'ExtractedFrame',
    recon: SceneReconstruction,
) -> Optional[np.ndarray]:
    """
    Back-project 2D points to 3D world coordinates using available depth maps
    and camera poses, exactly matching reconstruction.backproject_depth.
    """
    # Find depth map for this frame
    depth_map = None
    for dm in recon.depth_maps:
        if dm.frame_index == frame.frame_index:
            depth_map = dm
            break
    
    if depth_map is None:
        if recon.depth_maps:
            depth_map = min(
                recon.depth_maps,
                key=lambda dm: abs(dm.frame_index - frame.frame_index)
            )
        else:
            return None

    # Find camera pose for this frame
    pose = None
    for p in recon.camera_poses:
        if p.frame_index == frame.frame_index:
            pose = p
            break
    if pose is None and recon.camera_poses:
        pose = recon.camera_poses[0]
    if pose is None:
        return None

    depth = depth_map.depth
    h, w = depth.shape[:2]
    img_h, img_w = frame.image.shape[:2]
    
    if (h, w) != (img_h, img_w):
        depth = cv2.resize(depth, (img_w, img_h))

    valid_mask = (depth > 0) & np.isfinite(depth)
    if not np.any(valid_mask):
        return None

    is_metric = getattr(depth_map, 'is_metric', False)
    if not is_metric:
        inv_all = 1.0 / (depth[valid_mask] + 1e-6)
        median_inv = np.median(inv_all) if len(inv_all) > 0 else 1.0
        if median_inv <= 0:
            median_inv = 1.0

    K = recon.intrinsics
    pts_3d_world = []

    R_inv = pose.rotation.T
    t_inv = -R_inv @ pose.translation.reshape(3, 1)

    for x, y in pts_2d:
        xi, yi = int(round(x)), int(round(y))
        xi = max(0, min(xi, img_w - 1))
        yi = max(0, min(yi, img_h - 1))
        
        raw_z = depth[yi, xi]
        if raw_z <= 0 or not np.isfinite(raw_z):
            pts_3d_world.append([np.nan, np.nan, np.nan])
            continue
        
        if is_metric:
            z = raw_z
        else:
            # Invert MiDaS depth and normalize median to 2.0 units, matching backproject_depth
            z = (1.0 / (raw_z + 1e-6)) / median_inv * 2.0
        
        x_cam = (xi - K.cx) * z / K.fx
        y_cam = (yi - K.cy) * z / K.fy
        z_cam = z
        
        pt_cam = np.array([x_cam, y_cam, z_cam], dtype=np.float64).reshape(3, 1)
        pt_world = (R_inv @ pt_cam + t_inv).ravel()
        pts_3d_world.append(pt_world)

    return np.array(pts_3d_world)


def _ransac_rigid_transform(
    src: np.ndarray,
    dst: np.ndarray,
    iterations: int = 1000,
    threshold: float = 0.1,
    allow_scale: bool = False,
) -> Tuple[Optional[np.ndarray], int, float, float]:
    """
    RANSAC estimation of rigid or similarity transform.
    src, dst: (N, 3) point sets.
    Returns (4x4 transform, inlier_count, inlier_ratio, scale).
    """
    n = len(src)
    if n < 4:
        return None, 0, 0.0, 1.0

    best_transform = None
    best_inliers = 0
    best_ratio = 0.0
    best_scale = 1.0

    for _ in range(iterations):
        idx = np.random.choice(n, 4, replace=False)
        T, s = _estimate_rigid_transform(src[idx], dst[idx], allow_scale=allow_scale)
        if T is None:
            continue

        src_h = np.hstack([src, np.ones((n, 1))])
        transformed = (T @ src_h.T).T[:, :3]
        errors = np.linalg.norm(transformed - dst, axis=1)
        inliers = np.sum(errors < threshold)
        
        if inliers > best_inliers:
            best_inliers = inliers
            best_ratio = inliers / n
            best_transform = T
            best_scale = s

    # Refine with all inliers
    if best_transform is not None and best_inliers >= 4:
        src_h = np.hstack([src, np.ones((n, 1))])
        transformed = (best_transform @ src_h.T).T[:, :3]
        errors = np.linalg.norm(transformed - dst, axis=1)
        inlier_mask = errors < threshold
        
        if np.sum(inlier_mask) >= 4:
            refined, refined_s = _estimate_rigid_transform(
                src[inlier_mask], dst[inlier_mask], allow_scale=allow_scale
            )
            if refined is not None:
                best_transform = refined
                best_scale = refined_s

    return best_transform, best_inliers, best_ratio, best_scale


def _estimate_rigid_transform(
    src: np.ndarray,
    dst: np.ndarray,
    allow_scale: bool = False,
) -> Tuple[Optional[np.ndarray], float]:
    """
    Estimate rigid or similarity transform using SVD (Kabsch / Umeyama).
    src, dst: (N, 3) corresponding points.
    Returns (4x4 transformation matrix, scale).
    """
    if len(src) < 3:
        return None, 1.0
    
    centroid_src = np.mean(src, axis=0)
    centroid_dst = np.mean(dst, axis=0)
    
    src_c = src - centroid_src
    dst_c = dst - centroid_dst
    
    H = src_c.T @ dst_c / len(src)
    U, S, Vt = np.linalg.svd(H)
    
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    scale = 1.0
    if allow_scale:
        var_src = np.mean(np.sum(src_c**2, axis=1))
        if var_src > 1e-8:
            s_raw = (1.0 / var_src) * np.sum(S)
            scale = float(np.clip(s_raw, 0.3, 3.0))
    
    t = centroid_dst - scale * (R @ centroid_src)
    
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = t
    
    return T, scale


def _icp_refinement(
    cloud1: PointCloud,
    cloud2: PointCloud,
    initial_transform: np.ndarray,
    config: AlignmentConfig,
) -> Tuple[np.ndarray, float, float]:
    """
    Refine alignment using multi-stage scale-adaptive ICP.
    """
    try:
        import open3d as o3d
        
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(cloud1.points)
        
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(cloud2.points)
        
        extent1 = float(np.linalg.norm(np.ptp(cloud1.points, axis=0)))
        extent2 = float(np.linalg.norm(np.ptp(cloud2.points, axis=0)))
        extent = max(extent1, extent2, 1.0)
        
        # Scale-adaptive voxel downsampling
        voxel_size = max(0.05, extent * 0.015)
        pcd1_down = pcd1.voxel_down_sample(voxel_size)
        pcd2_down = pcd2.voxel_down_sample(voxel_size)
        
        # Estimate normals with scale-adaptive radius
        normal_radius = voxel_size * 2.5
        pcd1_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
        pcd2_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
        
        threshold_coarse = max(config.icp_threshold * 4, extent * 0.08)
        threshold_fine = max(config.icp_threshold, extent * 0.03)
        
        # Evaluate initial transform
        init_eval = o3d.pipelines.registration.evaluate_registration(
            pcd2_down, pcd1_down, threshold_coarse, initial_transform
        )
        init_fitness = init_eval.fitness
        init_rmse = init_eval.inlier_rmse
        logger.info(f"Initial alignment eval: fitness={init_fitness:.4f}, rmse={init_rmse:.4f}")
        
        # Stage 1: Coarse Point-to-Point ICP for robustness
        res_coarse = o3d.pipelines.registration.registration_icp(
            pcd2_down, pcd1_down,
            threshold_coarse,
            initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
        )
        
        # Stage 2: Fine Point-to-Plane ICP
        current_trans = res_coarse.transformation if res_coarse.fitness >= init_fitness else initial_transform
        res_fine = o3d.pipelines.registration.registration_icp(
            pcd2_down, pcd1_down,
            threshold_fine,
            current_trans,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=config.icp_iterations),
        )
        
        best_trans = initial_transform
        best_fitness = init_fitness
        best_rmse = init_rmse
        
        if res_fine.fitness >= best_fitness:
            best_trans = res_fine.transformation
            best_fitness = res_fine.fitness
            best_rmse = res_fine.inlier_rmse
        elif res_coarse.fitness >= best_fitness:
            best_trans = res_coarse.transformation
            best_fitness = res_coarse.fitness
            best_rmse = res_coarse.inlier_rmse
            
        logger.info(f"ICP: fitness={best_fitness:.4f}, rmse={best_rmse:.6f}")
        return best_trans, best_fitness, best_rmse
        
    except Exception as e:
        logger.warning(f"ICP refinement failed: {e}")
        return initial_transform, 0.0, float('inf')


def _direct_icp(
    cloud1: PointCloud,
    cloud2: PointCloud,
    config: AlignmentConfig,
) -> Tuple[Optional[np.ndarray], float, float]:
    """Direct ICP without initial alignment (less reliable)."""
    try:
        import open3d as o3d
        
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(cloud1.points)
        
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(cloud2.points)
        
        # Center both point clouds
        center1 = pcd1.get_center()
        center2 = pcd2.get_center()
        pcd2.translate(center1 - center2)
        
        init = np.eye(4)
        init[:3, 3] = center1 - center2
        
        extent1 = float(np.linalg.norm(np.ptp(cloud1.points, axis=0)))
        extent2 = float(np.linalg.norm(np.ptp(cloud2.points, axis=0)))
        extent = max(extent1, extent2, 1.0)
        voxel_size = max(0.05, extent * 0.02)
        threshold = max(config.icp_threshold * 3, extent * 0.08)
        
        pcd1_down = pcd1.voxel_down_sample(voxel_size)
        pcd2_down = pcd2.voxel_down_sample(voxel_size)
        
        pcd1_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.5, max_nn=30))
        pcd2_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.5, max_nn=30))
        
        result = o3d.pipelines.registration.registration_icp(
            pcd2_down, pcd1_down,
            threshold,
            init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
        )
        
        if result.fitness > 0.1:
            return result.transformation, result.fitness, result.inlier_rmse
        
        return None, 0.0, float('inf')
        
    except Exception as e:
        logger.warning(f"Direct ICP failed: {e}")
        return None, 0.0, float('inf')

