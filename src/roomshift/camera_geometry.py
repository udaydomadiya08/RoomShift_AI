"""
Camera geometry module — pose estimation, feature matching, and coordinate transforms.

Uses OpenCV SIFT/ORB features with geometric verification for camera motion estimation.
Coordinate conventions: OpenCV (X-right, Y-down, Z-forward).
"""
import logging
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np

from .types import CameraIntrinsics, CameraPose, ExtractedFrame
from .config import ReconstructionConfig

logger = logging.getLogger(__name__)


def estimate_intrinsics(width: int, height: int) -> CameraIntrinsics:
    """
    Estimate camera intrinsics from image dimensions.
    
    Uses a reasonable focal length assumption for phone cameras.
    Phone cameras typically have a focal length of ~26-28mm equivalent,
    which corresponds to roughly 0.8-0.9 * max(width, height) in pixels.
    """
    focal = 0.85 * max(width, height)
    return CameraIntrinsics(
        fx=focal,
        fy=focal,
        cx=width / 2.0,
        cy=height / 2.0,
        width=width,
        height=height,
    )


def detect_features(
    image: np.ndarray,
    config: ReconstructionConfig,
) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """Detect features in an image using SIFT or ORB."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    if config.feature_type == "sift":
        detector = cv2.SIFT_create(nfeatures=config.max_features)
    else:
        detector = cv2.ORB_create(nfeatures=config.max_features)
    
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    
    if descriptors is None:
        return [], np.array([])
    
    return keypoints, descriptors


def match_features(
    desc1: np.ndarray,
    desc2: np.ndarray,
    config: ReconstructionConfig,
) -> List[cv2.DMatch]:
    """
    Match feature descriptors using Lowe's ratio test.
    Returns list of good matches.
    """
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return []
    
    if desc1.dtype == np.float32:
        matcher = cv2.BFMatcher(cv2.NORM_L2)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    
    try:
        raw_matches = matcher.knnMatch(desc1, desc2, k=2)
    except cv2.error:
        return []

    # Lowe's ratio test
    good = []
    for match_pair in raw_matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < config.match_ratio * n.distance:
                good.append(m)
    
    return good


def estimate_pose_pair(
    kp1: List[cv2.KeyPoint],
    kp2: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    intrinsics: CameraIntrinsics,
    config: ReconstructionConfig,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Estimate relative camera pose from feature matches.
    
    Returns (R, t, mask) where R is 3x3 rotation, t is 3x1 translation,
    and mask indicates inlier matches. Returns None if estimation fails.
    """
    if len(matches) < config.min_matches:
        logger.debug(f"Too few matches ({len(matches)}) for pose estimation")
        return None

    pts1 = np.float64([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float64([kp2[m.trainIdx].pt for m in matches])

    K = intrinsics.to_matrix()
    
    # Essential matrix from calibrated points
    E, mask_e = cv2.findEssentialMat(
        pts1, pts2, K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=config.ransac_threshold,
    )

    if E is None or mask_e is None:
        return None

    inliers = mask_e.ravel().astype(bool)
    inlier_count = np.sum(inliers)
    
    if inlier_count < config.min_matches:
        return None

    # Recover pose
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K, mask=mask_e.copy())
    
    if R is None or t is None:
        return None

    return R, t.reshape(3), mask_e


def estimate_poses_sequential(
    frames: List[ExtractedFrame],
    intrinsics: CameraIntrinsics,
    config: ReconstructionConfig,
) -> List[CameraPose]:
    """
    Estimate camera poses for a sequence of frames using sequential matching.
    
    First camera is placed at origin. Each subsequent camera pose is computed
    relative to the previous one and chained.
    """
    if not frames:
        return []

    # Detect features for all frames
    frame_features = []
    for f in frames:
        kp, desc = detect_features(f.image, config)
        frame_features.append((kp, desc))
        logger.debug(f"Frame {f.frame_index}: {len(kp)} features")

    # First camera at identity
    poses = [CameraPose(
        frame_index=frames[0].frame_index,
        timestamp=frames[0].timestamp,
        rotation=np.eye(3),
        translation=np.zeros(3),
        confidence=1.0,
    )]

    # Current accumulated pose
    R_acc = np.eye(3)
    t_acc = np.zeros(3)

    for i in range(1, len(frames)):
        kp1, desc1 = frame_features[i - 1]
        kp2, desc2 = frame_features[i]

        matches = match_features(desc1, desc2, config)
        result = estimate_pose_pair(kp1, kp2, matches, intrinsics, config)

        if result is None:
            # Skip this frame but try to continue from previous good pose
            logger.warning(
                f"Could not estimate pose for frame {frames[i].frame_index}, "
                f"only {len(matches)} matches"
            )
            # Still add pose with low confidence (copy previous)
            poses.append(CameraPose(
                frame_index=frames[i].frame_index,
                timestamp=frames[i].timestamp,
                rotation=R_acc.copy(),
                translation=t_acc.copy(),
                confidence=0.1,
            ))
            continue

        R_rel, t_rel, _ = result

        # Chain: R_acc = R_rel @ R_prev, t_acc = R_rel @ t_prev + t_rel
        R_acc = R_rel @ R_acc
        t_acc = R_rel @ t_acc + t_rel

        poses.append(CameraPose(
            frame_index=frames[i].frame_index,
            timestamp=frames[i].timestamp,
            rotation=R_acc.copy(),
            translation=t_acc.copy(),
            confidence=0.8,
        ))

    good_poses = sum(1 for p in poses if p.confidence > 0.5)
    logger.info(f"Estimated {good_poses}/{len(poses)} poses with good confidence")
    return poses


def triangulate_points(
    kp1: List[cv2.KeyPoint],
    kp2: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    pose1: CameraPose,
    pose2: CameraPose,
    intrinsics: CameraIntrinsics,
) -> Optional[np.ndarray]:
    """
    Triangulate 3D points from two-view correspondences.
    Returns (N, 3) array of 3D points or None.
    """
    if len(matches) < 8:
        return None

    K = intrinsics.to_matrix()
    P1 = K @ pose1.projection_matrix
    P2 = K @ pose2.projection_matrix

    pts1 = np.float64([kp1[m.queryIdx].pt for m in matches]).T  # 2xN
    pts2 = np.float64([kp2[m.trainIdx].pt for m in matches]).T  # 2xN

    points_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)  # 4xN
    points_3d = (points_4d[:3] / points_4d[3:]).T  # Nx3

    # Filter points with unreasonable depth
    mask = np.all(np.isfinite(points_3d), axis=1)
    norms = np.linalg.norm(points_3d, axis=1)
    mask &= (norms > 0.01) & (norms < 100.0)

    return points_3d[mask] if np.sum(mask) > 0 else None


def transform_points(points: np.ndarray, R: np.ndarray, t: np.ndarray, s: float = 1.0) -> np.ndarray:
    """Apply rotation, translation, and scale to points. points: (N,3)."""
    return s * (points @ R.T) + t.reshape(1, 3)


def compose_transforms(
    R1: np.ndarray, t1: np.ndarray,
    R2: np.ndarray, t2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compose two rigid transforms: T = T2 * T1."""
    R = R2 @ R1
    t = R2 @ t1 + t2
    return R, t
