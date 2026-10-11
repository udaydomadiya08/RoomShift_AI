"""
Tests for 3D scene alignment improvements:
- World-coordinate camera pose transforms in 2D-to-3D backprojection
- Consistent depth scale normalization between detection, reconstruction, and alignment
- Scale-adaptive RANSAC similarity transform estimation
- Multi-stage scale-adaptive ICP refinement on real-scale point clouds
"""
import numpy as np
import pytest

from roomshift.types import (
    CameraIntrinsics, CameraPose, DepthMap, ExtractedFrame,
    PointCloud, SceneReconstruction, SceneAlignment,
    ObjectDetection,
)
from roomshift.config import AlignmentConfig, DetectionConfig
from roomshift.scene_alignment import (
    _backproject_2d_to_3d,
    _estimate_rigid_transform,
    _ransac_rigid_transform,
    _icp_refinement,
)
from roomshift.object_detection import _estimate_3d_position


class TestAlignmentFixes:
    def test_backproject_2d_to_3d_applies_camera_pose(self):
        """Regression: _backproject_2d_to_3d must transform points to world coordinates."""
        # Camera translated by (10, 0, 0) in world
        # world-to-cam translation is (-10, 0, 0)
        pose = CameraPose(
            frame_index=0,
            timestamp=0.0,
            rotation=np.eye(3),
            translation=np.array([-10.0, 0.0, 0.0]),
        )
        intrinsics = CameraIntrinsics(fx=100.0, fy=100.0, cx=50.0, cy=50.0, width=100, height=100)
        
        # Depth map with constant raw depth = 500.0
        depth = np.full((100, 100), 500.0, dtype=np.float32)
        depth_map = DepthMap(depth=depth, frame_index=0, is_metric=False)
        
        recon = SceneReconstruction(
            video_path="test.mp4",
            point_cloud=None,
            camera_poses=[pose],
            intrinsics=intrinsics,
            depth_maps=[depth_map],
        )
        
        frame = ExtractedFrame(frame_index=0, timestamp=0.0, image=np.zeros((100, 100, 3), dtype=np.uint8), source_video="test.mp4")
        
        # Point at optical center (50, 50)
        pts_2d = np.array([[50.0, 50.0]])
        pts_3d = _backproject_2d_to_3d(pts_2d, frame, recon)
        
        assert pts_3d is not None
        assert len(pts_3d) == 1
        # In camera frame: x=0, y=0, z=2.0 (median normalized to 2.0)
        # In world frame: camera is at +10.0, so x_world should be 10.0
        assert np.isclose(pts_3d[0][0], 10.0, atol=1e-3)
        assert np.isclose(pts_3d[0][1], 0.0, atol=1e-3)
        assert np.isclose(pts_3d[0][2], 2.0, atol=1e-3)

    def test_object_detection_and_reconstruction_scale_consistency(self):
        """Regression: object 3D positions and point clouds must share the same depth scale."""
        pose = CameraPose(
            frame_index=0, timestamp=0.0,
            rotation=np.eye(3), translation=np.zeros(3)
        )
        intrinsics = CameraIntrinsics(fx=100.0, fy=100.0, cx=50.0, cy=50.0, width=100, height=100)
        depth = np.full((100, 100), 500.0, dtype=np.float32)
        depth_map = DepthMap(depth=depth, frame_index=0, is_metric=False)
        
        recon = SceneReconstruction(
            video_path="test.mp4", point_cloud=None,
            camera_poses=[pose], intrinsics=intrinsics, depth_maps=[depth_map]
        )
        frame = ExtractedFrame(frame_index=0, timestamp=0.0, image=np.zeros((100, 100, 3), dtype=np.uint8), source_video="test.mp4")
        
        det = ObjectDetection("d1", 0, "chair", 0.9, (40, 40, 60, 60))
        pt_world = _estimate_3d_position(det, frame, recon)
        
        assert pt_world is not None
        # Z coordinate should be ~2.0 units (normalized median), NOT ~0.003 units
        assert np.isclose(pt_world[2], 2.0, atol=0.1)

    def test_similarity_transform_with_scale(self):
        """Umeyama estimation recovers rotation, translation, and scale."""
        np.random.seed(42)
        src = np.random.randn(30, 3) * 5.0
        
        # Ground truth: 90 deg rotation around Z, scale 1.4, translation [2, -3, 1]
        R_gt = np.array([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        s_gt = 1.4
        t_gt = np.array([2.0, -3.0, 1.0])
        
        dst = s_gt * (src @ R_gt.T) + t_gt
        
        T, s = _estimate_rigid_transform(src, dst, allow_scale=True)
        assert T is not None
        assert np.isclose(s, s_gt, atol=1e-3)
        assert np.allclose(T[:3, 3], t_gt, atol=1e-3)
        
        # Verify transformed points match dst
        transformed = s * (src @ (T[:3, :3] / s).T) + T[:3, 3]
        assert np.allclose(transformed, dst, atol=1e-3)

    def test_scale_adaptive_icp_refinement_succeeds(self):
        """Regression: multi-stage scale-adaptive ICP succeeds on coordinate spans > 10 units."""
        np.random.seed(42)
        # 15 unit extent (similar to real room test point clouds)
        pts1 = np.random.uniform(-7.5, 7.5, size=(5000, 3))
        # Small rigid offset of 0.3 units (would fail with old 0.05 fixed threshold)
        t_offset = np.array([0.3, -0.2, 0.25])
        pts2 = pts1 + t_offset
        
        cloud1 = PointCloud(points=pts1)
        cloud2 = PointCloud(points=pts2)
        
        config = AlignmentConfig(icp_threshold=0.05, icp_iterations=30)
        refined_trans, fitness, rmse = _icp_refinement(cloud1, cloud2, np.eye(4), config)
        
        # Multi-stage ICP should find correspondences and achieve high fitness
        assert fitness > 0.5, f"Expected fitness > 0.5, got {fitness}"
        assert rmse < 0.2, f"Expected rmse < 0.2, got {rmse}"
        # Refined translation should compensate for the offset
        assert np.allclose(refined_trans[:3, 3], -t_offset, atol=0.15)
