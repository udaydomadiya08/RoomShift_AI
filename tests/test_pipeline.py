"""
Comprehensive test suite for RoomShift AI.

Tests cover:
- Input validation
- Frame extraction
- Coordinate transformations
- Point cloud alignment utilities
- Object correspondence
- Change classification
- Visibility and uncertainty handling
- JSON serialization
- Report generation
- CLI execution
- Regression tests for failure cases
"""
import os
import sys
import json
import tempfile
import shutil
import numpy as np
import cv2
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from roomshift.types import (
    VideoMetadata, ExtractedFrame, FrameManifest, CameraIntrinsics,
    CameraPose, PointCloud, SceneReconstruction, SceneAlignment,
    ObjectDetection, TrackedObject, SpatialChange, ChangeType,
    ChangeReport, GroundTruthChange, DepthMap,
)
from roomshift.config import PipelineConfig, get_default_config
from roomshift.video_io import (
    validate_video, compute_blur_score, compute_frame_similarity,
    resize_frame, VideoValidationError,
)
from roomshift.camera_geometry import (
    estimate_intrinsics, detect_features, match_features,
    transform_points, compose_transforms,
)
from roomshift.change_detection import detect_changes
from roomshift.object_tracking import track_objects, match_objects_across_recordings
from roomshift.change_reporting import generate_json_report, generate_text_summary
from roomshift.evaluation import _compute_metrics


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def synthetic_video(tmp_dir):
    """Create a minimal synthetic video."""
    path = os.path.join(tmp_dir, "test.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, 30, (320, 240))
    for i in range(60):
        frame = np.random.randint(50, 200, (240, 320, 3), dtype=np.uint8)
        # Add some structure for feature detection
        cv2.rectangle(frame, (50+i, 50), (150+i, 150), (0, 0, 255), -1)
        cv2.circle(frame, (200, 120+i//3), 30, (255, 0, 0), -1)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def sample_frame():
    """Create a sample frame with texture."""
    img = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (200, 200), (0, 0, 255), -1)
    cv2.rectangle(img, (300, 200), (400, 350), (255, 0, 0), -1)
    # Add text/texture for features
    for i in range(0, 640, 30):
        cv2.line(img, (i, 0), (i, 480), (150, 150, 150), 1)
    return img


@pytest.fixture
def sample_intrinsics():
    return CameraIntrinsics(fx=500, fy=500, cx=320, cy=240, width=640, height=480)


@pytest.fixture
def sample_point_cloud():
    np.random.seed(42)
    pts = np.random.randn(1000, 3) * 2
    colors = np.random.rand(1000, 3)
    return PointCloud(points=pts, colors=colors)


@pytest.fixture
def sample_reconstruction(sample_intrinsics):
    np.random.seed(42)
    cloud = PointCloud(
        points=np.random.randn(500, 3),
        colors=np.random.rand(500, 3),
    )
    poses = [
        CameraPose(frame_index=0, timestamp=0.0, rotation=np.eye(3), translation=np.zeros(3)),
        CameraPose(frame_index=10, timestamp=0.33, rotation=np.eye(3), translation=np.array([0.1, 0, 0])),
    ]
    return SceneReconstruction(
        video_path="test.mp4",
        point_cloud=cloud,
        camera_poses=poses,
        intrinsics=sample_intrinsics,
        depth_maps=[],
        scale_status="relative",
        quality_score=0.5,
    )


# ─── Video I/O Tests ─────────────────────────────────────────────────────────

class TestVideoIO:
    def test_validate_missing_file(self):
        with pytest.raises(VideoValidationError, match="not found"):
            validate_video("/nonexistent/path.mp4")

    def test_validate_empty_file(self, tmp_dir):
        path = os.path.join(tmp_dir, "empty.mp4")
        open(path, 'w').close()
        with pytest.raises(VideoValidationError, match="empty"):
            validate_video(path)

    def test_validate_valid_video(self, synthetic_video):
        meta = validate_video(synthetic_video)
        assert meta.width == 320
        assert meta.height == 240
        assert meta.fps == 30.0
        assert meta.frame_count == 60
        assert meta.duration_seconds > 0

    def test_blur_score(self, sample_frame):
        score = compute_blur_score(sample_frame)
        assert score > 0
        
        # Blurred frame should have lower score
        blurred = cv2.GaussianBlur(sample_frame, (31, 31), 10)
        blur_score = compute_blur_score(blurred)
        assert blur_score < score

    def test_frame_similarity_identical(self, sample_frame):
        sim = compute_frame_similarity(sample_frame, sample_frame)
        assert sim > 0.99

    def test_frame_similarity_different(self, sample_frame):
        other = np.random.randint(0, 255, sample_frame.shape, dtype=np.uint8)
        sim = compute_frame_similarity(sample_frame, other)
        assert sim < 0.9

    def test_resize_frame(self, sample_frame):
        resized = resize_frame(sample_frame, 320)
        assert max(resized.shape[:2]) <= 320
        # Aspect ratio preserved
        orig_ratio = sample_frame.shape[1] / sample_frame.shape[0]
        new_ratio = resized.shape[1] / resized.shape[0]
        assert abs(orig_ratio - new_ratio) < 0.05

    def test_resize_already_small(self, sample_frame):
        small = cv2.resize(sample_frame, (100, 75))
        resized = resize_frame(small, 200)
        assert resized.shape == small.shape  # no change


# ─── Frame Sampling Tests ────────────────────────────────────────────────────

class TestFrameSampling:
    def test_extract_frames(self, synthetic_video):
        from roomshift.frame_sampling import extract_frames
        config = PipelineConfig().video
        config.max_frames = 10
        config.min_frames = 3
        
        manifest = extract_frames(synthetic_video, config)
        assert manifest.total_extracted >= 3
        assert manifest.total_extracted <= 10
        assert all(isinstance(f, ExtractedFrame) for f in manifest.frames)
        assert all(f.timestamp >= 0 for f in manifest.frames)
        
    def test_frame_timestamps_ordered(self, synthetic_video):
        from roomshift.frame_sampling import extract_frames
        config = PipelineConfig().video
        config.max_frames = 10
        
        manifest = extract_frames(synthetic_video, config)
        timestamps = [f.timestamp for f in manifest.frames]
        assert timestamps == sorted(timestamps)


# ─── Camera Geometry Tests ───────────────────────────────────────────────────

class TestCameraGeometry:
    def test_estimate_intrinsics(self):
        K = estimate_intrinsics(640, 480)
        assert K.fx > 0
        assert K.fy > 0
        assert K.cx == 320.0
        assert K.cy == 240.0
        mat = K.to_matrix()
        assert mat.shape == (3, 3)
        assert mat[2, 2] == 1.0

    def test_detect_features_sift(self, sample_frame):
        from roomshift.config import ReconstructionConfig
        config = ReconstructionConfig(feature_type="sift", max_features=500)
        kp, desc = detect_features(sample_frame, config)
        assert len(kp) > 0
        assert desc is not None
        assert desc.shape[0] == len(kp)

    def test_match_features(self, sample_frame):
        from roomshift.config import ReconstructionConfig
        config = ReconstructionConfig(feature_type="sift", max_features=500)
        
        # Slightly shifted frame for matching
        M = np.float32([[1, 0, 5], [0, 1, 3]])
        shifted = cv2.warpAffine(sample_frame, M, (sample_frame.shape[1], sample_frame.shape[0]))
        
        kp1, desc1 = detect_features(sample_frame, config)
        kp2, desc2 = detect_features(shifted, config)
        matches = match_features(desc1, desc2, config)
        assert len(matches) > 0

    def test_transform_points(self):
        pts = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        R = np.eye(3)
        t = np.array([1, 2, 3])
        
        result = transform_points(pts, R, t)
        expected = pts + t
        np.testing.assert_allclose(result, expected)

    def test_transform_points_with_rotation(self):
        pts = np.array([[1, 0, 0]], dtype=np.float64)
        # 90-degree rotation around Z
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        t = np.zeros(3)
        
        result = transform_points(pts, R, t)
        np.testing.assert_allclose(result, [[0, 1, 0]], atol=1e-10)

    def test_compose_transforms(self):
        R1 = np.eye(3)
        t1 = np.array([1, 0, 0])
        R2 = np.eye(3)
        t2 = np.array([0, 1, 0])
        
        R, t = compose_transforms(R1, t1, R2, t2)
        np.testing.assert_allclose(R, np.eye(3))
        np.testing.assert_allclose(t, [1, 1, 0])

    def test_camera_pose_center(self):
        pose = CameraPose(
            frame_index=0, timestamp=0,
            rotation=np.eye(3),
            translation=np.array([1, 2, 3]),
        )
        center = pose.camera_center.flatten()
        np.testing.assert_allclose(center, [-1, -2, -3])


# ─── Object Tracking Tests ──────────────────────────────────────────────────

class TestObjectTracking:
    def test_track_single_object(self):
        detections = [
            ObjectDetection("d1", 0, "chair", 0.9, (100, 100, 200, 200)),
            ObjectDetection("d2", 1, "chair", 0.85, (105, 102, 205, 202)),
            ObjectDetection("d3", 2, "chair", 0.88, (108, 104, 208, 204)),
        ]
        tracks = track_objects(detections)
        assert len(tracks) == 1
        assert tracks[0].label == "chair"
        assert len(tracks[0].detections) == 3

    def test_track_multiple_objects(self):
        detections = [
            ObjectDetection("d1", 0, "chair", 0.9, (100, 100, 200, 200)),
            ObjectDetection("d2", 0, "table", 0.8, (400, 100, 550, 200)),
            ObjectDetection("d3", 1, "chair", 0.85, (105, 102, 205, 202)),
            ObjectDetection("d4", 1, "table", 0.82, (403, 102, 553, 202)),
        ]
        tracks = track_objects(detections)
        assert len(tracks) == 2
        labels = sorted([t.label for t in tracks])
        assert labels == ["chair", "table"]

    def test_track_no_detections(self):
        tracks = track_objects([])
        assert len(tracks) == 0

    def test_match_objects_across_recordings(self):
        obj1 = TrackedObject("t1", "chair", [], mean_position_3d=np.array([1, 0, 0]))
        obj2 = TrackedObject("t2", "chair", [], mean_position_3d=np.array([1.1, 0.05, 0.02]))
        
        transform = np.eye(4)
        matches = match_objects_across_recordings([obj1], [obj2], transform, position_threshold=0.5)
        assert len(matches) == 1
        assert matches[0][0].track_id == "t1"
        assert matches[0][1].track_id == "t2"

    def test_no_match_different_positions(self):
        obj1 = TrackedObject("t1", "chair", [], mean_position_3d=np.array([0, 0, 0]))
        obj2 = TrackedObject("t2", "chair", [], mean_position_3d=np.array([10, 10, 10]))
        
        transform = np.eye(4)
        matches = match_objects_across_recordings([obj1], [obj2], transform, position_threshold=0.3)
        assert len(matches) == 0


# ─── Change Detection Tests ──────────────────────────────────────────────────

class TestChangeDetection:
    def _make_recon(self, pts=None):
        intrinsics = CameraIntrinsics(500, 500, 320, 240, 640, 480)
        cloud = PointCloud(points=pts if pts is not None else np.random.randn(100, 3))
        return SceneReconstruction(
            video_path="test.mp4", point_cloud=cloud,
            camera_poses=[], intrinsics=intrinsics, depth_maps=[],
        )

    def test_moved_object_detected(self):
        recon1 = self._make_recon()
        recon2 = self._make_recon()
        
        obj1 = TrackedObject("t1", "chair", [], mean_position_3d=np.array([1, 0, 0]))
        obj2 = TrackedObject("t2", "chair", [], mean_position_3d=np.array([3, 0, 0]))
        
        alignment = SceneAlignment(transform=np.eye(4), confidence=0.8)
        matches = [(obj1, obj2, 0.85)]
        
        from roomshift.config import ChangeDetectionConfig
        config = ChangeDetectionConfig(position_threshold=0.15, geometric_comparison=False)
        
        changes = detect_changes(recon1, recon2, [obj1], [obj2], matches, alignment, config)
        moved = [c for c in changes if c.change_type == ChangeType.MOVED]
        assert len(moved) == 1
        assert moved[0].displacement > 0
        assert moved[0].object_label == "chair"

    def test_no_change_when_same_position(self):
        recon1 = self._make_recon()
        recon2 = self._make_recon()
        
        obj1 = TrackedObject("t1", "chair", [], mean_position_3d=np.array([1, 0, 0]))
        obj2 = TrackedObject("t2", "chair", [], mean_position_3d=np.array([1.01, 0.01, 0]))
        
        alignment = SceneAlignment(transform=np.eye(4), confidence=0.8)
        matches = [(obj1, obj2, 0.9)]
        
        from roomshift.config import ChangeDetectionConfig
        config = ChangeDetectionConfig(position_threshold=0.15, geometric_comparison=False)
        
        changes = detect_changes(recon1, recon2, [obj1], [obj2], matches, alignment, config)
        moved = [c for c in changes if c.change_type == ChangeType.MOVED]
        assert len(moved) == 0

    def test_removed_object(self):
        pts = np.random.randn(200, 3)
        recon1 = self._make_recon(pts)
        recon2 = self._make_recon(pts)  # same cloud = location was observed
        
        obj_before = TrackedObject("t1", "box", [], mean_position_3d=np.mean(pts[:10], axis=0))
        
        alignment = SceneAlignment(transform=np.eye(4), confidence=0.8)
        
        from roomshift.config import ChangeDetectionConfig
        config = ChangeDetectionConfig(geometric_comparison=False)
        
        changes = detect_changes(recon1, recon2, [obj_before], [], [], alignment, config)
        removed = [c for c in changes if c.change_type == ChangeType.REMOVED]
        assert len(removed) >= 1

    def test_added_object(self):
        pts = np.random.randn(200, 3)
        recon1 = self._make_recon(pts)
        recon2 = self._make_recon(pts)
        
        obj_after = TrackedObject("t1", "bag", [], mean_position_3d=np.mean(pts[:10], axis=0))
        
        alignment = SceneAlignment(transform=np.eye(4), confidence=0.8)
        
        from roomshift.config import ChangeDetectionConfig
        config = ChangeDetectionConfig(geometric_comparison=False)
        
        changes = detect_changes(recon1, recon2, [], [obj_after], [], alignment, config)
        added = [c for c in changes if c.change_type == ChangeType.ADDED]
        assert len(added) >= 1

    def test_uncertain_with_low_alignment(self):
        recon1 = self._make_recon()
        recon2 = self._make_recon()
        
        obj1 = TrackedObject("t1", "chair", [], mean_position_3d=np.array([1, 0, 0]))
        
        # Very low alignment confidence
        alignment = SceneAlignment(transform=np.eye(4), confidence=0.1)
        
        from roomshift.config import ChangeDetectionConfig
        config = ChangeDetectionConfig(geometric_comparison=False, min_confidence=0.25)
        
        changes = detect_changes(recon1, recon2, [obj1], [], [], alignment, config)
        # With low alignment, should be uncertain
        uncertain = [c for c in changes if c.change_type == ChangeType.UNCERTAIN]
        assert len(uncertain) >= 1

    def test_camera_moves_no_objects_move(self):
        """Regression: camera motion alone should not produce MOVED changes."""
        pts = np.random.randn(200, 3)
        recon1 = self._make_recon(pts)
        recon2 = self._make_recon(pts)
        
        obj1 = TrackedObject("t1", "table", [], mean_position_3d=np.array([0, 0, 0]))
        obj2 = TrackedObject("t2", "table", [], mean_position_3d=np.array([0, 0, 0]))
        
        alignment = SceneAlignment(transform=np.eye(4), confidence=0.9)
        matches = [(obj1, obj2, 0.95)]
        
        from roomshift.config import ChangeDetectionConfig
        config = ChangeDetectionConfig(geometric_comparison=False)
        
        changes = detect_changes(recon1, recon2, [obj1], [obj2], matches, alignment, config)
        moved = [c for c in changes if c.change_type == ChangeType.MOVED]
        assert len(moved) == 0

    def test_unobserved_region_not_classified_as_empty(self):
        """Regression: a region not observed in the second recording should be UNCERTAIN."""
        # Cloud 2 is far from object position = region unobserved
        recon1 = self._make_recon(np.random.randn(100, 3) * 0.5)
        recon2 = self._make_recon(np.random.randn(100, 3) * 0.5 + 50)  # far away
        
        obj_before = TrackedObject("t1", "lamp", [], 
                                   mean_position_3d=np.array([0, 0, 0]))
        
        alignment = SceneAlignment(transform=np.eye(4), confidence=0.7)
        
        from roomshift.config import ChangeDetectionConfig
        config = ChangeDetectionConfig(geometric_comparison=False)
        
        changes = detect_changes(recon1, recon2, [obj_before], [], [], alignment, config)
        # Should be uncertain since the region wasn't observed in after
        assert any(c.change_type == ChangeType.UNCERTAIN for c in changes)


# ─── JSON Serialization Tests ────────────────────────────────────────────────

class TestSerialization:
    def test_spatial_change_to_dict(self):
        change = SpatialChange(
            change_id="test_1",
            change_type=ChangeType.MOVED,
            object_label="chair",
            confidence=0.85,
            old_position=np.array([1.0, 2.0, 3.0]),
            new_position=np.array([4.0, 5.0, 6.0]),
            displacement=5.196,
        )
        d = change.to_dict()
        assert d['change_id'] == "test_1"
        assert d['change_type'] == "moved"
        assert d['old_position'] == [1.0, 2.0, 3.0]
        assert d['displacement'] == 5.196
        # Must be JSON serializable
        json.dumps(d)

    def test_change_report_to_dict(self):
        changes = [SpatialChange(
            change_id="c1", change_type=ChangeType.ADDED,
            object_label="box", confidence=0.7,
        )]
        alignment = SceneAlignment(transform=np.eye(4), confidence=0.6)
        report = ChangeReport(changes=changes, alignment=alignment)
        d = report.to_dict()
        assert d['num_changes'] == 1
        assert d['alignment_confidence'] == 0.6
        json.dumps(d)

    def test_null_positions_serializable(self):
        change = SpatialChange(
            change_id="c1", change_type=ChangeType.UNCERTAIN,
            confidence=0.1,
        )
        d = change.to_dict()
        assert d['old_position'] is None
        assert d['new_position'] is None
        assert d['displacement'] is None
        json.dumps(d)


# ─── Report Generation Tests ─────────────────────────────────────────────────

class TestReporting:
    def test_generate_json_report(self, tmp_dir):
        changes = [SpatialChange(
            change_id="c1", change_type=ChangeType.MOVED,
            object_label="chair", confidence=0.85,
            old_position=np.array([1, 2, 3]),
            displacement=1.5,
        )]
        alignment = SceneAlignment(transform=np.eye(4), confidence=0.7)
        report = ChangeReport(changes=changes, alignment=alignment)
        
        path = os.path.join(tmp_dir, "report.json")
        generate_json_report(report, path)
        
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert data['num_changes'] == 1
        assert data['changes'][0]['change_type'] == 'moved'

    def test_generate_text_summary(self):
        changes = [
            SpatialChange("c1", ChangeType.MOVED, "chair", 0.8, displacement=1.2),
            SpatialChange("c2", ChangeType.ADDED, "box", 0.6),
            SpatialChange("c3", ChangeType.UNCERTAIN, "lamp", 0.1),
        ]
        alignment = SceneAlignment(transform=np.eye(4), confidence=0.7)
        report = ChangeReport(changes=changes, alignment=alignment)
        
        text = generate_text_summary(report)
        assert "ROOMSHIFT AI" in text
        assert "Moved objects:" in text
        assert "Added objects:" in text
        assert "chair" in text.lower()


# ─── Evaluation Tests ─────────────────────────────────────────────────────────

class TestEvaluation:
    def test_perfect_predictions(self):
        predicted = [
            SpatialChange("c1", ChangeType.MOVED, "chair", 0.9),
            SpatialChange("c2", ChangeType.ADDED, "box", 0.8),
        ]
        ground_truth = [
            GroundTruthChange("chair", "moved"),
            GroundTruthChange("box", "added"),
        ]
        result = _compute_metrics(predicted, ground_truth)
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1 == 1.0

    def test_false_positive(self):
        predicted = [
            SpatialChange("c1", ChangeType.MOVED, "chair", 0.9),
            SpatialChange("c2", ChangeType.ADDED, "phantom", 0.6),
        ]
        ground_truth = [
            GroundTruthChange("chair", "moved"),
        ]
        result = _compute_metrics(predicted, ground_truth)
        assert result.true_positives == 1
        assert result.false_positives == 1
        assert result.precision == 0.5

    def test_false_negative(self):
        predicted = [
            SpatialChange("c1", ChangeType.MOVED, "chair", 0.9),
        ]
        ground_truth = [
            GroundTruthChange("chair", "moved"),
            GroundTruthChange("box", "added"),
        ]
        result = _compute_metrics(predicted, ground_truth)
        assert result.false_negatives == 1
        assert result.recall == 0.5

    def test_uncertain_not_counted_as_fp(self):
        predicted = [
            SpatialChange("c1", ChangeType.UNCERTAIN, "maybe", 0.1),
        ]
        ground_truth = []
        result = _compute_metrics(predicted, ground_truth)
        # Uncertain should be filtered out
        assert result.false_positives == 0


# ─── Config Tests ────────────────────────────────────────────────────────────

class TestConfig:
    def test_default_config(self):
        config = get_default_config()
        assert config.video.max_frames == 30
        assert config.reconstruction.feature_type == "sift"
        assert config.detection.model_name == "yolov8n.pt"

    def test_config_save_load(self, tmp_dir):
        config = get_default_config()
        path = os.path.join(tmp_dir, "config.json")
        config.save(path)
        
        loaded = PipelineConfig.from_file(path)
        assert loaded.video.max_frames == config.video.max_frames
        assert loaded.reconstruction.feature_type == config.reconstruction.feature_type

    def test_alignment_reliable(self):
        a = SceneAlignment(transform=np.eye(4), confidence=0.5)
        assert a.is_reliable
        
        a2 = SceneAlignment(transform=np.eye(4), confidence=0.2)
        assert not a2.is_reliable


# ─── CLI Tests ───────────────────────────────────────────────────────────────

class TestCLI:
    def test_cli_help(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, '-m', 'roomshift.cli', '--help'],
            capture_output=True, text=True,
            cwd=os.path.join(os.path.dirname(__file__), '..'),
            env={**os.environ, 'PYTHONPATH': os.path.join(os.path.dirname(__file__), '..', 'src')},
        )
        assert result.returncode == 0
        assert 'RoomShift AI' in result.stdout or 'compare' in result.stdout

    def test_cli_missing_input(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, '-m', 'roomshift.cli', 'compare',
             '--before', '/nonexistent.mp4', '--after', '/nonexistent2.mp4'],
            capture_output=True, text=True,
            cwd=os.path.join(os.path.dirname(__file__), '..'),
            env={**os.environ, 'PYTHONPATH': os.path.join(os.path.dirname(__file__), '..', 'src')},
        )
        assert result.returncode != 0

    def test_cli_evaluate_3rscan_help(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, '-m', 'roomshift.cli', 'evaluate-3rscan', '--help'],
            capture_output=True, text=True,
            cwd=os.path.join(os.path.dirname(__file__), '..'),
            env={**os.environ, 'PYTHONPATH': os.path.join(os.path.dirname(__file__), '..', 'src')},
        )
        assert result.returncode == 0
        assert '3RScan' in result.stdout or 'max-scenes' in result.stdout


# ─── Regression Tests ────────────────────────────────────────────────────────

class TestRegressions:
    def test_similar_looking_objects_not_auto_matched(self):
        """Two objects with same label but far apart should not be matched."""
        obj1 = TrackedObject("t1", "chair", [], mean_position_3d=np.array([0, 0, 0]))
        obj2 = TrackedObject("t2", "chair", [], mean_position_3d=np.array([100, 100, 100]))
        
        transform = np.eye(4)
        matches = match_objects_across_recordings([obj1], [obj2], transform, position_threshold=0.3)
        assert len(matches) == 0

    def test_registration_failure_handled(self):
        """When alignment fails, changes should be marked uncertain."""
        from roomshift.types import SceneReconstruction, PointCloud
        
        recon1 = SceneReconstruction(
            "a.mp4", PointCloud(np.random.randn(50, 3)),
            [], CameraIntrinsics(500, 500, 320, 240, 640, 480), [],
        )
        recon2 = SceneReconstruction(
            "b.mp4", PointCloud(np.random.randn(50, 3)),
            [], CameraIntrinsics(500, 500, 320, 240, 640, 480), [],
        )
        
        obj = TrackedObject("t1", "table", [], mean_position_3d=np.array([1, 0, 0]))
        alignment = SceneAlignment(transform=np.eye(4), confidence=0.05)
        
        from roomshift.config import ChangeDetectionConfig
        config = ChangeDetectionConfig(geometric_comparison=False, min_confidence=0.25)
        
        changes = detect_changes(recon1, recon2, [obj], [], [], alignment, config)
        assert all(c.change_type == ChangeType.UNCERTAIN for c in changes)
        assert any("alignment" in str(c.limitations).lower() for c in changes)

    def test_unknown_scale_not_reported_as_metres(self):
        """Distances from relative reconstruction must not claim metric units."""
        change = SpatialChange(
            "c1", ChangeType.MOVED, "chair", 0.8,
            old_position=np.array([0, 0, 0]),
            new_position=np.array([1, 0, 0]),
            displacement=1.0,
            scale_status="relative",
        )
        d = change.to_dict()
        assert d['scale_status'] == "relative"
        
        text = generate_text_summary(ChangeReport(
            changes=[change],
            alignment=SceneAlignment(transform=np.eye(4), confidence=0.7),
        ))
        assert "relative" in text.lower()

    def test_genuine_move_produces_displacement(self):
        """A genuinely moved object should have nonzero displacement."""
        recon = SceneReconstruction(
            "test.mp4", PointCloud(np.random.randn(100, 3)),
            [], CameraIntrinsics(500, 500, 320, 240, 640, 480), [],
        )
        
        obj1 = TrackedObject("t1", "cup", [], mean_position_3d=np.array([0, 0, 0]))
        obj2 = TrackedObject("t2", "cup", [], mean_position_3d=np.array([2, 0, 0]))
        
        alignment = SceneAlignment(transform=np.eye(4), confidence=0.9)
        matches = [(obj1, obj2, 0.9)]
        
        from roomshift.config import ChangeDetectionConfig
        config = ChangeDetectionConfig(position_threshold=0.15, geometric_comparison=False)
        
        changes = detect_changes(recon, recon, [obj1], [obj2], matches, alignment, config)
        moved = [c for c in changes if c.change_type == ChangeType.MOVED]
        assert len(moved) == 1
        assert moved[0].displacement > 1.5


# ─── Web Application Tests ──────────────────────────────────────────────────

class TestWebApp:
    @pytest.fixture
    def client(self):
        import app as flask_app
        flask_app.app.config['TESTING'] = True
        with flask_app.app.test_client() as client:
            yield client

    def test_index_page(self, client):
        res = client.get('/')
        assert res.status_code == 200
        assert b"RoomShift AI" in res.data

    def test_api_status_not_found(self, client):
        res = client.get('/api/status/nonexistent_job_123')
        assert res.status_code == 404
        assert b"Job not found" in res.data

    def test_api_process_missing_files(self, client):
        res = client.post('/api/process', data={})
        assert res.status_code == 400
        assert b"required" in res.data

