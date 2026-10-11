"""
Regression tests for object association audit:
1. Stationary bed viewpoint shift recognized as stationary (not MOVED).
2. Ambiguous twin-bed reconciliation across viewpoints marked UNCERTAIN (not MOVED).
3. Desk and dining table synonym matching.
4. Unmatched bed in a room with existing beds not marked ADDED.
5. Consistent shared reconstruction coordinate frames for unmatched after detections.
"""
import numpy as np
import pytest

from roomshift.types import (
    TrackedObject,
    ObjectDetection,
    SceneAlignment,
    SceneReconstruction,
    PointCloud,
    CameraIntrinsics,
    ChangeType,
)
from roomshift.config import ChangeDetectionConfig
from roomshift.object_tracking import match_objects_across_recordings
from roomshift.change_detection import (
    detect_changes,
    _classify_matched_pair,
    _classify_unmatched_after,
)


def _make_dummy_recon(points: np.ndarray) -> SceneReconstruction:
    cloud = PointCloud(points=points, colors=np.ones_like(points))
    return SceneReconstruction(
        video_path="dummy.mp4",
        point_cloud=cloud,
        camera_poses=[],
        intrinsics=CameraIntrinsics(fx=500, fy=500, cx=320, cy=240, width=640, height=480),
        depth_maps=[],
        scale_status="relative",
        quality_score=0.9,
    )


def test_stationary_bed_viewpoint_shift_not_classified_as_moved():
    """Verify that centroid displacement on a large stationary bed (e.g. 0.8 units) is treated as stationary."""
    config = ChangeDetectionConfig(position_threshold=0.15)
    alignment = SceneAlignment(
        transform=np.eye(4),
        confidence=1.0,
        fitness=1.0,
        rmse=0.2,
        method="icp",
    )

    # Bed before at (1.5, 0.0, 7.5), bed after with 0.8 unit viewpoint shift at (1.2, 0.2, 7.1)
    obj_b = TrackedObject(
        track_id="bed_before",
        label="bed",
        detections=[],
        mean_position_3d=np.array([1.5, 0.0, 7.5]),
        mean_confidence=0.9,
    )
    obj_a = TrackedObject(
        track_id="bed_after",
        label="bed",
        detections=[],
        mean_position_3d=np.array([1.2, 0.2, 7.1]),
        mean_confidence=0.9,
    )

    change = _classify_matched_pair(obj_b, obj_a, match_score=0.8, alignment=alignment, config=config)
    assert change is None, "Stationary bed with viewpoint shift under 1.2 units should not be classified as MOVED"


def test_ambiguous_twin_bed_reconciliation_marked_uncertain():
    """Verify that reconciling two distinct beds separated by 2.5m across viewpoints is marked UNCERTAIN, not MOVED."""
    config = ChangeDetectionConfig(position_threshold=0.15)
    alignment = SceneAlignment(
        transform=np.eye(4),
        confidence=1.0,
        fitness=1.0,
        rmse=0.2,
        method="icp",
    )

    pts = np.random.uniform(5.0, 9.0, (200, 3))
    recon_b = _make_dummy_recon(pts)
    recon_a = _make_dummy_recon(pts)

    # Twin Bed 1 in before (left bed)
    obj_b = TrackedObject(
        track_id="bed_1",
        label="bed",
        detections=[],
        mean_position_3d=np.array([1.0, -3.2, 8.0]),
        mean_confidence=0.85,
    )
    # Twin Bed 2 in after (right bed, 2.5m away)
    obj_a = TrackedObject(
        track_id="bed_2",
        label="bed",
        detections=[],
        mean_position_3d=np.array([1.1, -0.9, 7.1]),
        mean_confidence=0.85,
    )

    # Unmatched in Step 1, so they go into Step 2 reconciliation
    changes = detect_changes(
        recon_before=recon_b,
        recon_after=recon_a,
        objects_before=[obj_b],
        objects_after=[obj_a],
        object_matches=[],
        alignment=alignment,
        config=config,
    )

    # Should NOT have any MOVED changes
    moved_changes = [c for c in changes if c.change_type == ChangeType.MOVED and c.object_label == "bed"]
    assert len(moved_changes) == 0, "Ambiguous twin-bed distance should not trigger a MOVED bed classification"

    # Should have an UNCERTAIN change preserving ambiguity
    uncertain_beds = [c for c in changes if c.change_type == ChangeType.UNCERTAIN and c.object_label == "bed"]
    assert len(uncertain_beds) > 0, "Ambiguous twin bed pair must be flagged as UNCERTAIN"


def test_desk_dining_table_synonym_matching():
    """Verify that 'desk' in before and 'dining table' in after match across recordings."""
    obj_desk = TrackedObject(
        track_id="desk_1",
        label="desk",
        detections=[],
        mean_position_3d=np.array([0.5, 0.2, 5.0]),
        mean_confidence=0.8,
    )
    obj_table = TrackedObject(
        track_id="table_1",
        label="dining table",
        detections=[],
        mean_position_3d=np.array([0.6, 0.2, 5.1]),
        mean_confidence=0.7,
    )

    matches = match_objects_across_recordings(
        objects1=[obj_desk],
        objects2=[obj_table],
        transform=np.eye(4),
        position_threshold=0.3,
    )
    assert len(matches) == 1, "Desk and dining table should match as table synonyms when proximate"
    assert matches[0][0].track_id == "desk_1"
    assert matches[0][1].track_id == "table_1"


def test_unmatched_bed_in_room_with_existing_beds_not_added():
    """Verify that an unmatched bed detection in after is not marked as ADDED if beds already exist."""
    config = ChangeDetectionConfig(position_threshold=0.15)
    alignment = SceneAlignment(
        transform=np.eye(4),
        confidence=1.0,
        fitness=1.0,
        rmse=0.2,
        method="icp",
    )

    pts = np.random.uniform(1.0, 4.0, (100, 3))
    recon_b = _make_dummy_recon(pts)

    existing_bed = TrackedObject(
        track_id="bed_existing",
        label="bed",
        detections=[],
        mean_position_3d=np.array([1.0, 0.0, 7.0]),
        mean_confidence=0.9,
    )
    new_bed_det = TrackedObject(
        track_id="bed_viewpoint",
        label="bed",
        detections=[],
        mean_position_3d=np.array([0.0, 0.0, 2.5]),
        mean_confidence=0.75,
    )

    change = _classify_unmatched_after(
        new_bed_det,
        recon_b,
        alignment,
        config,
        objects_before=[existing_bed],
    )
    assert change.change_type != ChangeType.ADDED, "Should not classify bed as ADDED when room already contains beds"
    assert change.change_type == ChangeType.UNCERTAIN


def test_unmatched_after_coordinates_in_shared_frame():
    """Verify that unmatched after objects have new_position in the shared coordinate frame."""
    config = ChangeDetectionConfig(position_threshold=0.15)
    # Translation of +5.0 along Z
    T = np.eye(4)
    T[2, 3] = 5.0
    alignment = SceneAlignment(
        transform=T,
        confidence=1.0,
        fitness=1.0,
        rmse=0.2,
        method="icp",
    )

    # Cloud has points near (0, 0, 7) in before frame
    pts = np.array([[0.0, 0.0, 7.0], [0.1, 0.0, 7.1]])
    recon_b = _make_dummy_recon(pts)

    # Object in scene 2 local frame at (0, 0, 2.0).
    # When transformed by T (Z + 5), it should be at (0, 0, 7.0)!
    obj_after = TrackedObject(
        track_id="chair_after",
        label="chair",
        detections=[],
        mean_position_3d=np.array([0.0, 0.0, 2.0]),
        mean_confidence=0.8,
    )

    change = _classify_unmatched_after(
        obj_after,
        recon_b,
        alignment,
        config,
        objects_before=[],
    )
    assert change.change_type == ChangeType.ADDED
    assert change.new_position is not None
    # Verify new_position is in before's shared frame (Z ≈ 7.0), NOT scene 2's local frame (Z = 2.0)
    assert np.isclose(change.new_position[2], 7.0, atol=1e-3), (
        f"Expected Z ~ 7.0 in shared frame, got {change.new_position[2]}"
    )
