"""
Spatial change detection module — the central intelligence component.

Compares two aligned scene representations and classifies changes into:
MOVED, ADDED, REMOVED, GEOMETRIC_CHANGE, and UNCERTAIN.

Distinguishes genuine physical changes from:
- Camera motion / viewpoint differences
- Occlusion and incomplete observation
- Reconstruction noise
- Ambiguous object matches

See docs/CHANGE_DETECTION.md for the full algorithm description.
"""
import logging
from typing import List, Optional, Tuple, Set
import uuid

import numpy as np
from scipy.spatial import KDTree

from .types import (
    ChangeType, SpatialChange, SceneAlignment, SceneReconstruction,
    TrackedObject, PointCloud,
)
from .config import ChangeDetectionConfig

logger = logging.getLogger(__name__)


def detect_changes(
    recon_before: SceneReconstruction,
    recon_after: SceneReconstruction,
    objects_before: List[TrackedObject],
    objects_after: List[TrackedObject],
    object_matches: List[tuple],
    alignment: SceneAlignment,
    config: Optional[ChangeDetectionConfig] = None,
) -> List[SpatialChange]:
    """
    Detect spatial changes between two aligned scene reconstructions.
    
    Algorithm:
    1. Classify matched objects (MOVED if displaced, UNCHANGED otherwise).
    2. Identify REMOVED objects (in before, no match, location observed in after).
    3. Identify ADDED objects (in after, no match, location observed in before).
    4. Detect GEOMETRIC_CHANGE from point cloud comparison.
    5. Mark ambiguous cases as UNCERTAIN.
    
    All classifications consider alignment confidence and visibility.
    """
    if config is None:
        config = ChangeDetectionConfig()
    
    changes: List[SpatialChange] = []
    
    # Track which objects have been matched
    matched_before_ids = set()
    matched_after_ids = set()
    
    # Step 1: Process matched objects
    for obj_before, obj_after, match_score in object_matches:
        matched_before_ids.add(obj_before.track_id)
        matched_after_ids.add(obj_after.track_id)
        
        change = _classify_matched_pair(
            obj_before, obj_after, match_score,
            alignment, config
        )
        if change is not None:
            changes.append(change)
    
    # Step 2: Unmatched objects from BEFORE → possible REMOVED
    for obj in objects_before:
        if obj.track_id in matched_before_ids:
            continue
        change = _classify_unmatched_before(
            obj, recon_after, alignment, config
        )
        changes.append(change)
    
    # Step 3: Unmatched objects from AFTER → possible ADDED
    for obj in objects_after:
        if obj.track_id in matched_after_ids:
            continue
        change = _classify_unmatched_after(
            obj, recon_before, alignment, config
        )
        changes.append(change)
    
    # Step 4: Geometric change detection from point clouds
    if config.geometric_comparison:
        geo_changes = _detect_geometric_changes(
            recon_before, recon_after, alignment, config
        )
        changes.extend(geo_changes)
    
    # Apply alignment confidence penalty
    if not alignment.is_reliable:
        for change in changes:
            change.confidence *= alignment.confidence
            change.limitations.append(
                f"Scene alignment confidence is low ({alignment.confidence:.2f}). "
                "Change classification may be unreliable."
            )
            if change.confidence < config.min_confidence:
                change.change_type = ChangeType.UNCERTAIN
    
    # Sort by confidence (descending)
    changes.sort(key=lambda c: c.confidence, reverse=True)
    
    logger.info(
        f"Detected {len(changes)} changes: "
        f"{sum(1 for c in changes if c.change_type == ChangeType.MOVED)} moved, "
        f"{sum(1 for c in changes if c.change_type == ChangeType.ADDED)} added, "
        f"{sum(1 for c in changes if c.change_type == ChangeType.REMOVED)} removed, "
        f"{sum(1 for c in changes if c.change_type == ChangeType.GEOMETRIC_CHANGE)} geometric, "
        f"{sum(1 for c in changes if c.change_type == ChangeType.UNCERTAIN)} uncertain"
    )
    
    return changes


def _classify_matched_pair(
    obj_before: TrackedObject,
    obj_after: TrackedObject,
    match_score: float,
    alignment: SceneAlignment,
    config: ChangeDetectionConfig,
) -> Optional[SpatialChange]:
    """
    Classify a matched object pair: MOVED if displaced beyond threshold, or None if unchanged.
    """
    pos_before = obj_before.mean_position_3d
    pos_after = obj_after.mean_position_3d
    
    if pos_before is None or pos_after is None:
        return None
    
    # Transform after position to before's coordinate frame
    R = alignment.transform[:3, :3]
    t = alignment.transform[:3, 3]
    pos_after_aligned = R @ pos_after + t
    
    displacement = float(np.linalg.norm(pos_before - pos_after_aligned))
    
    if displacement < config.position_threshold:
        # Object hasn't moved significantly
        return None
    
    # Object has moved
    confidence = min(match_score * alignment.confidence * 1.5, 0.95)
    
    evidence = [
        f"Object '{obj_before.label}' detected in both recordings",
        f"3D displacement: {displacement:.3f} reconstruction units",
        f"Match score: {match_score:.2f}",
    ]
    
    limitations = []
    if alignment.confidence < 0.7:
        limitations.append(
            "Moderate alignment confidence — displacement may be partly due to registration error"
        )
    limitations.append("Distances are in relative reconstruction units, not metres")
    
    return SpatialChange(
        change_id=f"chg_{uuid.uuid4().hex[:6]}",
        change_type=ChangeType.MOVED,
        object_label=obj_before.label,
        confidence=round(confidence, 3),
        old_position=pos_before,
        new_position=pos_after_aligned,
        displacement=displacement,
        coordinate_frame="shared_reconstruction",
        scale_status="relative",
        evidence=evidence,
        before_timestamps=[d.timestamp for d in obj_before.detections
                          if hasattr(d, 'timestamp')],
        after_timestamps=[d.timestamp for d in obj_after.detections
                         if hasattr(d, 'timestamp')],
        visibility_status="observed",
        registration_confidence=alignment.confidence,
        limitations=limitations,
        before_object=obj_before,
        after_object=obj_after,
    )


def _classify_unmatched_before(
    obj: TrackedObject,
    recon_after: SceneReconstruction,
    alignment: SceneAlignment,
    config: ChangeDetectionConfig,
) -> SpatialChange:
    """
    Classify an object detected in BEFORE but not matched in AFTER.
    Could be REMOVED or UNCERTAIN (if location wasn't observed).
    """
    pos = obj.mean_position_3d
    
    if pos is None:
        return _make_uncertain_change(
            obj, "before",
            "Object position could not be estimated in 3D"
        )
    
    # Check if the object's location was observed in the after recording
    visibility = _check_visibility(pos, recon_after, alignment)
    
    if visibility in ("observed", "partial"):
        # Location was visible in after → object was likely removed
        det_conf = obj.mean_confidence if obj.mean_confidence > 0 else 0.8
        base_conf = min(det_conf * alignment.confidence, 0.85)
        confidence = base_conf * 0.8 if visibility == "partial" else base_conf
        
        if confidence < config.min_confidence:
            return _make_uncertain_change(
                obj, "before",
                "Low confidence in detection, alignment, or visibility"
            )
        
        return SpatialChange(
            change_id=f"chg_{uuid.uuid4().hex[:6]}",
            change_type=ChangeType.REMOVED,
            object_label=obj.label,
            confidence=round(confidence, 3),
            old_position=pos,
            new_position=None,
            displacement=None,
            evidence=[
                f"Object '{obj.label}' detected in before recording",
                "Not matched to any object in after recording",
                f"Object's original location was {visibility} in after recording",
            ],
            visibility_status=visibility,
            registration_confidence=alignment.confidence,
            limitations=[
                "Distances are in relative reconstruction units",
                "Object may have been occluded or out of frame rather than removed",
            ],
            before_object=obj,
        )
    else:
        return _make_uncertain_change(
            obj, "before",
            f"Object's original location was {visibility} in the after recording"
        )


def _classify_unmatched_after(
    obj: TrackedObject,
    recon_before: SceneReconstruction,
    alignment: SceneAlignment,
    config: ChangeDetectionConfig,
) -> SpatialChange:
    """
    Classify an object detected in AFTER but not matched in BEFORE.
    Could be ADDED or UNCERTAIN (if location wasn't observed before).
    """
    pos = obj.mean_position_3d
    
    if pos is None:
        return _make_uncertain_change(
            obj, "after",
            "Object position could not be estimated in 3D"
        )
    
    # Transform position to before's frame for visibility check
    R = alignment.transform[:3, :3]
    t = alignment.transform[:3, 3]
    pos_in_before_frame = R @ pos + t
    
    # Check if position was observed in before recording
    visibility = _check_visibility_in_cloud(pos_in_before_frame, recon_before)
    
    if visibility in ("observed", "partial"):
        det_conf = obj.mean_confidence if obj.mean_confidence > 0 else 0.8
        base_conf = min(det_conf * alignment.confidence, 0.85)
        confidence = base_conf * 0.8 if visibility == "partial" else base_conf
        
        if confidence < config.min_confidence:
            return _make_uncertain_change(
                obj, "after",
                "Low confidence in detection, alignment, or visibility"
            )
        
        return SpatialChange(
            change_id=f"chg_{uuid.uuid4().hex[:6]}",
            change_type=ChangeType.ADDED,
            object_label=obj.label,
            confidence=round(confidence, 3),
            old_position=None,
            new_position=pos,
            displacement=None,
            evidence=[
                f"Object '{obj.label}' detected in after recording",
                "Not matched to any object in before recording",
                "Object's location was observed in before recording (appeared empty)",
            ],
            visibility_status="observed",
            registration_confidence=alignment.confidence,
            limitations=[
                "Distances are in relative reconstruction units",
                "Object may have been present but undetected in before recording",
            ],
            after_object=obj,
        )
    else:
        return _make_uncertain_change(
            obj, "after",
            f"Object's location was {visibility} in the before recording"
        )


def _detect_geometric_changes(
    recon_before: SceneReconstruction,
    recon_after: SceneReconstruction,
    alignment: SceneAlignment,
    config: ChangeDetectionConfig,
) -> List[SpatialChange]:
    """
    Detect geometric changes by comparing point clouds.
    
    Finds regions where one cloud has points but the other doesn't,
    beyond what can be explained by different camera trajectories.
    """
    changes = []
    
    cloud_before = recon_before.point_cloud
    cloud_after = recon_after.point_cloud
    
    if cloud_before is None or cloud_after is None:
        return changes
    
    if cloud_before.num_points < 50 or cloud_after.num_points < 50:
        return changes
    
    # Transform after cloud to before's frame
    R = alignment.transform[:3, :3]
    t = alignment.transform[:3, 3]
    pts_after_aligned = (cloud_after.points @ R.T) + t
    
    # Build KD-trees
    tree_before = KDTree(cloud_before.points)
    tree_after = KDTree(pts_after_aligned)
    
    # Find points in before that have no nearby point in after
    dists_b, _ = tree_after.query(cloud_before.points)
    
    # Find points in after that have no nearby point in before
    dists_a, _ = tree_before.query(pts_after_aligned)
    
    # Threshold for significant geometric difference
    geo_threshold = config.position_threshold * 3
    
    # Cluster significant differences
    removed_pts = cloud_before.points[dists_b > geo_threshold]
    added_pts = pts_after_aligned[dists_a > geo_threshold]
    
    # Only report if there's a meaningful cluster
    if len(removed_pts) > 50:
        centroid = np.mean(removed_pts, axis=0)
        extent = np.max(removed_pts, axis=0) - np.min(removed_pts, axis=0)
        
        if np.max(extent) > config.position_threshold:
            changes.append(SpatialChange(
                change_id=f"geo_{uuid.uuid4().hex[:6]}",
                change_type=ChangeType.GEOMETRIC_CHANGE,
                confidence=round(min(len(removed_pts) / 500.0, 0.7) * alignment.confidence, 3),
                old_position=centroid,
                evidence=[
                    f"{len(removed_pts)} points in before cloud have no correspondence in after cloud",
                    f"Cluster centroid at {centroid.tolist()}",
                    f"Cluster extent: {extent.tolist()}",
                ],
                visibility_status="observed",
                registration_confidence=alignment.confidence,
                limitations=[
                    "Geometric change may reflect different camera coverage rather than physical change",
                    "Positions are in relative reconstruction units",
                ],
            ))
    
    if len(added_pts) > 50:
        centroid = np.mean(added_pts, axis=0)
        extent = np.max(added_pts, axis=0) - np.min(added_pts, axis=0)
        
        if np.max(extent) > config.position_threshold:
            changes.append(SpatialChange(
                change_id=f"geo_{uuid.uuid4().hex[:6]}",
                change_type=ChangeType.GEOMETRIC_CHANGE,
                confidence=round(min(len(added_pts) / 500.0, 0.7) * alignment.confidence, 3),
                new_position=centroid,
                evidence=[
                    f"{len(added_pts)} points in after cloud have no correspondence in before cloud",
                    f"Cluster centroid at {centroid.tolist()}",
                ],
                visibility_status="observed",
                registration_confidence=alignment.confidence,
                limitations=[
                    "Geometric change may reflect different camera coverage rather than physical change",
                    "Positions are in relative reconstruction units",
                ],
            ))
    
    return changes


def _check_visibility(
    position: np.ndarray,
    reconstruction: SceneReconstruction,
    alignment: SceneAlignment,
) -> str:
    """
    Check whether a 3D position was observed in a reconstruction.
    
    Returns: "observed", "partial", or "unobserved"
    """
    # Transform position to the reconstruction's frame
    # alignment.transform goes from scene2 → scene1
    # We need the inverse if checking in scene2
    try:
        T_inv = np.linalg.inv(alignment.transform)
        pos_in_recon = T_inv[:3, :3] @ position + T_inv[:3, 3]
    except np.linalg.LinAlgError:
        return "unobserved"
    
    return _check_visibility_in_cloud(pos_in_recon, reconstruction)


def _check_visibility_in_cloud(
    position: np.ndarray,
    reconstruction: SceneReconstruction,
) -> str:
    """
    Check if a position is near observed geometry in a point cloud.
    """
    cloud = reconstruction.point_cloud
    if cloud is None or cloud.num_points == 0:
        return "unobserved"
    
    tree = KDTree(cloud.points)
    dist, _ = tree.query(position.reshape(1, 3))
    dist = dist[0]
    
    if dist < 0.8:
        return "observed"
    elif dist < 2.0:
        return "partial"
    else:
        return "unobserved"


def _make_uncertain_change(
    obj: TrackedObject,
    recording: str,
    reason: str,
) -> SpatialChange:
    """Create an UNCERTAIN change record."""
    return SpatialChange(
        change_id=f"unc_{uuid.uuid4().hex[:6]}",
        change_type=ChangeType.UNCERTAIN,
        object_label=obj.label,
        confidence=0.1,
        old_position=obj.mean_position_3d if recording == "before" else None,
        new_position=obj.mean_position_3d if recording == "after" else None,
        evidence=[
            f"Object '{obj.label}' detected in {recording} recording",
            f"Classification uncertain: {reason}",
        ],
        visibility_status="uncertain",
        limitations=[reason],
        before_object=obj if recording == "before" else None,
        after_object=obj if recording == "after" else None,
    )
