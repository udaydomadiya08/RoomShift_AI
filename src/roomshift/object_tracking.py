"""
Object tracking module — associates detections across frames within a recording.

Groups detections of the same object across multiple frames using
spatial proximity, label consistency, and appearance similarity.
"""
import logging
from typing import List, Optional, Dict
from collections import defaultdict
import uuid

import numpy as np

from .types import ObjectDetection, TrackedObject

logger = logging.getLogger(__name__)


def track_objects(
    detections: List[ObjectDetection],
    iou_threshold: float = 0.3,
    max_frame_gap: int = 5,
) -> List[TrackedObject]:
    """
    Associate detections across frames into tracked objects.
    
    Uses a simple greedy strategy:
    1. Group detections by frame.
    2. For each new frame, match detections to existing tracks via IoU and label.
    3. Create new tracks for unmatched detections.
    4. Compute mean 3D position from all associated detections.
    """
    if not detections:
        return []

    # Group by frame
    by_frame: Dict[int, List[ObjectDetection]] = defaultdict(list)
    for d in detections:
        by_frame[d.frame_index].append(d)
    
    sorted_frames = sorted(by_frame.keys())
    
    # Active tracks: track_id -> (last_frame, TrackedObject)
    active_tracks: Dict[str, tuple] = {}
    finished_tracks: List[TrackedObject] = []
    
    for frame_idx in sorted_frames:
        frame_dets = by_frame[frame_idx]
        matched_tracks = set()
        matched_dets = set()
        
        # Try to match each detection to existing tracks
        for det in frame_dets:
            best_track_id = None
            best_score = 0.0
            
            for track_id, (last_frame, track) in active_tracks.items():
                if track_id in matched_tracks:
                    continue
                if frame_idx - last_frame > max_frame_gap:
                    continue
                
                # Score = label match + IoU with last detection
                last_det = track.detections[-1]
                
                label_match = 1.0 if det.label == last_det.label else 0.0
                iou = _compute_iou(det.bbox, last_det.bbox)
                
                score = 0.5 * label_match + 0.5 * iou
                
                if score > best_score and score > iou_threshold * 0.5:
                    best_score = score
                    best_track_id = track_id
            
            if best_track_id is not None:
                active_tracks[best_track_id][1].detections.append(det)
                active_tracks[best_track_id] = (frame_idx, active_tracks[best_track_id][1])
                matched_tracks.add(best_track_id)
                matched_dets.add(id(det))
            else:
                # Start new track
                track_id = f"obj_{uuid.uuid4().hex[:6]}"
                track = TrackedObject(
                    track_id=track_id,
                    label=det.label,
                    detections=[det],
                )
                active_tracks[track_id] = (frame_idx, track)
                matched_tracks.add(track_id)
        
        # Close tracks that haven't been updated
        to_close = []
        for track_id, (last_frame, track) in active_tracks.items():
            if frame_idx - last_frame > max_frame_gap:
                to_close.append(track_id)
        
        for tid in to_close:
            _, track = active_tracks.pop(tid)
            _finalize_track(track)
            finished_tracks.append(track)
    
    # Close remaining active tracks
    for _, (_, track) in active_tracks.items():
        _finalize_track(track)
        finished_tracks.append(track)
    
    logger.info(f"Tracked {len(finished_tracks)} objects from {len(detections)} detections")
    return finished_tracks


def _finalize_track(track: TrackedObject):
    """Compute summary statistics for a completed track."""
    if not track.detections:
        return
    
    # Mean confidence
    track.mean_confidence = np.mean([d.confidence for d in track.detections])
    
    # Most common label
    from collections import Counter
    labels = [d.label for d in track.detections]
    track.label = Counter(labels).most_common(1)[0][0]
    
    # Mean 3D position
    positions = [d.position_3d for d in track.detections if d.position_3d is not None]
    if positions:
        valid = [p for p in positions if np.all(np.isfinite(p))]
        if valid:
            track.mean_position_3d = np.mean(valid, axis=0)


def _compute_iou(bbox1: tuple, bbox2: tuple) -> float:
    """Compute IoU between two bounding boxes (x1, y1, x2, y2)."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def match_objects_across_recordings(
    objects1: List[TrackedObject],
    objects2: List[TrackedObject],
    transform: np.ndarray,
    position_threshold: float = 0.3,
    appearance_weight: float = 0.4,
) -> List[tuple]:
    """
    Match tracked objects between two recordings.
    
    Uses label matching, 3D position proximity (after alignment transform),
    and optionally appearance similarity.
    
    Returns list of (obj1, obj2, match_score) tuples.
    """
    if not objects1 or not objects2:
        return []
    
    matches = []
    used2 = set()
    
    # Transform scene2 positions to scene1 frame
    R = transform[:3, :3]
    t = transform[:3, 3]
    
    for obj1 in objects1:
        if obj1.mean_position_3d is None:
            continue
        
        best_match = None
        best_score = 0.0
        
        for j, obj2 in enumerate(objects2):
            if j in used2:
                continue
            if obj2.mean_position_3d is None:
                continue
            
            # Label similarity (must match identical class label)
            if obj1.label != obj2.label:
                continue
            label_score = 1.0

            # Position proximity (after transform)
            pos2_transformed = R @ obj2.mean_position_3d + t
            distance = np.linalg.norm(obj1.mean_position_3d - pos2_transformed)
            if distance > position_threshold:
                continue
            position_score = max(0.0, 1.0 - distance / position_threshold)

            # Combined score
            score = (1.0 - appearance_weight) * position_score + appearance_weight * label_score

            if score > best_score and score > 0.3:
                best_score = score
                best_match = (obj1, obj2, score, distance, j)
        
        if best_match is not None:
            obj1, obj2, score, dist, j = best_match
            matches.append((obj1, obj2, score))
            used2.add(j)
    
    logger.info(f"Matched {len(matches)} objects across recordings")
    return matches
