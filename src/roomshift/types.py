"""
Typed data structures for the RoomShift AI pipeline.

All inter-module communication uses these well-defined types.
Coordinates use a right-handed system: X-right, Y-down, Z-forward (OpenCV convention).
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple
import numpy as np


# ─── Video & Frame Types ─────────────────────────────────────────────────────

@dataclass
class VideoMetadata:
    """Metadata extracted from a video file."""
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    codec: str = ""
    rotation: int = 0  # degrees from orientation metadata


@dataclass
class ExtractedFrame:
    """A frame extracted from a video with its provenance."""
    frame_index: int        # index in original video
    timestamp: float        # seconds from start
    image: np.ndarray       # BGR image (H, W, 3)
    source_video: str       # path to source video
    quality_score: float = 1.0  # blur/sharpness metric


@dataclass
class FrameManifest:
    """Collection of extracted frames from one video."""
    video_path: str
    metadata: VideoMetadata
    frames: List[ExtractedFrame]
    sampling_strategy: str = ""
    total_extracted: int = 0


# ─── Camera & Geometry Types ─────────────────────────────────────────────────

@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    dist_coeffs: Optional[np.ndarray] = None  # distortion coefficients

    def to_matrix(self) -> np.ndarray:
        """Return 3x3 intrinsic matrix."""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)


@dataclass
class CameraPose:
    """Camera extrinsic pose (world-to-camera transform)."""
    frame_index: int
    timestamp: float
    rotation: np.ndarray      # 3x3 rotation matrix
    translation: np.ndarray   # 3x1 translation vector
    confidence: float = 1.0

    @property
    def projection_matrix(self) -> np.ndarray:
        """Return 3x4 [R|t] matrix."""
        return np.hstack([self.rotation, self.translation.reshape(3, 1)])

    @property
    def camera_center(self) -> np.ndarray:
        """Return camera center in world coordinates."""
        return -self.rotation.T @ self.translation.reshape(3, 1)


# ─── Reconstruction Types ─────────────────────────────────────────────────────

@dataclass
class PointCloud:
    """3D point cloud with optional color and normals."""
    points: np.ndarray          # (N, 3)
    colors: Optional[np.ndarray] = None   # (N, 3) RGB [0,1]
    normals: Optional[np.ndarray] = None  # (N, 3)
    confidence: Optional[np.ndarray] = None  # (N,) per-point confidence

    @property
    def num_points(self) -> int:
        return self.points.shape[0] if self.points is not None else 0


@dataclass
class DepthMap:
    """Monocular depth estimation result."""
    depth: np.ndarray      # (H, W) relative depth values
    frame_index: int
    is_metric: bool = False  # whether depth is in real-world units
    scale_factor: float = 1.0  # multiply depth by this for metric scale
    model_name: str = "midas"


@dataclass
class SceneReconstruction:
    """Complete reconstruction of one recording."""
    video_path: str
    point_cloud: Optional[PointCloud]
    camera_poses: List[CameraPose]
    intrinsics: CameraIntrinsics
    depth_maps: List[DepthMap]
    scale_status: str = "unknown"  # "metric", "relative", "unknown"
    quality_score: float = 0.0
    coordinate_frame: str = "opencv"  # "opencv" = X-right, Y-down, Z-forward
    metadata: Dict[str, Any] = field(default_factory=dict)


# ─── Alignment Types ─────────────────────────────────────────────────────────

@dataclass
class SceneAlignment:
    """Registration result between two scene reconstructions."""
    transform: np.ndarray     # 4x4 transformation matrix (scene2 -> scene1)
    scale: float = 1.0        # relative scale factor
    inlier_count: int = 0
    inlier_ratio: float = 0.0
    fitness: float = 0.0      # ICP fitness score
    rmse: float = float('inf')
    confidence: float = 0.0   # overall alignment confidence [0, 1]
    method: str = ""
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_reliable(self) -> bool:
        """Whether alignment is reliable enough for change detection."""
        return self.confidence > 0.3


# ─── Object Detection Types ──────────────────────────────────────────────────

@dataclass
class ObjectDetection:
    """Single object detection in one frame."""
    detection_id: str
    frame_index: int
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) pixels
    mask: Optional[np.ndarray] = None  # binary mask
    position_3d: Optional[np.ndarray] = None  # estimated 3D centroid
    extent_3d: Optional[np.ndarray] = None    # 3D bounding extent
    embedding: Optional[np.ndarray] = None    # appearance embedding


@dataclass
class TrackedObject:
    """Object tracked across multiple frames of one recording."""
    track_id: str
    label: str
    detections: List[ObjectDetection]
    mean_confidence: float = 0.0
    mean_position_3d: Optional[np.ndarray] = None
    appearance_descriptor: Optional[np.ndarray] = None

    def best_detection(self) -> ObjectDetection:
        """Return highest-confidence detection."""
        return max(self.detections, key=lambda d: d.confidence)


# ─── Change Detection Types ──────────────────────────────────────────────────

class ChangeType(Enum):
    """Classification of spatial changes."""
    MOVED = "moved"
    ADDED = "added"
    REMOVED = "removed"
    GEOMETRIC_CHANGE = "geometric_change"
    UNCERTAIN = "uncertain"


@dataclass
class SpatialChange:
    """A detected spatial change between recordings."""
    change_id: str
    change_type: ChangeType
    object_label: Optional[str] = None
    confidence: float = 0.0
    old_position: Optional[np.ndarray] = None  # 3D position in shared frame
    new_position: Optional[np.ndarray] = None
    displacement: Optional[float] = None  # Euclidean distance (units depend on scale)
    coordinate_frame: str = "shared_reconstruction"
    scale_status: str = "relative"  # "metric" or "relative"
    evidence: List[str] = field(default_factory=list)
    before_timestamps: List[float] = field(default_factory=list)
    after_timestamps: List[float] = field(default_factory=list)
    visibility_status: str = "observed"  # "observed", "partial", "unobserved"
    registration_confidence: float = 0.0
    limitations: List[str] = field(default_factory=list)
    before_object: Optional[TrackedObject] = None
    after_object: Optional[TrackedObject] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dictionary."""
        return {
            "change_id": self.change_id,
            "change_type": self.change_type.value,
            "object_label": self.object_label,
            "confidence": round(self.confidence, 3),
            "old_position": self.old_position.tolist() if self.old_position is not None else None,
            "new_position": self.new_position.tolist() if self.new_position is not None else None,
            "displacement": round(self.displacement, 4) if self.displacement is not None else None,
            "coordinate_frame": self.coordinate_frame,
            "scale_status": self.scale_status,
            "evidence": self.evidence,
            "before_timestamps": self.before_timestamps,
            "after_timestamps": self.after_timestamps,
            "visibility_status": self.visibility_status,
            "registration_confidence": round(self.registration_confidence, 3),
            "limitations": self.limitations,
        }


@dataclass
class ChangeReport:
    """Complete change detection report."""
    changes: List[SpatialChange]
    alignment: Optional[SceneAlignment] = None
    before_reconstruction: Optional[SceneReconstruction] = None
    after_reconstruction: Optional[SceneReconstruction] = None
    processing_time_seconds: float = 0.0
    pipeline_version: str = "0.1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_version": self.pipeline_version,
            "processing_time_seconds": round(self.processing_time_seconds, 2),
            "alignment_confidence": round(self.alignment.confidence, 3) if self.alignment else None,
            "num_changes": len(self.changes),
            "changes": [c.to_dict() for c in self.changes],
            "metadata": self.metadata,
        }


# ─── Evaluation Types ─────────────────────────────────────────────────────────

@dataclass
class GroundTruthChange:
    """Ground truth annotation for evaluation."""
    object_label: str
    change_type: str  # "moved", "added", "removed", "none"
    approximate_position: Optional[List[float]] = None
    notes: str = ""


@dataclass
class EvaluationResult:
    """Metrics from evaluating against ground truth."""
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    per_category: Dict[str, Dict[str, float]] = field(default_factory=dict)
    details: List[Dict[str, Any]] = field(default_factory=list)
