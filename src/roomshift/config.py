"""
Centralized configuration for the RoomShift AI pipeline.

All configurable parameters in one place with sensible defaults
optimized for Apple M1 with 8 GB RAM.
"""
from dataclasses import dataclass, field
from typing import Optional
import os
import json


@dataclass
class VideoConfig:
    """Video ingestion settings."""
    max_frames: int = 30           # maximum frames to extract per video
    min_frames: int = 10           # minimum frames needed
    target_resolution: int = 640   # resize longest edge to this
    blur_threshold: float = 50.0   # Laplacian variance threshold for blur
    similarity_threshold: float = 0.92  # skip frames too similar to previous
    min_frame_interval: float = 0.3     # minimum seconds between frames


@dataclass
class ReconstructionConfig:
    """3D reconstruction settings."""
    feature_type: str = "sift"     # "sift" or "orb"
    max_features: int = 2000       # features per image
    match_ratio: float = 0.7      # Lowe's ratio test threshold
    min_matches: int = 30          # minimum matches for pose estimation
    ransac_threshold: float = 3.0  # pixels
    depth_model: str = "midas_small"  # "midas_small", "midas_large", "dpt_hybrid"
    max_depth_frames: int = 15     # limit depth estimation to save memory
    points_per_depth: int = 5000   # points sampled from each depth map
    voxel_size: float = 0.02      # voxel downsampling size for point clouds


@dataclass
class AlignmentConfig:
    """Scene alignment settings."""
    cross_match_ratio: float = 0.75
    ransac_iterations: int = 5000
    ransac_threshold: float = 5.0  # pixels for fundamental matrix
    icp_threshold: float = 0.05   # ICP correspondence distance
    icp_iterations: int = 50
    min_inlier_ratio: float = 0.15
    min_confidence: float = 0.3    # below this, alignment is unreliable


@dataclass
class DetectionConfig:
    """Object detection settings."""
    model_name: str = "yolov8n.pt"  # nano model for speed
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    max_detections: int = 50
    detect_every_n: int = 1        # detect on every nth sampled frame
    device: str = "cpu"            # "cpu", "mps", or "cuda"


@dataclass
class ChangeDetectionConfig:
    """Change detection settings."""
    position_threshold: float = 0.15  # min displacement to count as moved (relative units)
    match_distance_threshold: float = 0.3  # max distance for object matching
    appearance_weight: float = 0.4    # weight for appearance vs position matching
    min_confidence: float = 0.25      # below this, classify as uncertain
    visibility_check: bool = True     # check if region was observed
    geometric_comparison: bool = True # compare point cloud geometry


@dataclass
class VisualizationConfig:
    """Visualization settings."""
    point_size: int = 3
    change_marker_size: int = 8
    moved_color: str = "#FF6B35"     # orange
    added_color: str = "#2ECC71"     # green
    removed_color: str = "#E74C3C"   # red
    uncertain_color: str = "#95A5A6" # gray
    geometric_color: str = "#3498DB" # blue
    background_color: str = "#1a1a2e"
    show_cameras: bool = True
    show_connections: bool = True


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""
    video: VideoConfig = field(default_factory=VideoConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    change_detection: ChangeDetectionConfig = field(default_factory=ChangeDetectionConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    # Global settings
    output_dir: str = "outputs"
    cache_reconstructions: bool = True
    verbose: bool = False
    random_seed: int = 42

    @classmethod
    def from_file(cls, path: str) -> 'PipelineConfig':
        """Load config from JSON file."""
        with open(path) as f:
            data = json.load(f)
        config = cls()
        for section_name, section_data in data.items():
            if hasattr(config, section_name) and isinstance(section_data, dict):
                section = getattr(config, section_name)
                for key, value in section_data.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
            elif hasattr(config, section_name):
                setattr(config, section_name, section_data)
        return config

    def to_dict(self) -> dict:
        """Serialize config to dict."""
        result = {}
        for key, val in self.__dict__.items():
            if hasattr(val, '__dict__'):
                result[key] = val.__dict__
            else:
                result[key] = val
        return result

    def save(self, path: str):
        """Save config to JSON file."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def get_default_config() -> PipelineConfig:
    """Return default configuration optimized for M1 8GB."""
    return PipelineConfig()


def get_quality_config() -> PipelineConfig:
    """Return higher-quality configuration (needs more RAM/time)."""
    config = PipelineConfig()
    config.video.max_frames = 50
    config.video.target_resolution = 960
    config.reconstruction.max_features = 4000
    config.reconstruction.depth_model = "dpt_hybrid"
    config.reconstruction.points_per_depth = 10000
    config.detection.model_name = "yolov8s.pt"
    config.detection.confidence_threshold = 0.3
    return config
