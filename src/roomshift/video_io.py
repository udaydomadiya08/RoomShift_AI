"""
Video I/O module — validation, metadata extraction, and frame reading.

Handles variable frame rates, orientation metadata, different resolutions,
and decoding failures with clear error messages.
"""
import os
import logging
import subprocess
import json
from typing import Optional, Tuple

import cv2
import numpy as np

from .types import VideoMetadata

logger = logging.getLogger(__name__)


class VideoValidationError(Exception):
    """Raised when a video file cannot be used."""
    pass


def validate_video(path: str) -> VideoMetadata:
    """
    Validate a video file and extract metadata.
    
    Raises VideoValidationError with a clear message if the file is unusable.
    """
    if not os.path.exists(path):
        raise VideoValidationError(f"Video file not found: {path}")
    
    if os.path.getsize(path) == 0:
        raise VideoValidationError(f"Video file is empty: {path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise VideoValidationError(
            f"Cannot open video: {path}. "
            "Check that the file is a valid video and FFmpeg/codecs are installed."
        )

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

        if width == 0 or height == 0:
            raise VideoValidationError(f"Invalid video dimensions: {width}x{height}")
        
        if fps <= 0:
            logger.warning(f"Could not determine FPS for {path}, defaulting to 30")
            fps = 30.0

        if frame_count <= 0:
            # Try to count frames manually (slower)
            frame_count = _count_frames(cap)
            if frame_count <= 0:
                raise VideoValidationError(f"Video has no readable frames: {path}")

        duration = frame_count / fps if fps > 0 else 0.0

        # Try to get rotation from FFprobe
        rotation = _get_rotation(path)

        # Test that we can actually read a frame
        ret, frame = cap.read()
        if not ret or frame is None:
            raise VideoValidationError(f"Cannot read any frame from video: {path}")

        return VideoMetadata(
            path=os.path.abspath(path),
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_seconds=round(duration, 2),
            codec=codec,
            rotation=rotation,
        )
    finally:
        cap.release()


def read_frame(path: str, frame_index: int) -> Optional[np.ndarray]:
    """Read a single frame by index. Returns BGR image or None."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        return frame if ret else None
    finally:
        cap.release()


def read_frames_batch(path: str, indices: list) -> list:
    """Read multiple frames efficiently by seeking. Returns list of (index, frame) tuples."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    
    results = []
    sorted_indices = sorted(indices)
    try:
        for idx in sorted_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                results.append((idx, frame))
    finally:
        cap.release()
    return results


def resize_frame(frame: np.ndarray, max_size: int) -> np.ndarray:
    """Resize frame so longest edge is max_size, preserving aspect ratio."""
    h, w = frame.shape[:2]
    if max(h, w) <= max_size:
        return frame
    scale = max_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def compute_blur_score(frame: np.ndarray) -> float:
    """Compute Laplacian variance as a sharpness/blur metric. Higher = sharper."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def compute_frame_similarity(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    Compute structural similarity between two frames using histogram comparison.
    Returns value in [0, 1] where 1 = identical.
    """
    # Resize to common small size for fast comparison
    size = (64, 64)
    f1 = cv2.resize(frame1, size)
    f2 = cv2.resize(frame2, size)
    
    # Convert to HSV for better comparison
    h1 = cv2.cvtColor(f1, cv2.COLOR_BGR2HSV)
    h2 = cv2.cvtColor(f2, cv2.COLOR_BGR2HSV)
    
    # Compute histograms
    hist1 = cv2.calcHist([h1], [0, 1], None, [30, 32], [0, 180, 0, 256])
    hist2 = cv2.calcHist([h2], [0, 1], None, [30, 32], [0, 180, 0, 256])
    
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


def _count_frames(cap: cv2.VideoCapture) -> int:
    """Count frames by reading through the video."""
    count = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ret = cap.grab()
        if not ret:
            break
        count += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return count


def _get_rotation(path: str) -> int:
    """Get video rotation from FFprobe metadata."""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_streams', '-select_streams', 'v:0', path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            streams = data.get('streams', [])
            if streams:
                # Check side_data for rotation
                side_data = streams[0].get('side_data_list', [])
                for sd in side_data:
                    if 'rotation' in sd:
                        return int(sd['rotation'])
                # Check tags
                tags = streams[0].get('tags', {})
                if 'rotate' in tags:
                    return int(tags['rotate'])
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        pass
    return 0
