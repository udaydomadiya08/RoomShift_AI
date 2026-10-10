"""
Frame sampling module — intelligent extraction of useful frames from video.

Balances coverage, quality, and computational cost. Avoids redundant
nearly-identical frames while preserving temporal distribution.
"""
import logging
from typing import List, Optional

import cv2
import numpy as np

from .types import ExtractedFrame, FrameManifest, VideoMetadata
from .video_io import (
    validate_video, resize_frame, compute_blur_score,
    compute_frame_similarity
)
from .config import VideoConfig

logger = logging.getLogger(__name__)


def extract_frames(
    video_path: str,
    config: Optional[VideoConfig] = None,
) -> FrameManifest:
    """
    Extract a set of useful, non-redundant frames from a video.
    
    Strategy:
    1. Compute candidate frame indices evenly spaced across the video.
    2. Read each candidate frame.
    3. Check sharpness (reject blurry frames).
    4. Check similarity to previous accepted frame (reject near-duplicates).
    5. Resize to target resolution.
    6. Return manifest with metadata.
    """
    if config is None:
        config = VideoConfig()

    metadata = validate_video(video_path)
    
    # Compute candidate indices
    candidates = _compute_candidate_indices(metadata, config)
    logger.info(
        f"Sampling {len(candidates)} candidate frames from "
        f"{metadata.frame_count} total ({metadata.duration_seconds:.1f}s)"
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for frame extraction: {video_path}")

    frames: List[ExtractedFrame] = []
    sharp_frames: List[tuple] = []  # (idx, raw_frame, blur_score)
    prev_frame = None

    try:
        for idx in candidates:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, raw_frame = cap.read()
            if not ret or raw_frame is None:
                continue

            # Check blur
            blur_score = compute_blur_score(raw_frame)
            if blur_score < config.blur_threshold:
                logger.debug(f"Frame {idx}: rejected (blur={blur_score:.1f})")
                continue

            sharp_frames.append((idx, raw_frame, blur_score))

            # Check similarity to previous accepted frame
            if prev_frame is not None:
                sim = compute_frame_similarity(raw_frame, prev_frame)
                if sim > config.similarity_threshold:
                    logger.debug(f"Frame {idx}: rejected (similarity={sim:.3f})")
                    continue

            # Resize
            resized = resize_frame(raw_frame, config.target_resolution)

            # Compute timestamp
            timestamp = idx / metadata.fps if metadata.fps > 0 else 0.0

            frames.append(ExtractedFrame(
                frame_index=idx,
                timestamp=round(timestamp, 3),
                image=resized,
                source_video=video_path,
                quality_score=min(blur_score / 500.0, 1.0),  # normalize
            ))
            prev_frame = raw_frame

            if len(frames) >= config.max_frames:
                break

    finally:
        cap.release()

    # If similarity threshold was too aggressive and we have fewer than min_frames,
    # fallback to evenly spaced sharp frames
    if len(frames) < config.min_frames and len(sharp_frames) >= config.min_frames:
        logger.info(
            f"Similarity filter left only {len(frames)} frames; "
            f"falling back to {min(config.min_frames, len(sharp_frames))} sharp candidate frames."
        )
        sample_indices = np.linspace(0, len(sharp_frames) - 1, min(config.min_frames, len(sharp_frames)), dtype=int)
        frames = []
        for s_idx in sample_indices:
            idx, raw_frame, blur_score = sharp_frames[s_idx]
            resized = resize_frame(raw_frame, config.target_resolution)
            timestamp = idx / metadata.fps if metadata.fps > 0 else 0.0
            frames.append(ExtractedFrame(
                frame_index=idx,
                timestamp=round(timestamp, 3),
                image=resized,
                source_video=video_path,
                quality_score=min(blur_score / 500.0, 1.0),
            ))
    elif len(frames) < config.min_frames:
        logger.warning(
            f"Only extracted {len(frames)} frames (minimum: {config.min_frames}). "
            "Video may be too short, blurry, or have insufficient motion."
        )

    manifest = FrameManifest(
        video_path=video_path,
        metadata=metadata,
        frames=frames,
        sampling_strategy="uniform_with_quality_filter",
        total_extracted=len(frames),
    )
    logger.info(f"Extracted {len(frames)} frames from {video_path}")
    return manifest


def _compute_candidate_indices(
    metadata: VideoMetadata,
    config: VideoConfig,
) -> List[int]:
    """
    Compute evenly-spaced candidate frame indices.
    
    Respects min_frame_interval and samples more candidates than max_frames
    to allow for rejection of blurry/duplicate frames.
    """
    total = metadata.frame_count
    fps = metadata.fps

    # Minimum frame gap based on time interval
    min_gap = max(1, int(config.min_frame_interval * fps))

    # How many candidates can we have given the min gap?
    max_possible = total // min_gap

    # Sample more candidates than needed (2x) to account for rejections
    num_candidates = min(max_possible, config.max_frames * 2)
    num_candidates = max(num_candidates, config.min_frames)

    if num_candidates >= total:
        return list(range(0, total, max(1, min_gap)))

    # Evenly space candidates
    indices = np.linspace(0, total - 1, num_candidates, dtype=int).tolist()

    # Ensure minimum gap
    filtered = [indices[0]]
    for idx in indices[1:]:
        if idx - filtered[-1] >= min_gap:
            filtered.append(idx)

    return filtered
