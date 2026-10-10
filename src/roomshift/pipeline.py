"""
Pipeline module — end-to-end orchestration of the RoomShift AI processing pipeline.

Coordinates video ingestion, reconstruction, alignment, detection,
change classification, reporting, and visualization.
"""
import logging
import os
import time
from typing import Optional

import numpy as np

from .config import PipelineConfig, get_default_config
from .types import ChangeReport, SceneAlignment
from .frame_sampling import extract_frames
from .reconstruction import reconstruct_scene
from .scene_alignment import align_scenes, transform_point_cloud
from .object_detection import detect_objects_in_frames
from .object_tracking import track_objects, match_objects_across_recordings
from .change_detection import detect_changes
from .change_reporting import generate_json_report, generate_text_summary
from .visualization import create_scene_visualization, create_change_summary_figure

logger = logging.getLogger(__name__)


def run_pipeline(
    before_video: str,
    after_video: str,
    output_dir: str = "outputs",
    config: Optional[PipelineConfig] = None,
) -> ChangeReport:
    """
    Run the complete RoomShift AI pipeline.
    
    Steps:
    1. Extract frames from both videos
    2. Reconstruct 3D scenes
    3. Detect objects in both recordings
    4. Align scenes into shared coordinate frame
    5. Match objects across recordings
    6. Detect spatial changes
    7. Generate reports and visualizations
    
    Returns a ChangeReport with all results.
    """
    if config is None:
        config = get_default_config()
    
    # Set random seed
    np.random.seed(config.random_seed)
    
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()
    
    # ── Step 1: Frame Extraction ──────────────────────────────
    logger.info("=" * 50)
    logger.info("STEP 1: Extracting frames")
    logger.info("=" * 50)
    
    manifest_before = extract_frames(before_video, config.video)
    manifest_after = extract_frames(after_video, config.video)
    
    logger.info(
        f"Extracted: {manifest_before.total_extracted} frames (before), "
        f"{manifest_after.total_extracted} frames (after)"
    )
    
    # ── Step 2: 3D Reconstruction ─────────────────────────────
    logger.info("=" * 50)
    logger.info("STEP 2: 3D Reconstruction")
    logger.info("=" * 50)
    
    recon_before = reconstruct_scene(manifest_before, config.reconstruction)
    recon_after = reconstruct_scene(manifest_after, config.reconstruction)
    
    logger.info(
        f"Reconstruction quality: before={recon_before.quality_score:.2f}, "
        f"after={recon_after.quality_score:.2f}"
    )
    
    # ── Step 3: Object Detection ──────────────────────────────
    logger.info("=" * 50)
    logger.info("STEP 3: Object Detection")
    logger.info("=" * 50)
    
    detections_before = detect_objects_in_frames(
        manifest_before.frames, recon_before, config.detection
    )
    detections_after = detect_objects_in_frames(
        manifest_after.frames, recon_after, config.detection
    )
    
    # Track objects within each recording
    objects_before = track_objects(detections_before)
    objects_after = track_objects(detections_after)
    
    logger.info(
        f"Tracked objects: {len(objects_before)} (before), {len(objects_after)} (after)"
    )
    
    # ── Step 4: Scene Alignment ───────────────────────────────
    logger.info("=" * 50)
    logger.info("STEP 4: Scene Alignment")
    logger.info("=" * 50)
    
    alignment = align_scenes(
        recon_before, recon_after,
        manifest_before.frames, manifest_after.frames,
        config.alignment, config.reconstruction,
    )
    
    # ── Step 5: Cross-Recording Object Matching ───────────────
    logger.info("=" * 50)
    logger.info("STEP 5: Object Matching")
    logger.info("=" * 50)
    
    object_matches = match_objects_across_recordings(
        objects_before, objects_after,
        alignment.transform,
        config.change_detection.match_distance_threshold,
        config.change_detection.appearance_weight,
    )
    
    # ── Step 6: Change Detection ──────────────────────────────
    logger.info("=" * 50)
    logger.info("STEP 6: Change Detection")
    logger.info("=" * 50)
    
    changes = detect_changes(
        recon_before, recon_after,
        objects_before, objects_after,
        object_matches, alignment,
        config.change_detection,
    )
    
    elapsed = time.time() - start_time
    
    # ── Build Report ──────────────────────────────────────────
    report = ChangeReport(
        changes=changes,
        alignment=alignment,
        before_reconstruction=recon_before,
        after_reconstruction=recon_after,
        processing_time_seconds=elapsed,
        metadata={
            'before_video': before_video,
            'after_video': after_video,
            'config': str(config.to_dict()),
        },
    )
    
    # ── Step 7: Generate Outputs ──────────────────────────────
    logger.info("=" * 50)
    logger.info("STEP 7: Generating outputs")
    logger.info("=" * 50)
    
    # JSON report
    json_path = os.path.join(output_dir, "change_report.json")
    generate_json_report(report, json_path)
    
    # Text summary
    summary = generate_text_summary(report)
    summary_path = os.path.join(output_dir, "change_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(summary)
    logger.info(f"Text summary saved to {summary_path}")
    print("\n" + summary)
    
    # 3D visualization
    try:
        viz_path = os.path.join(output_dir, "scene_3d.html")
        create_scene_visualization(report, config.visualization, viz_path)
        
        chart_path = os.path.join(output_dir, "changes_chart.html")
        chart_fig = create_change_summary_figure(report, config.visualization)
        chart_fig.write_html(chart_path, include_plotlyjs='cdn')
        logger.info(f"Change summary chart saved to {chart_path}")
    except Exception as e:
        logger.warning(f"Visualization generation failed: {e}")
    
    logger.info(f"Pipeline completed in {elapsed:.1f}s")
    logger.info(f"All outputs saved to {output_dir}/")
    
    return report
