"""
Change reporting module — generates structured JSON and human-readable summaries.

Converts change detection results into clear reports with spatial locations,
confidence, supporting evidence, and limitations.
"""
import json
import logging
import os
from typing import List, Dict, Any
from datetime import datetime

from .types import SpatialChange, ChangeType, ChangeReport, SceneAlignment

logger = logging.getLogger(__name__)


def generate_json_report(report: ChangeReport, output_path: str) -> str:
    """
    Generate a structured JSON change report.
    
    Schema follows the specification in docs/CHANGE_DETECTION.md.
    """
    data = report.to_dict()
    data['generated_at'] = datetime.now().isoformat()
    data['schema_version'] = '1.0'
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    logger.info(f"JSON report saved to {output_path}")
    return output_path


def generate_text_summary(report: ChangeReport) -> str:
    """
    Generate a human-readable text summary of detected changes.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("ROOMSHIFT AI — SPATIAL CHANGE REPORT")
    lines.append("=" * 60)
    lines.append("")
    
    # Alignment info
    if report.alignment:
        lines.append(f"Scene Alignment Confidence: {report.alignment.confidence:.1%}")
        lines.append(f"Alignment Method: {report.alignment.method}")
        if not report.alignment.is_reliable:
            lines.append("⚠ WARNING: Scene alignment is unreliable.")
            lines.append("  Change classifications may be inaccurate.")
        lines.append("")
    
    # Summary counts
    counts = _count_by_type(report.changes)
    lines.append("CHANGE SUMMARY")
    lines.append("-" * 40)
    lines.append(f"  Moved objects:       {counts.get('moved', 0)}")
    lines.append(f"  Added objects:       {counts.get('added', 0)}")
    lines.append(f"  Removed objects:     {counts.get('removed', 0)}")
    lines.append(f"  Geometric changes:   {counts.get('geometric_change', 0)}")
    lines.append(f"  Uncertain:           {counts.get('uncertain', 0)}")
    lines.append(f"  Total:               {len(report.changes)}")
    lines.append("")
    
    # Individual changes
    if report.changes:
        lines.append("DETECTED CHANGES")
        lines.append("-" * 40)
        for i, change in enumerate(report.changes, 1):
            lines.append(f"\n{i}. {_change_description(change)}")
            lines.append(f"   Type: {change.change_type.value.upper()}")
            lines.append(f"   Confidence: {change.confidence:.1%}")
            
            if change.old_position is not None:
                pos_str = _format_position(change.old_position)
                lines.append(f"   Original position: {pos_str}")
            
            if change.new_position is not None:
                pos_str = _format_position(change.new_position)
                lines.append(f"   New position: {pos_str}")
            
            if change.displacement is not None:
                lines.append(
                    f"   Displacement: {change.displacement:.3f} "
                    f"({change.scale_status} units)"
                )
            
            if change.evidence:
                lines.append("   Evidence:")
                for ev in change.evidence:
                    lines.append(f"     • {ev}")
            
            if change.limitations:
                lines.append("   Limitations:")
                for lim in change.limitations:
                    lines.append(f"     ⚠ {lim}")
    else:
        lines.append("No changes detected.")
        lines.append("(This may indicate no physical changes, or insufficient")
        lines.append(" reconstruction/alignment quality to detect them.)")
    
    lines.append("")
    lines.append("-" * 40)
    lines.append(f"Processing time: {report.processing_time_seconds:.1f}s")
    lines.append(f"Pipeline version: {report.pipeline_version}")
    lines.append("")
    
    return "\n".join(lines)


def _change_description(change: SpatialChange) -> str:
    """Generate a natural-language description of a change."""
    label = change.object_label or "Unknown object"
    
    if change.change_type == ChangeType.MOVED:
        desc = f"A {label} was moved from its original position"
        if change.displacement is not None:
            desc += f" (displacement: {change.displacement:.3f} {change.scale_status} units)"
        return desc
    
    elif change.change_type == ChangeType.ADDED:
        desc = f"A {label} appears to have been added"
        if change.visibility_status == "observed":
            desc += " — its location was observed empty in the earlier recording"
        return desc
    
    elif change.change_type == ChangeType.REMOVED:
        desc = f"A {label} appears to have been removed"
        if change.visibility_status == "observed":
            desc += " — its original location was visible in the later recording"
        return desc
    
    elif change.change_type == ChangeType.GEOMETRIC_CHANGE:
        return "A spatial/geometric change was detected in the scene geometry"
    
    else:  # UNCERTAIN
        desc = f"A possible change involving a {label} was detected"
        desc += ", but insufficient evidence exists to classify it confidently"
        return desc


def _format_position(pos) -> str:
    """Format a 3D position for display."""
    if pos is None:
        return "unknown"
    return f"({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"


def _count_by_type(changes: List[SpatialChange]) -> Dict[str, int]:
    """Count changes by type."""
    counts = {}
    for c in changes:
        key = c.change_type.value
        counts[key] = counts.get(key, 0) + 1
    return counts
