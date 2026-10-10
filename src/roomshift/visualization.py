"""
3D Visualization module — creates interactive visualizations of scene reconstructions and changes.

Uses Plotly for browser-based 3D visualization with rotation, zoom, and inspection.
"""
import logging
import os
from typing import List, Optional, Dict

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .types import (
    PointCloud, SceneReconstruction, SceneAlignment,
    SpatialChange, ChangeType, ChangeReport,
)
from .config import VisualizationConfig

logger = logging.getLogger(__name__)


def create_scene_visualization(
    report: ChangeReport,
    config: Optional[VisualizationConfig] = None,
    output_path: Optional[str] = None,
) -> go.Figure:
    """
    Create a comprehensive 3D visualization of the scene and detected changes.
    
    Shows:
    - Before point cloud (blue tint)
    - After point cloud (orange tint), aligned
    - Change markers with color coding
    - Camera positions
    - Displacement arrows for moved objects
    """
    if config is None:
        config = VisualizationConfig()

    fig = go.Figure()

    # Add before point cloud
    if report.before_reconstruction and report.before_reconstruction.point_cloud:
        _add_point_cloud(
            fig, report.before_reconstruction.point_cloud,
            name="Before Scene", opacity=0.5, color_tint=(0.6, 0.7, 1.0)
        )

    # Add after point cloud (transformed to shared frame)
    if report.after_reconstruction and report.after_reconstruction.point_cloud and report.alignment:
        cloud_after = report.after_reconstruction.point_cloud
        if report.alignment.transform is not None:
            R = report.alignment.transform[:3, :3]
            t = report.alignment.transform[:3, 3]
            pts = (cloud_after.points @ R.T) + t
            aligned_cloud = PointCloud(
                points=pts,
                colors=cloud_after.colors,
            )
        else:
            aligned_cloud = cloud_after
        
        _add_point_cloud(
            fig, aligned_cloud,
            name="After Scene", opacity=0.5, color_tint=(1.0, 0.7, 0.5)
        )

    # Add camera positions
    if config.show_cameras:
        if report.before_reconstruction:
            _add_cameras(fig, report.before_reconstruction, "Before Cameras", "#4A90D9")
        if report.after_reconstruction and report.alignment:
            _add_cameras(fig, report.after_reconstruction, "After Cameras", "#E67E22",
                        transform=report.alignment.transform)

    # Add change markers
    _add_change_markers(fig, report.changes, config)

    # Layout
    fig.update_layout(
        title=dict(
            text="RoomShift AI — 3D Spatial Change Detection",
            font=dict(size=20, color='white'),
        ),
        scene=dict(
            bgcolor=config.background_color,
            xaxis=dict(title="X", color='white', gridcolor='#333'),
            yaxis=dict(title="Y", color='white', gridcolor='#333'),
            zaxis=dict(title="Z", color='white', gridcolor='#333'),
            aspectmode='data',
        ),
        paper_bgcolor='#0d0d1a',
        plot_bgcolor='#0d0d1a',
        font=dict(color='white'),
        legend=dict(
            bgcolor='rgba(30,30,50,0.8)',
            bordercolor='#555',
            font=dict(color='white'),
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        fig.write_html(output_path, include_plotlyjs='cdn')
        logger.info(f"3D visualization saved to {output_path}")

    return fig


def create_change_summary_figure(
    report: ChangeReport,
    config: Optional[VisualizationConfig] = None,
) -> go.Figure:
    """Create a summary figure showing change statistics."""
    if config is None:
        config = VisualizationConfig()

    # Count changes by type
    type_counts = {}
    for c in report.changes:
        key = c.change_type.value
        type_counts[key] = type_counts.get(key, 0) + 1

    colors_map = {
        'moved': config.moved_color,
        'added': config.added_color,
        'removed': config.removed_color,
        'geometric_change': config.geometric_color,
        'uncertain': config.uncertain_color,
    }

    labels = list(type_counts.keys())
    values = list(type_counts.values())
    colors = [colors_map.get(l, '#999') for l in labels]
    display_labels = [l.replace('_', ' ').title() for l in labels]

    fig = go.Figure(data=[go.Bar(
        x=display_labels,
        y=values,
        marker_color=colors,
        text=values,
        textposition='auto',
    )])

    fig.update_layout(
        title="Detected Changes Summary",
        xaxis_title="Change Type",
        yaxis_title="Count",
        paper_bgcolor='#0d0d1a',
        plot_bgcolor='#1a1a2e',
        font=dict(color='white'),
        yaxis=dict(gridcolor='#333'),
    )

    return fig


def _add_point_cloud(
    fig: go.Figure,
    cloud: PointCloud,
    name: str,
    opacity: float = 0.6,
    color_tint: tuple = (1, 1, 1),
    max_points: int = 15000,
):
    """Add a point cloud to the figure."""
    if cloud is None or cloud.points is None or len(cloud.points) == 0:
        return

    pts = cloud.points
    colors = cloud.colors

    # Subsample if too many points
    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        pts = pts[idx]
        if colors is not None:
            colors = colors[idx]

    # Apply color tint
    if colors is not None:
        r = (colors[:, 0] * color_tint[0] * 255).astype(int).clip(0, 255)
        g = (colors[:, 1] * color_tint[1] * 255).astype(int).clip(0, 255)
        b = (colors[:, 2] * color_tint[2] * 255).astype(int).clip(0, 255)
        color_strs = [f'rgb({r[i]},{g[i]},{b[i]})' for i in range(len(r))]
    else:
        tr, tg, tb = [int(c * 180) for c in color_tint]
        color_strs = f'rgb({tr},{tg},{tb})'

    fig.add_trace(go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode='markers',
        marker=dict(size=2, color=color_strs, opacity=opacity),
        name=name,
        hoverinfo='skip',
    ))


def _add_cameras(
    fig: go.Figure,
    reconstruction: SceneReconstruction,
    name: str,
    color: str,
    transform: Optional[np.ndarray] = None,
):
    """Add camera positions as markers."""
    positions = []
    for pose in reconstruction.camera_poses:
        center = pose.camera_center.flatten()
        if transform is not None:
            R = transform[:3, :3]
            t = transform[:3, 3]
            center = R @ center + t
        positions.append(center)

    if not positions:
        return

    positions = np.array(positions)
    fig.add_trace(go.Scatter3d(
        x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
        mode='markers+lines',
        marker=dict(size=5, color=color, symbol='diamond'),
        line=dict(color=color, width=2),
        name=name,
        hovertemplate='Camera %{pointNumber}<extra></extra>',
    ))


def _add_change_markers(
    fig: go.Figure,
    changes: List[SpatialChange],
    config: VisualizationConfig,
):
    """Add change detection markers to the figure."""
    color_map = {
        ChangeType.MOVED: config.moved_color,
        ChangeType.ADDED: config.added_color,
        ChangeType.REMOVED: config.removed_color,
        ChangeType.GEOMETRIC_CHANGE: config.geometric_color,
        ChangeType.UNCERTAIN: config.uncertain_color,
    }

    for change in changes:
        color = color_map.get(change.change_type, '#999')
        label = change.object_label or change.change_type.value

        # Old position marker
        if change.old_position is not None:
            pos = change.old_position
            fig.add_trace(go.Scatter3d(
                x=[pos[0]], y=[pos[1]], z=[pos[2]],
                mode='markers+text',
                marker=dict(
                    size=config.change_marker_size,
                    color=color,
                    symbol='x' if change.change_type == ChangeType.REMOVED else 'circle',
                    opacity=0.9,
                    line=dict(width=2, color='white'),
                ),
                text=[f"{label} (before)"],
                textposition='top center',
                textfont=dict(size=10, color=color),
                name=f"{change.change_type.value}: {label}",
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    f"Type: {change.change_type.value}<br>"
                    f"Confidence: {change.confidence:.0%}<br>"
                    f"Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
                    "<extra></extra>"
                ),
                showlegend=True,
            ))

        # New position marker
        if change.new_position is not None:
            pos = change.new_position
            fig.add_trace(go.Scatter3d(
                x=[pos[0]], y=[pos[1]], z=[pos[2]],
                mode='markers+text',
                marker=dict(
                    size=config.change_marker_size,
                    color=color,
                    symbol='circle',
                    opacity=0.9,
                    line=dict(width=2, color='white'),
                ),
                text=[f"{label} (after)"],
                textposition='top center',
                textfont=dict(size=10, color=color),
                name=f"{change.change_type.value}: {label} (new)",
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    f"Type: {change.change_type.value}<br>"
                    f"Confidence: {change.confidence:.0%}<br>"
                    f"Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
                    "<extra></extra>"
                ),
                showlegend=False,
            ))

        # Displacement arrow for moved objects
        if (change.change_type == ChangeType.MOVED and
            change.old_position is not None and change.new_position is not None):
            _add_displacement_arrow(
                fig, change.old_position, change.new_position, color
            )


def _add_displacement_arrow(fig, start, end, color):
    """Add a line showing displacement between old and new position."""
    fig.add_trace(go.Scatter3d(
        x=[start[0], end[0]],
        y=[start[1], end[1]],
        z=[start[2], end[2]],
        mode='lines',
        line=dict(color=color, width=4, dash='dash'),
        showlegend=False,
        hoverinfo='skip',
    ))
