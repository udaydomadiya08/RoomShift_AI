"""
Synthetic test data generator for RoomShift AI.

Creates simple synthetic videos with known objects for testing the pipeline
without requiring real phone recordings.
"""
import os
import cv2
import numpy as np
import json
from typing import List, Tuple


def create_synthetic_room_frame(
    width: int = 640,
    height: int = 480,
    objects: List[dict] = None,
    camera_offset: Tuple[int, int] = (0, 0),
    background_color: Tuple[int, int, int] = (200, 200, 200),
) -> np.ndarray:
    """
    Create a synthetic room frame with colored rectangular objects.
    
    Each object dict: {
        'label': str, 'x': int, 'y': int, 'w': int, 'h': int,
        'color': (B, G, R)
    }
    """
    frame = np.full((height, width, 3), background_color, dtype=np.uint8)
    
    # Draw floor/wall lines for texture (helps feature matching)
    for i in range(0, width, 40):
        cv2.line(frame, (i + camera_offset[0] % 40, 0), 
                (i + camera_offset[0] % 40, height), (180, 180, 180), 1)
    for i in range(0, height, 40):
        cv2.line(frame, (0, i + camera_offset[1] % 40), 
                (width, i + camera_offset[1] % 40), (180, 180, 180), 1)
    
    # Add texture noise for feature detection
    noise = np.random.randint(0, 15, (height, width, 3), dtype=np.uint8)
    frame = cv2.add(frame, noise)
    
    # Draw objects
    if objects:
        for obj in objects:
            x = obj['x'] + camera_offset[0]
            y = obj['y'] + camera_offset[1]
            w, h = obj['w'], obj['h']
            color = obj.get('color', (0, 0, 255))
            
            # Object body
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
            # Border
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
            # Label
            cv2.putText(frame, obj.get('label', '?'), (x + 5, y + h // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame


def create_synthetic_video(
    output_path: str,
    objects: List[dict],
    num_frames: int = 90,
    fps: int = 30,
    width: int = 640,
    height: int = 480,
    camera_motion: str = "pan",  # "pan", "static", "orbit"
) -> str:
    """
    Create a synthetic video with known objects and camera motion.
    
    Returns the path to the created video.
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video writer for {output_path}")
    
    for i in range(num_frames):
        t = i / num_frames
        
        if camera_motion == "pan":
            # Slow horizontal pan
            offset_x = int(50 * np.sin(t * 2 * np.pi))
            offset_y = int(20 * np.sin(t * np.pi))
        elif camera_motion == "orbit":
            # Circular orbit
            offset_x = int(40 * np.cos(t * 2 * np.pi))
            offset_y = int(40 * np.sin(t * 2 * np.pi))
        else:
            offset_x = 0
            offset_y = 0
        
        frame = create_synthetic_room_frame(
            width, height, objects, (offset_x, offset_y)
        )
        writer.write(frame)
    
    writer.release()
    return output_path


def create_test_scenario(
    output_dir: str,
    scenario: str = "moved",
) -> dict:
    """
    Create a complete test scenario with before/after videos and ground truth.
    
    Scenarios:
    - "no_change": Same scene, different camera paths
    - "moved": One object moved
    - "added": One object added
    - "removed": One object removed
    - "multiple": Multiple simultaneous changes
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Base objects (always present in "before")
    base_objects = [
        {'label': 'chair', 'x': 100, 'y': 200, 'w': 80, 'h': 100, 'color': (139, 69, 19)},
        {'label': 'table', 'x': 300, 'y': 150, 'w': 120, 'h': 80, 'color': (42, 42, 165)},
        {'label': 'box', 'x': 450, 'y': 300, 'w': 60, 'h': 60, 'color': (0, 165, 255)},
    ]
    
    gt_changes = []
    
    if scenario == "no_change":
        after_objects = list(base_objects)
        gt_changes = [
            {"object_label": "chair", "change_type": "none"},
            {"object_label": "table", "change_type": "none"},
        ]
        
    elif scenario == "moved":
        after_objects = list(base_objects)
        # Move the chair
        after_objects[0] = dict(after_objects[0])
        after_objects[0]['x'] = 200
        after_objects[0]['y'] = 300
        gt_changes = [
            {"object_label": "chair", "change_type": "moved", 
             "approximate_position": [200, 300, 0]},
        ]
        
    elif scenario == "added":
        after_objects = list(base_objects)
        after_objects.append(
            {'label': 'bag', 'x': 500, 'y': 100, 'w': 50, 'h': 70, 'color': (0, 128, 0)}
        )
        gt_changes = [
            {"object_label": "bag", "change_type": "added",
             "approximate_position": [500, 100, 0]},
        ]
        
    elif scenario == "removed":
        after_objects = [obj for obj in base_objects if obj['label'] != 'box']
        gt_changes = [
            {"object_label": "box", "change_type": "removed",
             "approximate_position": [450, 300, 0]},
        ]
        
    elif scenario == "multiple":
        after_objects = list(base_objects)
        # Move chair
        after_objects[0] = dict(after_objects[0])
        after_objects[0]['x'] = 200
        # Remove box
        after_objects = [o for o in after_objects if o.get('label') != 'box']
        # Add lamp
        after_objects.append(
            {'label': 'lamp', 'x': 50, 'y': 50, 'w': 40, 'h': 100, 'color': (0, 255, 255)}
        )
        gt_changes = [
            {"object_label": "chair", "change_type": "moved"},
            {"object_label": "box", "change_type": "removed"},
            {"object_label": "lamp", "change_type": "added"},
        ]
    else:
        after_objects = list(base_objects)
    
    # Create videos
    before_path = os.path.join(output_dir, "before.mp4")
    after_path = os.path.join(output_dir, "after.mp4")
    
    create_synthetic_video(before_path, base_objects, camera_motion="pan")
    create_synthetic_video(after_path, after_objects, camera_motion="orbit")
    
    # Save ground truth
    gt_path = os.path.join(output_dir, "ground_truth.json")
    gt_data = {
        "scenario": scenario,
        "changes": gt_changes,
        "before_objects": [{"label": o["label"], "position": [o["x"], o["y"]]} for o in base_objects],
        "after_objects": [{"label": o["label"], "position": [o["x"], o["y"]]} for o in after_objects],
    }
    with open(gt_path, 'w') as f:
        json.dump(gt_data, f, indent=2)
    
    return {
        'before': before_path,
        'after': after_path,
        'ground_truth': gt_path,
        'scenario': scenario,
    }


def create_all_test_scenarios(base_dir: str = "examples") -> List[dict]:
    """Create all test scenarios."""
    scenarios = []
    for name in ["no_change", "moved", "added", "removed", "multiple"]:
        output_dir = os.path.join(base_dir, name)
        info = create_test_scenario(output_dir, name)
        scenarios.append(info)
    return scenarios
