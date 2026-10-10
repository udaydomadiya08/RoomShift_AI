"""
Analyze ground truth changes across all 3RScan validation and train scenes.
"""
import json
import numpy as np
import os

def analyze():
    meta_path = 'data/external/3rscan/3RScan.json'
    objects_path = 'data/external/3rscan/objects.json'
    
    with open(meta_path) as f:
        scenes = json.load(f)
    with open(objects_path) as f:
        obj_data = json.load(f)
        
    scan_objects = {s['scan']: {int(o['id']): o for o in s.get('objects', []) if 'id' in o}
                    for s in obj_data.get('scans', [])}
                    
    total_pairs = 0
    total_moved = 0
    total_removed = 0
    split_counts = {'train': 0, 'validation': 0, 'test': 0}
    displacement_list = []
    
    val_scene_summary = []
    
    for scene in scenes:
        split = scene.get('type', 'unknown')
        if split in split_counts:
            split_counts[split] += 1
            
        ref_id = scene['reference']
        ref_objs = scan_objects.get(ref_id, {})
        
        for scan in scene.get('scans', []):
            rescan_id = scan.get('reference')
            rescan_objs = scan_objects.get(rescan_id, {})
            total_pairs += 1
            
            rigid = scan.get('rigid', [])
            removed = scan.get('removed', [])
            
            pair_moved = 0
            for item in rigid:
                if isinstance(item, dict):
                    t = item.get('transform')
                    if t and len(t) == 16:
                        # 3RScan uses column-major 4x4 matrix format
                        mat = np.array(t).reshape((4, 4), order='F')
                        trans = mat[:3, 3]
                        disp = float(np.linalg.norm(trans))
                        displacement_list.append(disp)
                        pair_moved += 1
                elif isinstance(item, (int, str)):
                    pair_moved += 1
                    
            total_moved += pair_moved
            total_removed += len(removed)
            
            if split == 'validation' and (pair_moved > 0 or len(removed) > 0):
                val_scene_summary.append({
                    'reference': ref_id,
                    'rescan': rescan_id,
                    'moved_count': pair_moved,
                    'removed_count': len(removed),
                    'ref_objects_count': len(ref_objs),
                    'rescan_objects_count': len(rescan_objs)
                })

    print(f"Total reference-rescan pairs: {total_pairs}")
    print(f"Total documented MOVED changes: {total_moved}")
    print(f"Total documented REMOVED changes: {total_removed}")
    print(f"Split scene counts: {split_counts}")
    if displacement_list:
        print(f"Average rigid displacement magnitude: {np.mean(displacement_list):.3f} m")
        print(f"Median rigid displacement magnitude:  {np.median(displacement_list):.3f} m")
        print(f"Min / Max displacement: {np.min(displacement_list):.3f} m / {np.max(displacement_list):.3f} m")
        
    print(f"\nValidation scene pairs with changes: {len(val_scene_summary)}")
    for s in val_scene_summary[:5]:
        print(f"  Ref {s['reference'][:8]}.. -> Rescan {s['rescan'][:8]}..: "
              f"{s['moved_count']} moved, {s['removed_count']} removed, "
              f"objects: {s['ref_objects_count']} ref, {s['rescan_objects_count']} rescan")

if __name__ == '__main__':
    analyze()
