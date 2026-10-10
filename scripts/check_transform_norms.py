"""
Check translation norms across all 3RScan global transforms.
"""
import json
import numpy as np

with open('data/external/3rscan/3RScan.json') as f:
    scenes = json.load(f)

norms = []
for scene in scenes:
    for scan in scene.get('scans', []):
        t = scan.get('transform')
        if t and len(t) == 16:
            mat = np.array(t).reshape((4, 4), order='F')
            norms.append(np.linalg.norm(mat[:3, 3]))

print(f"Total global transforms found: {len(norms)}")
print(f"Norm range: min={min(norms):.3f} m, max={max(norms):.3f} m, mean={np.mean(norms):.3f} m, median={np.median(norms):.3f} m")
