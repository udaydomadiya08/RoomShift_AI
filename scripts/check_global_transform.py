"""
Inspect 3RScan global transformation matrix formatting and translation units.
"""
import json
import numpy as np

def check():
    with open('data/external/3rscan/3RScan.json') as f:
        scenes = json.load(f)
        
    for scene in scenes:
        for scan in scene.get('scans', []):
            t = scan.get('transform')
            if t and len(t) == 16:
                mat = np.array(t).reshape((4, 4), order='F')
                print("Scene Ref:", scene['reference'])
                print("Rescan:", scan['reference'])
                print("Raw matrix (col-major):")
                print(mat)
                trans = mat[:3, 3]
                print(f"Translation vector: {trans}")
                print(f"Norm in raw units: {np.linalg.norm(trans):.2f}")
                print(f"If millimeters -> meters: {np.linalg.norm(trans) * 0.001:.3f} m")
                return

if __name__ == '__main__':
    check()
