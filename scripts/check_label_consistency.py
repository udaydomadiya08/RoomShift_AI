"""
Check object labels across reference and rescan in 3RScan.
"""
import json

with open('data/external/3rscan/3RScan.json') as f:
    scenes = json.load(f)
with open('data/external/3rscan/objects.json') as f:
    obj_data = json.load(f)

scan_objs = {s['scan']: s['objects'] for s in obj_data.get('scans', [])}

for scene in scenes:
    if scene.get('type') == 'validation':
        ref_id = scene['reference']
        for scan in scene.get('scans', []):
            res_id = scan.get('reference')
            if ref_id in scan_objs and res_id in scan_objs:
                ref_list = scan_objs[ref_id]
                res_list = scan_objs[res_id]
                ref_dict = {int(o['id']): o.get('label') for o in ref_list if 'id' in o}
                res_dict = {int(o['id']): o.get('label') for o in res_list if 'id' in o}
                
                label_mismatches = []
                for oid in ref_dict:
                    if oid in res_dict and ref_dict[oid] != res_dict[oid]:
                        label_mismatches.append((oid, ref_dict[oid], res_dict[oid]))
                        
                print(f"Scene {ref_id[:8]} -> {res_id[:8]}:")
                print(f"  Total ref objects: {len(ref_dict)}, rescan objects: {len(res_dict)}")
                print(f"  Shared IDs with mismatched labels: {len(label_mismatches)}")
                if label_mismatches:
                    print("  Sample mismatches:", label_mismatches[:5])
                break
        break
