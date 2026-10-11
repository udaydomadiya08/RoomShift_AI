"""
Audit object ID correspondence between reference scan and rescan in objects.json.
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
                ref_ids = set(int(o['id']) for o in ref_list if 'id' in o)
                res_ids = set(int(o['id']) for o in res_list if 'id' in o)
                common_ids = ref_ids.intersection(res_ids)
                ref_only = ref_ids - res_ids
                res_only = res_ids - ref_ids
                print(f"Scene: {ref_id[:8]} -> {res_id[:8]}")
                print(f"  Ref objects total: {len(ref_ids)}")
                print(f"  Rescan objects total: {len(res_ids)}")
                print(f"  Common instance IDs: {len(common_ids)}")
                print(f"  Ref-only instance IDs: {len(ref_only)}: {list(ref_only)[:10]}")
                print(f"  Rescan-only instance IDs: {len(res_only)}: {list(res_only)[:10]}")
                print(f"  scan['rigid']: {scan.get('rigid')}")
                print(f"  scan['removed']: {scan.get('removed')}")
                print(f"  scan['nonrigid']: {scan.get('nonrigid')}")
                
                # Check labels of ref_only
                ref_dict = {int(o['id']): o.get('label') for o in ref_list}
                print("  Labels of ref_only IDs:", [ref_dict[i] for i in list(ref_only)[:5]])
                break
        break
