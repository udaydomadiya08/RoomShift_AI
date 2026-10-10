"""
Inspect 3RScan downloaded metadata and ground truth objects.
"""
import json
import os

def main():
    objects_path = 'data/external/3rscan/objects.json'
    meta_path = 'data/external/3rscan/3RScan.json'
    
    with open(objects_path) as f:
        obj_data = json.load(f)
    with open(meta_path) as f:
        rscan_data = json.load(f)
        
    print(f"Total scans with objects in objects.json: {len(obj_data.get('scans', []))}")
    
    first_scan = obj_data['scans'][0]
    print(f"Sample scan ID: {first_scan.get('scan')}")
    print(f"Objects in sample scan: {len(first_scan.get('objects', []))}")
    if first_scan.get('objects'):
        print("Sample object:", json.dumps(first_scan['objects'][0], indent=2))
        
    # Check alignment between 3RScan.json and objects.json
    scan_map = {s['scan']: s for s in obj_data['scans']}
    
    # Find a pair with verified rigid changes
    found_pair = None
    for entry in rscan_data:
        ref_id = entry['reference']
        for scan in entry.get('scans', []):
            rescan_id = scan.get('reference')
            rigid_changes = scan.get('rigid', [])
            if rigid_changes and isinstance(rigid_changes[0], dict):
                if ref_id in scan_map and rescan_id in scan_map:
                    found_pair = (entry, scan)
                    break
        if found_pair:
            break
            
    if found_pair:
        entry, scan = found_pair
        ref_id = entry['reference']
        rescan_id = scan.get('reference')
        print(f"\nFound paired scan with 3D object changes:")
        print(f"Scene type: {entry.get('type')}")
        print(f"Reference scan: {ref_id} ({len(scan_map[ref_id]['objects'])} objects)")
        print(f"Rescan: {rescan_id} ({len(scan_map[rescan_id]['objects'])} objects)")
        print(f"Global transform: {scan.get('transform') is not None}")
        print(f"Number of rigid changes: {len(scan.get('rigid', []))}")
        print("First rigid change:", json.dumps(scan['rigid'][0], indent=2))

if __name__ == '__main__':
    main()
