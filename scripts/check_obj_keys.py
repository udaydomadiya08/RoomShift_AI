"""
Inspect all keys and attributes in objects.json.
"""
import json

with open('data/external/3rscan/objects.json') as f:
    data = json.load(f)

keys = set()
for s in data.get('scans', [])[:50]:
    for obj in s.get('objects', []):
        keys.update(obj.keys())

print("All keys found on objects in objects.json:", sorted(list(keys)))
