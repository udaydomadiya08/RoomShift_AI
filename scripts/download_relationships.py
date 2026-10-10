"""
Download relationships.json from 3DSSG.
"""
import urllib.request
import os

url = 'http://campar.in.tum.de/public_datasets/3DSSG/3DSSG/relationships.json'
dest = 'data/external/3rscan/relationships.json'

if not os.path.exists(dest):
    print("Downloading relationships.json...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
        with open(dest, 'wb') as f:
            f.write(data)
    print(f"Downloaded relationships.json: {len(data) / 1024 / 1024:.2f} MB")
else:
    print("Already exists:", os.path.getsize(dest))
