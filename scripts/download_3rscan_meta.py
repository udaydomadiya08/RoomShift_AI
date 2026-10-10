"""
Download official 3RScan and 3DSSG metadata files.
"""
import urllib.request
import os
import sys

def download():
    os.makedirs('data/external/3rscan', exist_ok=True)
    urls = {
        'objects.json': 'http://campar.in.tum.de/public_datasets/3DSSG/3DSSG/objects.json',
        'classes.txt': 'http://campar.in.tum.de/public_datasets/3DSSG/3DSSG/classes.txt',
        'relationships.txt': 'http://campar.in.tum.de/public_datasets/3DSSG/3DSSG/relationships.txt'
    }
    
    for filename, url in urls.items():
        dest = os.path.join('data/external/3rscan', filename)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"{filename} already exists ({os.path.getsize(dest) / 1024:.1f} KB)")
            continue
        print(f"Downloading {filename} from {url}...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
                with open(dest, 'wb') as f:
                    f.write(data)
            print(f"Successfully downloaded {filename}: {len(data) / 1024:.1f} KB")
        except Exception as e:
            print(f"Failed to download {filename}: {e}")

if __name__ == '__main__':
    download()
