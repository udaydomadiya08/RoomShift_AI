"""
Check for accessible real scan data files.
"""
import urllib.request
import json
import os

def check_sources():
    # Check if there are public releases on WaldJohannaU/3RScan
    api_url = "https://api.github.com/repos/WaldJohannaU/3RScan/releases"
    try:
        req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            releases = json.loads(resp.read().decode())
            print(f"Releases on WaldJohannaU/3RScan: {len(releases)}")
            for r in releases:
                print("Release:", r.get('tag_name'), "Assets:", [a['name'] for a in r.get('assets', [])])
    except Exception as e:
        print("Failed to check releases:", e)

if __name__ == '__main__':
    check_sources()
