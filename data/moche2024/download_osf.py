"""
Download data files from OSF project ukqs8 (Moche et al. 2024).

Victim identifiability, number of victims, and unit asking
in charitable giving. 5 studies, N=7,996.
"""
import urllib.request
import json
import os
import sys
import ssl

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "download_log.txt")
API_URL = "https://api.osf.io/v2/nodes/ukqs8/files/osfstorage/"

# Create an SSL context that doesn't verify (for corporate/proxy environments)
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


def log(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")


def fetch_json(url):
    """Fetch JSON from a URL."""
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
        return json.loads(resp.read().decode())


def list_files(url, depth=0):
    """Recursively list all files from an OSF storage endpoint."""
    prefix = "  " * depth
    data = fetch_json(url)

    # Save raw API response for debugging
    if depth == 0:
        with open(os.path.join(BASE_DIR, "api_response.json"), "w") as f:
            json.dump(data, f, indent=2)

    files = []
    for item in data.get("data", []):
        attrs = item.get("attributes", {})
        links = item.get("links", {})
        name = attrs.get("name", "unknown")
        kind = attrs.get("kind", "")
        if kind == "folder":
            folder_url = links.get("related", {})
            if isinstance(folder_url, dict):
                folder_url = folder_url.get("href", "")
            if folder_url:
                log(f"{prefix}[folder] {name}")
                files.extend(list_files(folder_url, depth + 1))
        elif kind == "file":
            download = links.get("download", "")
            size = attrs.get("size", 0)
            files.append({"name": name, "download": download, "size": size})
            log(f"{prefix}[file] {name}  ({size} bytes)")

    # Handle pagination
    next_url = data.get("links", {}).get("next")
    if next_url:
        files.extend(list_files(next_url, depth))

    return files


def download_file(file_info, dest_dir):
    """Download a single file."""
    name = file_info["name"]
    url = file_info["download"]
    dest = os.path.join(dest_dir, name)
    if not url:
        log(f"  SKIP {name}: no download URL")
        return False
    log(f"  Downloading {name} ...")
    try:
        urllib.request.urlretrieve(url, dest)
        actual = os.path.getsize(dest)
        log(f"  OK: {name} ({actual} bytes)")
        return True
    except Exception as e:
        log(f"  ERROR downloading {name}: {e}")
        return False


def main():
    # Clear log
    with open(LOG_FILE, "w") as f:
        f.write("")

    log("=" * 60)
    log("OSF Download: Moche et al. (2024) - Project ukqs8")
    log("=" * 60)
    log(f"\nListing files from {API_URL} ...")

    try:
        files = list_files(API_URL)
    except Exception as e:
        log(f"\nERROR listing files: {e}")
        log("Trying alternate approach with children nodes...")
        try:
            children_data = fetch_json(
                "https://api.osf.io/v2/nodes/ukqs8/children/"
            )
            log(f"Found {len(children_data.get('data', []))} child nodes")
            for child in children_data.get("data", []):
                cid = child.get("id", "")
                ctitle = child.get("attributes", {}).get("title", "")
                log(f"  Child: {ctitle} (id={cid})")
                child_files_url = (
                    f"https://api.osf.io/v2/nodes/{cid}/files/osfstorage/"
                )
                try:
                    child_files = list_files(child_files_url, depth=1)
                    files.extend(child_files)
                except Exception as e2:
                    log(f"    ERROR: {e2}")
        except Exception as e2:
            log(f"ERROR with children: {e2}")
            files = []

    log(f"\nFound {len(files)} file(s) total.\n")

    if not files:
        log("No files found.")
        return

    ok = 0
    for f in files:
        if download_file(f, dest_dir=BASE_DIR):
            ok += 1

    log(f"\nDownloaded {ok}/{len(files)} files.")
    log(f"\nFiles in {BASE_DIR}:")
    for fn in sorted(os.listdir(BASE_DIR)):
        fp = os.path.join(BASE_DIR, fn)
        if os.path.isfile(fp) and fn not in ("download_osf.py", "download_log.txt", "api_response.json"):
            log(f"  {fn}  ({os.path.getsize(fp)} bytes)")


if __name__ == "__main__":
    main()
