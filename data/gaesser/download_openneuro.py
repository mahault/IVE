"""
Download behavioral/phenotypic data from OpenNeuro dataset ds001439.

Gaesser, B., Hirschfeld-Kroen, J., Wasserman, E. A., Horn, M., & Young, L.
(2019). A role for the medial temporal lobe subsystem in guiding prosociality:
the effect of episodic processes on willingness to help others. Social
Cognitive and Affective Neuroscience, 14(4), 397-410.

Data sources:
  - OpenNeuro (fMRI + BIDS): https://openneuro.org/datasets/ds001439
  - OSF (behavioral):        https://osf.io/9k4n7/
  - GitHub mirror:           https://github.com/OpenNeuroDatasets/ds001439

This script downloads:
  1. dataset_description.json  (from OpenNeuro GitHub mirror)
  2. participants.tsv          (from OpenNeuro GitHub mirror)
  3. All *_events.tsv files    (from OpenNeuro GitHub mirror)
  4. Behavioral data files     (from OSF project 9k4n7)

It does NOT download the large NIfTI fMRI files. To get the full
fMRI dataset (~several GB), run:
    aws s3 sync --no-sign-request s3://openneuro.org/ds001439 ./ds001439_full/
  or:
    datalad install https://github.com/OpenNeuroDatasets/ds001439.git
"""

import json
import os
import ssl
import urllib.request
import urllib.error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ID = "ds001439"

# OpenNeuro datasets are mirrored on GitHub via git-annex.
# Small text files (TSV, JSON) are stored directly; large files are annexed.
GITHUB_RAW = (
    f"https://raw.githubusercontent.com/OpenNeuroDatasets/{DATASET_ID}/main"
)

# Fallback: try 'master' branch if 'main' doesn't exist
GITHUB_RAW_MASTER = (
    f"https://raw.githubusercontent.com/OpenNeuroDatasets/{DATASET_ID}/master"
)

# GitHub API to list repository tree
GITHUB_API_TREE = (
    f"https://api.github.com/repos/OpenNeuroDatasets/{DATASET_ID}"
    "/git/trees/HEAD?recursive=1"
)

# OSF API for behavioral data
OSF_API_URL = "https://api.osf.io/v2/nodes/9k4n7/files/osfstorage/"

# Number of subjects from the paper (18 completed fMRI Experiment 1)
EXPECTED_SUBJECTS = 18


def _build_opener():
    """Build a URL opener that handles SSL and sets a user-agent."""
    ctx = ssl.create_default_context()
    handler = urllib.request.HTTPSHandler(context=ctx)
    opener = urllib.request.build_opener(handler)
    opener.addheaders = [("User-Agent", "ive-download/1.0")]
    return opener


OPENER = _build_opener()


def fetch_url(url, as_json=False, timeout=30):
    """Fetch a URL and return bytes or parsed JSON."""
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "ive-download/1.0")
    if as_json:
        req.add_header("Accept", "application/json")
    try:
        with OPENER.open(req, timeout=timeout) as resp:
            raw = resp.read()
            if as_json:
                return json.loads(raw.decode("utf-8"))
            return raw
    except urllib.error.HTTPError as exc:
        print(f"  HTTP {exc.code} for {url}")
        return None
    except Exception as exc:
        print(f"  Error fetching {url}: {exc}")
        return None


def save_bytes(data, rel_path):
    """Save raw bytes to a file under BASE_DIR, creating dirs as needed."""
    dest = os.path.join(BASE_DIR, rel_path)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        f.write(data)
    print(f"  Saved {dest} ({len(data)} bytes)")
    return dest


# ── Step 1: Get the full file tree from the GitHub API ──────────────────────

def get_github_tree():
    """Return a list of file paths in the OpenNeuro GitHub mirror."""
    print("Fetching repository file tree from GitHub API ...")
    data = fetch_url(GITHUB_API_TREE, as_json=True)
    if data is None:
        # Try alternative tree endpoint
        alt_url = (
            f"https://api.github.com/repos/OpenNeuroDatasets/{DATASET_ID}"
            "/git/trees/master?recursive=1"
        )
        data = fetch_url(alt_url, as_json=True)

    if data is None:
        print("  Could not fetch tree from GitHub API.")
        return []

    tree = data.get("tree", [])
    paths = [
        item["path"]
        for item in tree
        if item.get("type") == "blob"
    ]
    print(f"  Found {len(paths)} files in repository.")

    # Report total dataset info
    total_size = sum(item.get("size", 0) for item in tree if item.get("type") == "blob")
    nifti_size = sum(
        item.get("size", 0)
        for item in tree
        if item.get("type") == "blob" and item["path"].endswith((".nii", ".nii.gz"))
    )
    print(f"  Total tracked size (text refs): ~{total_size / 1024:.0f} KB")
    print(f"  Note: NIfTI files are git-annex pointers, real size is much larger.")

    return paths


def select_files_to_download(all_paths):
    """Select only the behavioral/metadata files we want."""
    wanted = []
    for p in all_paths:
        basename = os.path.basename(p)
        # Root-level metadata
        if p in (
            "dataset_description.json",
            "participants.tsv",
            "participants.json",
            "task-prosocial_bold.json",
            "README",
            "CHANGES",
        ):
            wanted.append(p)
        # Any events.tsv file (trial-level behavioral data)
        elif basename.endswith("_events.tsv"):
            wanted.append(p)
        # Any events JSON sidecar
        elif basename.endswith("_events.json"):
            wanted.append(p)
        # Task JSON sidecars at any level
        elif "task-" in basename and basename.endswith(".json"):
            wanted.append(p)
    return wanted


def download_from_github(file_paths):
    """Download selected files from the GitHub raw content endpoint."""
    print(f"\nDownloading {len(file_paths)} files from OpenNeuro GitHub mirror ...")
    downloaded = []
    for rel_path in file_paths:
        url = f"{GITHUB_RAW}/{rel_path}"
        data = fetch_url(url)
        if data is None:
            # Try master branch
            url = f"{GITHUB_RAW_MASTER}/{rel_path}"
            data = fetch_url(url)
        if data is not None:
            # Check if this is a git-annex pointer instead of real content
            text = data.decode("utf-8", errors="replace")
            if text.startswith("/annex/") or "git-annex" in text[:200]:
                print(f"  SKIP {rel_path}: git-annex pointer (large file)")
                continue
            dest = save_bytes(data, os.path.join("openneuro", rel_path))
            downloaded.append(dest)
        else:
            print(f"  FAILED {rel_path}")
    return downloaded


# ── Step 2: Download behavioral data from OSF ──────────────────────────────

def list_osf_files(url):
    """Recursively list all files from an OSF storage endpoint."""
    data = fetch_url(url, as_json=True)
    if data is None:
        return []

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
                print(f"  [folder] {name}")
                files.extend(list_osf_files(folder_url))
        elif kind == "file":
            download = links.get("download", "")
            size = attrs.get("size", 0)
            files.append({"name": name, "download": download, "size": size})
            print(f"  [file] {name}  ({size} bytes)")

    # Pagination
    next_url = data.get("links", {}).get("next")
    if next_url:
        files.extend(list_osf_files(next_url))

    return files


def download_osf_files():
    """Download all behavioral data files from the OSF project."""
    print("\nListing files on OSF project 9k4n7 (Gaesser behavioral data) ...")
    files = list_osf_files(OSF_API_URL)
    print(f"  Found {len(files)} file(s) on OSF.\n")

    osf_dir = os.path.join(BASE_DIR, "osf_behavioral")
    os.makedirs(osf_dir, exist_ok=True)

    downloaded = []
    for f in files:
        name = f["name"]
        url = f["download"]
        if not url:
            print(f"  SKIP {name}: no download URL")
            continue
        dest = os.path.join(osf_dir, name)
        print(f"  Downloading {name} ({f['size']} bytes) ...")
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "ive-download/1.0")
            with OPENER.open(req, timeout=60) as resp:
                data = resp.read()
            with open(dest, "wb") as fh:
                fh.write(data)
            actual = os.path.getsize(dest)
            print(f"  Saved {dest} ({actual} bytes)")
            downloaded.append(dest)
        except Exception as exc:
            print(f"  FAILED {name}: {exc}")

    return downloaded


# ── Step 3: Fallback - hardcoded file list if API is unavailable ────────────

# Based on BIDS convention for 18 subjects, 8 functional runs + 2 localizer
# runs per subject, task name "prosocial" (inferred from paper).
FALLBACK_ROOT_FILES = [
    "dataset_description.json",
    "participants.tsv",
    "participants.json",
    "README",
    "CHANGES",
]


def download_fallback():
    """Try to download key root files directly by guessing standard paths."""
    print("\nFallback: downloading root-level files by name ...")
    downloaded = []
    for fname in FALLBACK_ROOT_FILES:
        for base in [GITHUB_RAW, GITHUB_RAW_MASTER]:
            url = f"{base}/{fname}"
            data = fetch_url(url)
            if data is not None:
                text = data.decode("utf-8", errors="replace")
                if text.startswith("/annex/") or "git-annex" in text[:200]:
                    print(f"  SKIP {fname}: git-annex pointer")
                    break
                dest = save_bytes(data, os.path.join("openneuro", fname))
                downloaded.append(dest)
                break
        else:
            print(f"  Could not download {fname}")
    return downloaded


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Gaesser et al. (2019) - Prosocial willingness-to-help fMRI data")
    print(f"OpenNeuro dataset: {DATASET_ID}")
    print(f"Destination: {BASE_DIR}")
    print("=" * 70)

    os.makedirs(BASE_DIR, exist_ok=True)

    # Part A: OpenNeuro / GitHub files
    all_paths = get_github_tree()

    if all_paths:
        wanted = select_files_to_download(all_paths)
        print(f"\nSelected {len(wanted)} behavioral/metadata files to download:")
        for p in wanted:
            print(f"  {p}")
        openneuro_files = download_from_github(wanted)

        # Report on full dataset
        nifti_paths = [p for p in all_paths if p.endswith((".nii", ".nii.gz"))]
        print(f"\n--- Full dataset summary ---")
        print(f"  Total files in dataset: {len(all_paths)}")
        print(f"  NIfTI files (fMRI):     {len(nifti_paths)}")
        print(f"  Subjects (from paper):  {EXPECTED_SUBJECTS}")
        print(f"  Note: NIfTI sizes are git-annex pointers on GitHub.")
        print(f"  To get full fMRI data, use:")
        print(f"    aws s3 sync --no-sign-request s3://openneuro.org/{DATASET_ID} ./{DATASET_ID}_full/")
        print(f"    or: datalad install https://github.com/OpenNeuroDatasets/{DATASET_ID}.git")
    else:
        print("\nGitHub tree unavailable; trying fallback downloads ...")
        openneuro_files = download_fallback()

    # Part B: OSF behavioral data
    osf_files = download_osf_files()

    # Summary
    all_downloaded = openneuro_files + osf_files
    print("\n" + "=" * 70)
    print(f"Download complete. {len(all_downloaded)} files saved.")
    print("=" * 70)
    print(f"\nFiles in {BASE_DIR}:")
    for root, dirs, files in os.walk(BASE_DIR):
        for fn in sorted(files):
            if fn.endswith(".py") or fn.startswith("."):
                continue
            fp = os.path.join(root, fn)
            rel = os.path.relpath(fp, BASE_DIR)
            sz = os.path.getsize(fp)
            print(f"  {rel}  ({sz:,} bytes)")


if __name__ == "__main__":
    main()
