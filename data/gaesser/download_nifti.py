"""
Download NIfTI fMRI files from OpenNeuro dataset ds001439.

Gaesser et al. (2019) - Prosocial willingness-to-help fMRI data.

This downloads the full fMRI dataset (~several GB) via AWS S3.
Requires: AWS CLI installed (pip install awscli).

Usage:
    python download_nifti.py                        # Download all subjects
    python download_nifti.py --subjects sub-04 sub-05  # Specific subjects
    python download_nifti.py --task ieh             # Only ieh task (not tom)
    python download_nifti.py --dryrun               # Show what would be downloaded
"""

import argparse
import os
import subprocess
import sys

DATASET_ID = "ds001439"
S3_BUCKET = f"s3://openneuro.org/{DATASET_ID}"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DEST = os.path.join(BASE_DIR, "openneuro_nifti")

# All subjects from participants.tsv
ALL_SUBJECTS = [
    "sub-04", "sub-05", "sub-06", "sub-07", "sub-08",
    "sub-11", "sub-12", "sub-13", "sub-14", "sub-16",
    "sub-17", "sub-18", "sub-19", "sub-20", "sub-21",
    "sub-22", "sub-24", "sub-25",
]


def download_via_s3(dest_dir, subjects=None, tasks=None, dryrun=False):
    """Download NIfTI files via AWS S3 (no credentials needed).

    Args:
        dest_dir: Destination directory.
        subjects: List of subject IDs. None = all.
        tasks: List of task names ('ieh', 'tom'). None = both.
        dryrun: If True, only show what would be downloaded.
    """
    os.makedirs(dest_dir, exist_ok=True)
    subjects = subjects or ALL_SUBJECTS
    tasks = tasks or ["ieh", "tom"]

    # Build include patterns for aws s3 sync
    include_patterns = []
    for sub in subjects:
        for task in tasks:
            include_patterns.append(f"--include={sub}/func/{sub}_task-{task}_*")

    # Also include root-level metadata
    include_patterns.extend([
        "--include=dataset_description.json",
        "--include=participants.tsv",
        "--include=participants.json",
        "--include=README",
        "--include=CHANGES",
    ])

    cmd = [
        "aws", "s3", "sync",
        "--no-sign-request",
        "--exclude=*",  # exclude everything first
    ] + include_patterns + [
        S3_BUCKET,
        dest_dir,
    ]

    if dryrun:
        cmd.insert(3, "--dryrun")

    print(f"Downloading from {S3_BUCKET}")
    print(f"Destination: {dest_dir}")
    print(f"Subjects: {', '.join(subjects)}")
    print(f"Tasks: {', '.join(tasks)}")
    if dryrun:
        print("(DRY RUN - no files will be downloaded)")
    print()
    print(f"Command: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(cmd, check=True)
        print(f"\nDownload complete (exit code {result.returncode}).")
    except FileNotFoundError:
        print("ERROR: AWS CLI not found. Install with: pip install awscli")
        print("Alternative: use datalad to download the dataset:")
        print(f"  datalad install https://github.com/OpenNeuroDatasets/{DATASET_ID}.git")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Download failed (exit code {e.returncode}).")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download Gaesser et al. (2019) NIfTI fMRI files from OpenNeuro."
    )
    parser.add_argument(
        "--dest", default=DEFAULT_DEST,
        help=f"Destination directory (default: {DEFAULT_DEST})"
    )
    parser.add_argument(
        "--subjects", nargs="+", default=None,
        help="Subject IDs to download (e.g., sub-04 sub-05). Default: all 18."
    )
    parser.add_argument(
        "--task", nargs="+", default=None, dest="tasks",
        help="Task names to download (ieh, tom). Default: both."
    )
    parser.add_argument(
        "--dryrun", action="store_true",
        help="Show what would be downloaded without downloading."
    )
    args = parser.parse_args()
    download_via_s3(args.dest, args.subjects, args.tasks, args.dryrun)


if __name__ == "__main__":
    main()
