#!/usr/bin/env python3
"""Record skeleton-stream readiness for the current local environment."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def main() -> int:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[3]
    output_dir = script_path.parent / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = repo_root / "procedure1" / "artifacts" / "clean_manifest.csv"

    video_rows = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            video_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "user_id": row["user_id"],
                    "label": row["label"],
                    "video_path": row["video_path"],
                    "video_exists": str((repo_root / row["video_path"]).exists()),
                }
            )

    mediapipe_available = importlib.util.find_spec("mediapipe") is not None
    status = "ready" if mediapipe_available else "blocked_missing_mediapipe"
    for row in video_rows:
        row["skeleton_status"] = status

    with (output_dir / "skeleton_extraction_status.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("sample_id", "user_id", "label", "video_path", "video_exists", "skeleton_status"),
        )
        writer.writeheader()
        writer.writerows(video_rows)

    with (output_dir / "skeleton_readiness.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "mediapipe_available": mediapipe_available,
                "clean_videos": len(video_rows),
                "status": status,
                "note": "Run extract_skeleton_features.py to materialize 21-hand-landmark tensors from video."
                if mediapipe_available
                else "Install mediapipe to materialize 21-hand-landmark skeleton tensors from video.",
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")

    print(f"Wrote skeleton readiness artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
