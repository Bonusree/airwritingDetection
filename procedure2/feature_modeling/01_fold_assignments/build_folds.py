#!/usr/bin/env python3
"""Expand procedure2's leave-one-writer-out folds into nested outer/inner splits.

`procedure2/preprocess/preprocess.py` (Section V-A) already fixes the outer
leave-one-writer-out test assignment in `lowo_folds.csv`: one fold per writer,
that writer's samples held out as test, everyone else's samples marked train.

The baselines in this folder need an inner validation split too, for model
selection that never touches the outer test writer. This script adds that
inner layer with the same grouped, count-balanced assignment procedure1 used
(`assign_balanced_user_groups` in `procedure1/preprocess_clean_augment.py`),
so `common/ml_utils.run_nested_cv` works unmodified against
`sample_splits.csv` here exactly as it does for procedure1.

Outputs (this folder's `artifacts/`):
  * `nested_fold_assignments.csv`: outer_fold_id, inner_fold_id, user_id, role,
    outer_split_policy, inner_split_policy.
  * `sample_splits.csv`: outer_fold_id, inner_fold_id, sample_id, user_id,
    label, role, json_path, image_path, video_path.
  * `fold_summary.json`: counts and policy notes.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def main() -> int:
    script_path = Path(__file__).resolve()
    preprocess_dir = script_path.parents[2] / "preprocess" / "artifacts"
    output_dir = script_path.parent / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = load_csv(preprocess_dir / "manifest.csv")
    accepted_rows = [row for row in manifest_rows if row["accepted"] == "True"]
    lowo_rows = load_csv(preprocess_dir / "lowo_folds.csv")

    users = sorted({row["user_id"] for row in accepted_rows})
    sample_counts_by_user: dict[str, int] = {}
    for row in accepted_rows:
        sample_counts_by_user[row["user_id"]] = sample_counts_by_user.get(row["user_id"], 0) + 1

    outer_fold_ids = sorted({int(row["fold_id"]) for row in lowo_rows})
    test_writer_by_fold = {
        int(row["fold_id"]): row["test_writer"] for row in lowo_rows
    }

    outer_split_policy = "leave_one_writer_out" if len(users) >= 5 else f"grouped_{len(users)}_fold"
    nested_rows: list[dict[str, Any]] = []
    inner_fold_counts: dict[int, int] = {}

    for outer_fold_id in outer_fold_ids:
        test_writer = test_writer_by_fold[outer_fold_id]
        development_users = [user for user in users if user != test_writer]
        inner_fold_count = min(3, len(development_users))
        inner_fold_counts[outer_fold_id] = inner_fold_count
        inner_split_policy = f"grouped_{inner_fold_count}_fold_inner_validation"
        validation_groups = assign_balanced_user_groups(development_users, sample_counts_by_user, inner_fold_count)

        for inner_fold_id, validation_users in enumerate(validation_groups):
            validation_set = set(validation_users)
            for user in users:
                if user == test_writer:
                    role = "test"
                elif user in validation_set:
                    role = "val"
                else:
                    role = "train"
                nested_rows.append(
                    {
                        "outer_fold_id": outer_fold_id,
                        "inner_fold_id": inner_fold_id,
                        "user_id": user,
                        "role": role,
                        "outer_split_policy": outer_split_policy,
                        "inner_split_policy": inner_split_policy,
                    }
                )

    write_csv(output_dir / "nested_fold_assignments.csv", nested_rows)

    role_by_key = {
        (row["outer_fold_id"], row["inner_fold_id"], row["user_id"]): row["role"] for row in nested_rows
    }
    split_keys = sorted({(row["outer_fold_id"], row["inner_fold_id"]) for row in nested_rows})
    sample_split_rows: list[dict[str, Any]] = []
    for outer_fold_id, inner_fold_id in split_keys:
        for row in accepted_rows:
            sample_split_rows.append(
                {
                    "outer_fold_id": outer_fold_id,
                    "inner_fold_id": inner_fold_id,
                    "sample_id": row["sample_id"],
                    "user_id": row["user_id"],
                    "label": row["label"],
                    "role": role_by_key[(outer_fold_id, inner_fold_id, row["user_id"])],
                    "json_path": row["json_path"],
                    "image_path": row["image_path"],
                    "video_path": row["video_path"],
                }
            )
    write_csv(output_dir / "sample_splits.csv", sample_split_rows)

    write_json(
        output_dir / "fold_summary.json",
        {
            "writers": users,
            "accepted_samples": len(accepted_rows),
            "outer_split_policy": outer_split_policy,
            "outer_folds": len(outer_fold_ids),
            "inner_fold_counts_by_outer": {str(k): v for k, v in inner_fold_counts.items()},
            "sample_split_rows": len(sample_split_rows),
        },
    )

    print(f"Wrote fold assignment artifacts to {output_dir}")
    print(f"Writers: {users}")
    print(f"Outer folds: {len(outer_fold_ids)} ({outer_split_policy})")
    print(f"Sample split rows: {len(sample_split_rows)}")
    return 0


def assign_balanced_user_groups(
    users: list[str],
    sample_counts_by_user: dict[str, int],
    group_count: int,
) -> list[list[str]]:
    if group_count <= 0:
        return []
    groups: list[list[str]] = [[] for _ in range(group_count)]
    group_sizes = [0 for _ in range(group_count)]
    for user in sorted(users, key=lambda item: (-sample_counts_by_user.get(item, 0), item)):
        group_index = min(range(group_count), key=lambda index: (group_sizes[index], index))
        groups[group_index].append(user)
        group_sizes[group_index] += sample_counts_by_user.get(user, 0)
    return [sorted(group) for group in groups]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
