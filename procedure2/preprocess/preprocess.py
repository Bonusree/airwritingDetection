#!/usr/bin/env python3
"""Procedure 2 preprocessing pipeline for the Phase 1 Bangla vowel air-writing dataset.

This script implements the preprocessing stages described in
`Bangla_AirWriting_IEEE_Paper.pdf` ("Gesture-Triggered Segmentation and Causal
Stroke-Aware Trajectory Features for Real-Time Bangla Vowel Air-Writing
Recognition"), applied to the raw samples already committed under `output/`:

  * Section III-C / Reproducibility Statement: the causal 8-D trajectory
    feature encoding (delta_p, delta_q, sin/cos alpha, sin/cos beta,
    same_stroke, diff_stroke), reproduced here exactly as the capture client
    implements it (see "Formula notes" in the README for how this was
    reverse-engineered and validated against the stored dataset).
  * Section IV / VI: stroke-count over-segmentation caused by gesture-
    classification flicker, mitigated here with a stroke-debounce step
    (an ablation the paper specifies but leaves unimplemented).
  * Section III-D: arc-length resampling performed separately within each
    stroke to a fixed length T=256, with the 8-D features recomputed on the
    resampled sequence rather than interpolating the categorical stroke-flag
    channel directly.
  * Section V-A: leave-one-writer-out evaluation protocol, expressed here as
    fold assignments.
  * Section IV / Table II-III, Fig. 3-5: dataset characterization (recomputed
    on the current, larger snapshot of the dataset rather than the paper's
    152-sample snapshot).

The dataset scope follows the paper's Phase 1: only the four vowels
(অ, আ, ই, ঈ) are considered; any other label (e.g. "for_bonhi/ঔ") falls
outside Phase 1 and is not part of this preprocessing run.

Implementation uses only the Python standard library, with an optional
matplotlib dependency for the Fig. 3-5 style characterization plots (the
pipeline still runs to completion, skipping only the plots, if matplotlib is
unavailable).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PHASE1_VOWELS = ("অ", "আ", "ই", "ঈ")  # অ, আ, ই, ঈ
VIDEO_EXTENSIONS = (".webm", ".mp4")
MIN_VALID_POINTS = 10  # matches the capture client's own acceptance rule
TARGET_LENGTH = 256  # T, adopted in Section III-D from the median-260 length distribution
FEATURE_NAMES = (
    "delta_p",
    "delta_q",
    "sin_alpha",
    "cos_alpha",
    "sin_beta",
    "cos_beta",
    "same_stroke",
    "diff_stroke",
)
FEATURE_EPS = 1e-9  # d_i = ||(delta_p, delta_q)|| + eps, Eq. (1)
FIDELITY_TOLERANCE = 1e-4
# Okabe-Ito colorblind-safe categorical palette, fixed order (not cycled).
WRITER_COLORS = ("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
RAW_COLOR = "#D55E00"
DEBOUNCED_COLOR = "#0072B2"
HIST_COLOR = "#0072B2"


@dataclass
class Sample:
    json_path: Path
    path_user: str
    path_label: str
    sample_id: str = ""
    user_id: str = ""
    label: str = ""
    created_at: str = ""
    image_path: Path | None = None
    video_path: Path | None = None
    raw_points: list[tuple[float, float]] = field(default_factory=list)
    raw_strokes: list[int] = field(default_factory=list)
    stored_features: list[list[float]] = field(default_factory=list)
    dedup_points: list[tuple[float, float]] = field(default_factory=list)
    dedup_strokes: list[int] = field(default_factory=list)
    dropped_duplicates: int = 0
    debounced_points: list[tuple[float, float]] = field(default_factory=list)
    debounced_strokes: list[int] = field(default_factory=list)
    debounce_stats: dict[str, int] = field(default_factory=dict)
    cleaned_features: list[list[float]] = field(default_factory=list)
    resampled_points: list[tuple[float, float]] = field(default_factory=list)
    resampled_strokes: list[int] = field(default_factory=list)
    resampled_features: list[list[float]] = field(default_factory=list)
    fidelity: dict[str, float] = field(default_factory=dict)
    exclusions: list[tuple[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_excluded(self) -> bool:
        return bool(self.exclusions)


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    input_dir = (repo_root / args.input_dir).resolve()
    output_dir = (repo_root / args.output_dir).resolve()

    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    samples = discover_samples(input_dir)
    for sample in samples:
        load_and_validate(sample)

    mark_duplicate_sample_ids(samples)

    clean_samples = [s for s in samples if not s.is_excluded]
    for sample in clean_samples:
        process_clean_sample(sample, debounce_k=args.debounce_k, target_length=args.target_length)

    lowo_rows = build_lowo_folds(clean_samples)

    write_manifest(repo_root, output_dir, samples)
    write_exclusions(repo_root, output_dir, samples)
    write_fidelity_report(output_dir, samples)
    write_stroke_debounce_report(output_dir, clean_samples, args.debounce_k)
    write_cleaned_jsonl(output_dir, clean_samples, args.target_length)
    write_lowo_folds(output_dir, lowo_rows)
    counts = write_dataset_characterization(output_dir, clean_samples, args.debounce_k, args.target_length)
    write_run_summary(output_dir, samples, clean_samples, args, counts)

    if not args.no_plots:
        make_figures(output_dir, clean_samples, counts)

    print_summary(samples, clean_samples, counts)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="Repository root. Defaults to current directory.")
    parser.add_argument("--input-dir", default="output", help="Raw dataset directory relative to repo root.")
    parser.add_argument(
        "--output-dir",
        default="procedure2/preprocess/artifacts",
        help="Artifact directory relative to repo root.",
    )
    parser.add_argument(
        "--debounce-k",
        type=int,
        default=3,
        help=(
            "Minimum point count for a stroke to survive debounce; shorter "
            "strokes are merged into a neighboring stroke (see README for why "
            "point count, not frame gap, is used as the debounce signal)."
        ),
    )
    parser.add_argument(
        "--target-length",
        type=int,
        default=TARGET_LENGTH,
        help="Resampled sequence length T (Section III-D).",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip matplotlib figure generation.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Discovery and validation
# ---------------------------------------------------------------------------


def discover_samples(input_dir: Path) -> list[Sample]:
    samples: list[Sample] = []
    for json_path in sorted(input_dir.glob("*/*/*.json")):
        path_label = normalize_text(json_path.parent.name)
        path_user = normalize_text(json_path.parent.parent.name)
        if path_label not in PHASE1_VOWELS:
            continue  # out of Phase 1 scope (e.g. for_bonhi/ঔ)
        samples.append(Sample(json_path=json_path, path_user=path_user, path_label=path_label))
    return samples


def load_and_validate(sample: Sample) -> None:
    try:
        with sample.json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:  # noqa: BLE001 - report the exact parse failure
        sample.sample_id = sample.json_path.stem
        sample.user_id = sample.path_user
        sample.label = sample.path_label
        sample.exclusions.append(("corrupt_json", str(exc)))
        return

    if not isinstance(data, dict):
        sample.sample_id = sample.json_path.stem
        sample.user_id = sample.path_user
        sample.label = sample.path_label
        sample.exclusions.append(("corrupt_json", "top-level JSON is not an object"))
        return

    sample.sample_id = normalize_text(data.get("sample_id") or sample.json_path.stem)
    sample.user_id = normalize_text(data.get("username") or sample.path_user)
    sample.label = normalize_text(data.get("label") or sample.path_label)
    sample.created_at = normalize_text(data.get("created_at") or "")
    sample.image_path = find_companion_file(sample, ".png")
    sample.video_path = find_companion_file(sample, VIDEO_EXTENSIONS)

    required_fields = ("sample_id", "username", "label", "frame_size", "trajectory")
    missing = [name for name in required_fields if name not in data or data.get(name) in (None, "")]
    if missing:
        sample.exclusions.append(("missing_required_fields", ",".join(missing)))

    if sample.user_id != sample.path_user:
        sample.warnings.append(f"path_user_mismatch:{sample.path_user}")
    if sample.label != sample.path_label:
        sample.exclusions.append(("path_label_mismatch", f"path={sample.path_label};json={sample.label}"))
    if sample.label not in PHASE1_VOWELS:
        sample.exclusions.append(("out_of_phase1_scope", f"label={sample.label}"))

    if not valid_frame_size(data.get("frame_size")):
        sample.exclusions.append(("invalid_frame_size", repr(data.get("frame_size"))))
    if sample.image_path is None:
        sample.warnings.append("missing_png")
    if sample.video_path is None:
        sample.warnings.append("missing_video")

    validate_trajectory(sample, data.get("trajectory"))


def validate_trajectory(sample: Sample, trajectory: Any) -> None:
    if not isinstance(trajectory, dict):
        sample.exclusions.append(("invalid_trajectory", "trajectory is missing or not an object"))
        return

    points_raw = trajectory.get("points_normalized")
    strokes_raw = trajectory.get("stroke_ids")
    features_raw = trajectory.get("features_8d")
    num_points_raw = trajectory.get("num_points")

    if not isinstance(points_raw, list) or not isinstance(strokes_raw, list) or not isinstance(features_raw, list):
        sample.exclusions.append(("invalid_trajectory", "points_normalized/stroke_ids/features_8d not all lists"))
        return
    if isinstance(num_points_raw, int) and num_points_raw != len(points_raw):
        sample.exclusions.append(("num_points_mismatch", f"declared={num_points_raw};actual={len(points_raw)}"))
    if len(strokes_raw) != len(points_raw) or len(features_raw) != len(points_raw):
        sample.exclusions.append(
            ("length_mismatch", f"points={len(points_raw)};strokes={len(strokes_raw)};features={len(features_raw)}")
        )
        return

    points: list[tuple[float, float]] = []
    strokes: list[int] = []
    features: list[list[float]] = []
    dropped_invalid = 0
    for point, stroke, feat in zip(points_raw, strokes_raw, features_raw, strict=True):
        parsed_point = parse_point(point)
        stroke_id = parse_stroke_id(stroke)
        feat_row = parse_feature_row(feat)
        if parsed_point is None or stroke_id is None or feat_row is None:
            dropped_invalid += 1
            continue
        x, y = parsed_point
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            dropped_invalid += 1
            continue
        points.append((x, y))
        strokes.append(stroke_id)
        features.append(feat_row)

    if dropped_invalid:
        sample.warnings.append(f"dropped_invalid_points:{dropped_invalid}")

    sample.raw_points = points
    sample.raw_strokes = strokes
    sample.stored_features = features

    if len(points) < MIN_VALID_POINTS:
        sample.exclusions.append(("fewer_than_min_valid_points", f"valid_points={len(points)}"))


def normalize_text(value: Any) -> str:
    return unicodedata.normalize("NFC", str(value or "").strip())


def parse_point(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    return (x, y) if math.isfinite(x) and math.isfinite(y) else None


def parse_stroke_id(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_feature_row(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 8:
        return None
    row = []
    for item in value:
        try:
            number = float(item)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        row.append(number)
    return row


def valid_frame_size(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 2:
        return False
    try:
        width, height = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return False
    return math.isfinite(width) and math.isfinite(height) and width > 0 and height > 0


def find_companion_file(sample: Sample, extensions: str | tuple[str, ...]) -> Path | None:
    if isinstance(extensions, str):
        extensions = (extensions,)
    directory = sample.json_path.parent
    candidates = [sample.json_path.with_suffix(ext) for ext in extensions]
    for ext in extensions:
        candidates.append(directory / f"{sample.sample_id}{ext}")
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    for ext in extensions:
        matches = sorted(directory.glob(f"*{sample.sample_id}*{ext}"))
        if matches:
            return matches[0]
    return None


def mark_duplicate_sample_ids(samples: list[Sample]) -> None:
    seen: dict[str, Sample] = {}
    for sample in sorted(samples, key=lambda item: item.json_path.as_posix()):
        if not sample.sample_id:
            continue
        if sample.sample_id in seen:
            sample.exclusions.append(("duplicate_sample_id", f"first_json={seen[sample.sample_id].json_path.as_posix()}"))
            continue
        seen[sample.sample_id] = sample


# ---------------------------------------------------------------------------
# Causal 8-D feature encoding (Eqs. 1-3), exactly as the capture client
# computes it -- see README "Formula notes" for how the two behaviors below
# (no reset at stroke boundaries; defaults only at the trajectory's global
# first point) were reverse-engineered and validated against the dataset.
# ---------------------------------------------------------------------------


def causal_features(points: list[tuple[float, float]], strokes: list[int]) -> list[list[float]]:
    n = len(points)
    features: list[list[float]] = []
    sin_prev, cos_prev = 0.0, 1.0
    for i in range(n):
        if i == 0:
            dx, dy = 0.0, 0.0
            sin_a, cos_a = 0.0, 1.0
        else:
            dx = points[i][0] - points[i - 1][0]
            dy = points[i][1] - points[i - 1][1]
            distance = math.hypot(dx, dy) + FEATURE_EPS
            sin_a = dy / distance
            cos_a = dx / distance
        if i < 2:
            sin_b, cos_b = 0.0, 1.0
        else:
            sin_b = sin_a * cos_prev - cos_a * sin_prev
            cos_b = cos_a * cos_prev + sin_a * sin_prev
        same_stroke = 1.0 if i < n - 1 and strokes[i] == strokes[i + 1] else 0.0
        diff_stroke = 1.0 - same_stroke
        features.append(
            [
                round6(dx),
                round6(dy),
                round6(sin_a),
                round6(cos_a),
                round6(sin_b),
                round6(cos_b),
                same_stroke,
                diff_stroke,
            ]
        )
        sin_prev, cos_prev = sin_a, cos_a
    return features


def round6(value: float) -> float:
    return round(float(value), 6)


def euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ---------------------------------------------------------------------------
# Per-sample processing: dedup -> stroke debounce -> feature recompute -> resample
# ---------------------------------------------------------------------------


def process_clean_sample(sample: Sample, *, debounce_k: int, target_length: int) -> None:
    compute_fidelity(sample)

    dedup_points, dedup_strokes, dropped = drop_exact_duplicates(sample.raw_points, sample.raw_strokes)
    sample.dedup_points = dedup_points
    sample.dedup_strokes = dedup_strokes
    sample.dropped_duplicates = dropped

    debounced_points, debounced_strokes, stats = debounce_strokes(dedup_points, dedup_strokes, debounce_k)
    sample.debounced_points = debounced_points
    sample.debounced_strokes = debounced_strokes
    sample.debounce_stats = stats

    sample.cleaned_features = causal_features(debounced_points, debounced_strokes)

    resampled_points, resampled_strokes = resample_by_stroke(debounced_points, debounced_strokes, target_length)
    sample.resampled_points = resampled_points
    sample.resampled_strokes = resampled_strokes
    sample.resampled_features = causal_features(resampled_points, resampled_strokes)


def compute_fidelity(sample: Sample) -> None:
    """Compare our recomputed causal features against the capture client's
    stored features_8d on the untouched raw points, as a validation that this
    script's formula matches Eqs. (1)-(3) as actually implemented at
    collection time (see README for the ~a-few-percent expected divergence
    on near-zero-displacement frames, caused by 6-decimal rounding of
    points_normalized erasing the sub-pixel motion the stored features were
    originally computed from)."""
    recomputed = causal_features(sample.raw_points, sample.raw_strokes)
    max_diff = 0.0
    mismatches = 0
    total = 0
    for computed_row, stored_row in zip(recomputed, sample.stored_features, strict=True):
        row_diff = max(abs(round6(c) - round6(s)) for c, s in zip(computed_row, stored_row))
        max_diff = max(max_diff, row_diff)
        total += 1
        if row_diff > FIDELITY_TOLERANCE:
            mismatches += 1
    sample.fidelity = {
        "points_checked": total,
        "mismatches": mismatches,
        "mismatch_rate": round(mismatches / total, 6) if total else 0.0,
        "max_abs_diff": round6(max_diff),
    }


def drop_exact_duplicates(
    points: list[tuple[float, float]], strokes: list[int]
) -> tuple[list[tuple[float, float]], list[int], int]:
    if not points:
        return [], [], 0
    kept_points = [points[0]]
    kept_strokes = [strokes[0]]
    dropped = 0
    for point, stroke in zip(points[1:], strokes[1:]):
        if stroke == kept_strokes[-1] and point == kept_points[-1]:
            dropped += 1
            continue
        kept_points.append(point)
        kept_strokes.append(stroke)
    return kept_points, kept_strokes, dropped


def debounce_strokes(
    points: list[tuple[float, float]], strokes: list[int], k: int
) -> tuple[list[tuple[float, float]], list[int], dict[str, int]]:
    """Merge strokes shorter than k points into a neighboring stroke.

    The paper (Section V-B, VI) proposes merging strokes "separated by fewer
    than k frames" to fix gesture-classification flicker. The stored
    trajectory only records points captured during Draw frames, so the
    duration of a pen-up gap between two strokes is not recoverable from this
    data; a short stroke's own point count is used as the closest available
    proxy for a flicker artifact instead (see README).
    """
    if not points:
        return [], [], {"raw_strokes": 0, "debounced_strokes": 0, "merges": 0, "k": k}

    runs: list[list[int]] = []  # [start_index, end_index] inclusive, per contiguous stroke run
    start = 0
    for i in range(1, len(strokes) + 1):
        if i == len(strokes) or strokes[i] != strokes[i - 1]:
            runs.append([start, i - 1])
            start = i
    raw_stroke_count = len(runs)

    merges = 0
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for idx, run in enumerate(runs):
            length = run[1] - run[0] + 1
            if length >= k:
                continue
            if idx == 0:
                runs[idx + 1][0] = run[0]
            else:
                runs[idx - 1][1] = run[1]
            del runs[idx]
            merges += 1
            changed = True
            break

    new_strokes = [0] * len(strokes)
    for new_id, (run_start, run_end) in enumerate(runs, start=1):
        for i in range(run_start, run_end + 1):
            new_strokes[i] = new_id

    return points, new_strokes, {
        "raw_strokes": raw_stroke_count,
        "debounced_strokes": len(runs),
        "merges": merges,
        "k": k,
    }


def resample_by_stroke(
    points: list[tuple[float, float]], strokes: list[int], target_length: int
) -> tuple[list[tuple[float, float]], list[int]]:
    """Arc-length interpolation performed separately within each stroke
    (Section III-D), allocating the target length across strokes in
    proportion to arc length (falling back to point count for
    near-zero-length strokes so single-point flicker survivors still get a
    representative point)."""
    if not points:
        return [], []
    chunks = split_into_runs(points, strokes)
    if len(chunks) >= target_length:
        # More strokes than the budget allows: take one representative point per
        # stroke, in order, for the first target_length strokes.
        chosen = chunks[:target_length]
        return (
            [chunk_points[0] for chunk_points, _ in chosen],
            [stroke_id for _, stroke_id in chosen],
        )

    weights = [max(arc_length(chunk_points), float(len(chunk_points))) for chunk_points, _ in chunks]
    allocations = largest_remainder_allocation(weights, target_length, minimum=1)

    resampled_points: list[tuple[float, float]] = []
    resampled_strokes: list[int] = []
    for (chunk_points, stroke_id), allocation in zip(chunks, allocations, strict=True):
        interpolated = interpolate_chunk(chunk_points, allocation)
        resampled_points.extend(interpolated)
        resampled_strokes.extend([stroke_id] * len(interpolated))

    if len(resampled_points) > target_length:
        resampled_points = resampled_points[:target_length]
        resampled_strokes = resampled_strokes[:target_length]
    while len(resampled_points) < target_length:
        resampled_points.append(resampled_points[-1])
        resampled_strokes.append(resampled_strokes[-1])
    return resampled_points, resampled_strokes


def split_into_runs(
    points: list[tuple[float, float]], strokes: list[int]
) -> list[tuple[list[tuple[float, float]], int]]:
    chunks: list[tuple[list[tuple[float, float]], int]] = []
    current_points: list[tuple[float, float]] = []
    current_stroke: int | None = None
    for point, stroke in zip(points, strokes, strict=True):
        if current_stroke is None or stroke == current_stroke:
            current_points.append(point)
            current_stroke = stroke
            continue
        chunks.append((current_points, current_stroke))
        current_points = [point]
        current_stroke = stroke
    if current_points and current_stroke is not None:
        chunks.append((current_points, current_stroke))
    return chunks


def arc_length(points: list[tuple[float, float]]) -> float:
    return sum(euclidean(points[i], points[i - 1]) for i in range(1, len(points)))


def largest_remainder_allocation(weights: list[float], total: int, *, minimum: int) -> list[int]:
    n = len(weights)
    if n * minimum > total:
        # Degenerate case (more chunks than budget after minimums): even split.
        base = total // n
        allocation = [base] * n
        for i in range(total - base * n):
            allocation[i] += 1
        return allocation

    weight_sum = sum(weights) or float(n)
    raw = [total * (w / weight_sum) for w in weights]
    allocation = [max(minimum, int(math.floor(value))) for value in raw]

    while sum(allocation) > total:
        surplus_candidates = [i for i in range(n) if allocation[i] > minimum]
        if not surplus_candidates:
            break
        chosen = max(surplus_candidates, key=lambda i: allocation[i] - raw[i])
        allocation[chosen] -= 1

    while sum(allocation) < total:
        chosen = max(range(n), key=lambda i: raw[i] - allocation[i])
        allocation[chosen] += 1

    return allocation


def interpolate_chunk(points: list[tuple[float, float]], count: int) -> list[tuple[float, float]]:
    if count <= 0:
        return []
    if len(points) == 1 or count == 1:
        return [points[0]] * count if len(points) == 1 else [points[0]]

    cumulative = [0.0]
    for i in range(1, len(points)):
        cumulative.append(cumulative[-1] + euclidean(points[i], points[i - 1]))
    total_length = cumulative[-1]
    if total_length <= 1e-12:
        return [points[0]] * count

    targets = [total_length * i / (count - 1) for i in range(count)]
    result: list[tuple[float, float]] = []
    segment = 1
    for target in targets:
        while segment < len(cumulative) - 1 and cumulative[segment] < target:
            segment += 1
        prev_d, next_d = cumulative[segment - 1], cumulative[segment]
        ratio = 0.0 if next_d == prev_d else (target - prev_d) / (next_d - prev_d)
        p0, p1 = points[segment - 1], points[segment]
        result.append((p0[0] + (p1[0] - p0[0]) * ratio, p0[1] + (p1[1] - p0[1]) * ratio))
    return result


# ---------------------------------------------------------------------------
# Leave-one-writer-out folds (Section V-A)
# ---------------------------------------------------------------------------


def build_lowo_folds(clean_samples: list[Sample]) -> list[dict[str, Any]]:
    writers = sorted({s.user_id for s in clean_samples})
    rows: list[dict[str, Any]] = []
    for fold_id, test_writer in enumerate(writers):
        for sample in clean_samples:
            rows.append(
                {
                    "fold_id": fold_id,
                    "test_writer": test_writer,
                    "sample_id": sample.sample_id,
                    "user_id": sample.user_id,
                    "label": sample.label,
                    "role": "test" if sample.user_id == test_writer else "train",
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def relpath(path: Path | None, repo_root: Path) -> str:
    if path is None:
        return ""
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_manifest(repo_root: Path, output_dir: Path, samples: list[Sample]) -> None:
    rows = [
        {
            "sample_id": s.sample_id,
            "user_id": s.user_id,
            "label": s.label,
            "accepted": not s.is_excluded,
            "json_path": relpath(s.json_path, repo_root),
            "image_path": relpath(s.image_path, repo_root),
            "video_path": relpath(s.video_path, repo_root),
            "created_at": s.created_at,
            "raw_points": len(s.raw_points),
        }
        for s in samples
    ]
    write_csv(
        output_dir / "manifest.csv",
        rows,
        ("sample_id", "user_id", "label", "accepted", "json_path", "image_path", "video_path", "created_at", "raw_points"),
    )


def write_exclusions(repo_root: Path, output_dir: Path, samples: list[Sample]) -> None:
    exclusion_rows = []
    warning_rows = []
    for s in samples:
        for reason, details in s.exclusions:
            exclusion_rows.append(
                {
                    "sample_id": s.sample_id,
                    "user_id": s.user_id,
                    "label": s.label,
                    "json_path": relpath(s.json_path, repo_root),
                    "reason": reason,
                    "details": details,
                }
            )
        for warning in s.warnings:
            warning_rows.append(
                {
                    "sample_id": s.sample_id,
                    "user_id": s.user_id,
                    "label": s.label,
                    "json_path": relpath(s.json_path, repo_root),
                    "warning": warning,
                }
            )
    write_csv(
        output_dir / "excluded_samples.csv",
        exclusion_rows,
        ("sample_id", "user_id", "label", "json_path", "reason", "details"),
    )
    write_csv(
        output_dir / "sample_warnings.csv",
        warning_rows,
        ("sample_id", "user_id", "label", "json_path", "warning"),
    )


def write_fidelity_report(output_dir: Path, samples: list[Sample]) -> None:
    rows = []
    for s in samples:
        if not s.fidelity:
            continue
        rows.append(
            {
                "sample_id": s.sample_id,
                "user_id": s.user_id,
                "label": s.label,
                "points_checked": s.fidelity["points_checked"],
                "mismatches_above_1e-4": s.fidelity["mismatches"],
                "mismatch_rate": s.fidelity["mismatch_rate"],
                "max_abs_diff": s.fidelity["max_abs_diff"],
            }
        )
    write_csv(
        output_dir / "fidelity_report.csv",
        rows,
        ("sample_id", "user_id", "label", "points_checked", "mismatches_above_1e-4", "mismatch_rate", "max_abs_diff"),
    )


def write_stroke_debounce_report(output_dir: Path, clean_samples: list[Sample], k: int) -> None:
    rows = []
    for s in clean_samples:
        stats = s.debounce_stats
        rows.append(
            {
                "sample_id": s.sample_id,
                "user_id": s.user_id,
                "label": s.label,
                "raw_strokes": stats.get("raw_strokes", 0),
                "debounced_strokes": stats.get("debounced_strokes", 0),
                "merges": stats.get("merges", 0),
                "k": k,
                "duplicate_points_dropped": s.dropped_duplicates,
            }
        )
    write_csv(
        output_dir / "stroke_debounce_report.csv",
        rows,
        ("sample_id", "user_id", "label", "raw_strokes", "debounced_strokes", "merges", "k", "duplicate_points_dropped"),
    )


def write_cleaned_jsonl(output_dir: Path, clean_samples: list[Sample], target_length: int) -> None:
    with (output_dir / "cleaned_trajectories.jsonl").open("w", encoding="utf-8") as cleaned_handle, (
        output_dir / f"resampled_{target_length}.jsonl"
    ).open("w", encoding="utf-8") as resampled_handle:
        for s in clean_samples:
            cleaned_row = {
                "sample_id": s.sample_id,
                "user_id": s.user_id,
                "label": s.label,
                "num_points": len(s.debounced_points),
                "num_strokes": s.debounce_stats.get("debounced_strokes", 0),
                "points": [[round6(x), round6(y)] for x, y in s.debounced_points],
                "stroke_ids": s.debounced_strokes,
                "features_8d": s.cleaned_features,
                "feature_names": list(FEATURE_NAMES),
            }
            cleaned_handle.write(json.dumps(cleaned_row, ensure_ascii=False) + "\n")

            resampled_row = {
                "sample_id": s.sample_id,
                "user_id": s.user_id,
                "label": s.label,
                "num_points": len(s.resampled_points),
                "points": [[round6(x), round6(y)] for x, y in s.resampled_points],
                "stroke_ids": s.resampled_strokes,
                "features_8d": s.resampled_features,
                "feature_names": list(FEATURE_NAMES),
            }
            resampled_handle.write(json.dumps(resampled_row, ensure_ascii=False) + "\n")


def write_lowo_folds(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(
        output_dir / "lowo_folds.csv",
        rows,
        ("fold_id", "test_writer", "sample_id", "user_id", "label", "role"),
    )


def write_dataset_characterization(
    output_dir: Path, clean_samples: list[Sample], debounce_k: int, target_length: int
) -> dict[str, Any]:
    writers = sorted({s.user_id for s in clean_samples})
    labels = list(PHASE1_VOWELS)

    matrix: dict[tuple[str, str], int] = Counter((s.user_id, s.label) for s in clean_samples)
    long_rows = [
        {"user_id": writer, "label": label, "count": matrix.get((writer, label), 0)}
        for writer in writers
        for label in labels
    ]
    write_csv(output_dir / "per_writer_per_vowel_counts.csv", long_rows, ("user_id", "label", "count"))

    with (output_dir / "per_writer_per_vowel_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["writer", *labels])
        for w in writers:
            writer.writerow([w, *(matrix.get((w, label), 0) for label in labels)])

    samples_per_writer = {w: sum(1 for s in clean_samples if s.user_id == w) for w in writers}
    points_per_sample = [len(s.debounced_points) for s in clean_samples]
    raw_strokes_per_sample = [s.debounce_stats.get("raw_strokes", 0) for s in clean_samples]
    debounced_strokes_per_sample = [s.debounce_stats.get("debounced_strokes", 0) for s in clean_samples]

    counts = {
        "writers": writers,
        "vowel_classes": labels,
        "total_accepted_samples": len(clean_samples),
        "samples_per_writer": samples_per_writer,
        "points_per_sample": distribution_stats(points_per_sample),
        "strokes_per_sample_raw": distribution_stats(raw_strokes_per_sample),
        "strokes_per_sample_debounced": distribution_stats(debounced_strokes_per_sample),
        "debounce_k": debounce_k,
        "target_length": target_length,
        "raw_strokes_list": raw_strokes_per_sample,
        "debounced_strokes_list": debounced_strokes_per_sample,
        "points_per_sample_list": points_per_sample,
    }

    with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {k: v for k, v in counts.items() if not k.endswith("_list")},
            handle,
            ensure_ascii=False,
            indent=2,
        )

    return counts


def distribution_stats(values: list[int]) -> dict[str, float]:
    if not values:
        return {"min": 0, "max": 0, "median": 0, "mode": 0}
    counter = Counter(values)
    mode_value = counter.most_common(1)[0][0]
    return {
        "min": min(values),
        "max": max(values),
        "median": statistics.median(values),
        "mode": mode_value,
    }


def write_run_summary(
    output_dir: Path,
    samples: list[Sample],
    clean_samples: list[Sample],
    args: argparse.Namespace,
    counts: dict[str, Any],
) -> None:
    exclusion_reasons = Counter(reason for s in samples for reason, _ in s.exclusions)
    summary = {
        "phase1_vowels": list(PHASE1_VOWELS),
        "samples_discovered": len(samples),
        "samples_accepted": len(clean_samples),
        "samples_excluded": len(samples) - len(clean_samples),
        "exclusion_reasons": dict(exclusion_reasons),
        "debounce_k": args.debounce_k,
        "target_length": args.target_length,
        "writers": counts["writers"],
        "total_accepted_samples": counts["total_accepted_samples"],
        "samples_per_writer": counts["samples_per_writer"],
        "points_per_sample": counts["points_per_sample"],
        "strokes_per_sample_raw": counts["strokes_per_sample_raw"],
        "strokes_per_sample_debounced": counts["strokes_per_sample_debounced"],
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Figures (Fig. 3-5 style dataset characterization plots)
# ---------------------------------------------------------------------------


def make_figures(output_dir: Path, clean_samples: list[Sample], counts: dict[str, Any]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    figures_dir = output_dir / "figures"
    plot_samples_per_writer_per_vowel(plt, figures_dir, clean_samples)
    plot_length_distribution(plt, figures_dir, counts["points_per_sample_list"])
    plot_stroke_distribution(plt, figures_dir, counts["raw_strokes_list"], counts["debounced_strokes_list"])


def bengali_font(plt: Any) -> Any | None:
    """Return a matplotlib FontProperties for a Bengali-capable font if one
    is installed, so vowel labels in figures render as glyphs instead of
    tofu boxes under matplotlib's default DejaVu Sans."""
    try:
        import matplotlib.font_manager as fm
    except ImportError:
        return None
    for candidate in ("Noto Sans Bengali", "Noto Serif Bengali", "Kalpurush", "Nirmala UI"):
        try:
            path = fm.findfont(candidate, fallback_to_default=False)
        except Exception:
            continue
        return fm.FontProperties(fname=path)
    return None


def plot_samples_per_writer_per_vowel(plt: Any, figures_dir: Path, clean_samples: list[Sample]) -> None:
    writers = sorted({s.user_id for s in clean_samples})
    labels = list(PHASE1_VOWELS)
    matrix = Counter((s.user_id, s.label) for s in clean_samples)
    font = bengali_font(plt)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(labels))
    bar_width = 0.8 / max(len(writers), 1)
    for i, writer in enumerate(writers):
        values = [matrix.get((writer, label), 0) for label in labels]
        offsets = [xi + i * bar_width - 0.4 + bar_width / 2 for xi in x]
        ax.bar(offsets, values, width=bar_width, label=writer, color=WRITER_COLORS[i % len(WRITER_COLORS)])
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=16, fontproperties=font)
    ax.set_ylabel("Samples")
    ax.set_title("Samples collected per writer per vowel (Phase 1)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncols=len(writers))
    fig.tight_layout()
    fig.savefig(figures_dir / "samples_per_writer_per_vowel.png", dpi=150)
    plt.close(fig)


def plot_length_distribution(plt: Any, figures_dir: Path, points_per_sample: list[int]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(points_per_sample, bins=30, color=HIST_COLOR, edgecolor="white")
    ax.set_xlabel("Points per sample (after debounce)")
    ax.set_ylabel("Count")
    ax.set_title(f"Trajectory length distribution (n={len(points_per_sample)})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures_dir / "trajectory_length_distribution.png", dpi=150)
    plt.close(fig)


def plot_stroke_distribution(plt: Any, figures_dir: Path, raw_strokes: list[int], debounced_strokes: list[int]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    max_val = max(raw_strokes + debounced_strokes, default=1)
    bins = list(range(1, max_val + 2))
    ax.hist(raw_strokes, bins=bins, alpha=0.7, label="Raw (gesture-triggered)", color=RAW_COLOR)
    ax.hist(debounced_strokes, bins=bins, alpha=0.7, label="Debounced", color=DEBOUNCED_COLOR)
    ax.set_xlabel("Detected strokes per sample")
    ax.set_ylabel("Count")
    ax.set_title(f"Stroke-count distribution before/after debounce (n={len(raw_strokes)})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir / "stroke_count_distribution.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------


def print_summary(samples: list[Sample], clean_samples: list[Sample], counts: dict[str, Any]) -> None:
    print(f"Phase 1 samples discovered (labels {PHASE1_VOWELS}): {len(samples)}")
    print(f"Accepted after validation: {len(clean_samples)}")
    print(f"Excluded: {len(samples) - len(clean_samples)}")
    print(f"Writers: {counts['writers']}")
    print(f"Samples per writer: {counts['samples_per_writer']}")
    print(f"Points per sample: {counts['points_per_sample']}")
    print(f"Strokes per sample (raw): {counts['strokes_per_sample_raw']}")
    print(f"Strokes per sample (debounced, k={counts['debounce_k']}): {counts['strokes_per_sample_debounced']}")


if __name__ == "__main__":
    raise SystemExit(main())
