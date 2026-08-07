#!/usr/bin/env python3
"""Extract 21-hand-landmark skeleton tensors from clean sample videos."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-airwriting")

import cv2
import mediapipe as mp
import numpy as np


TARGET_FRAMES = 128
LANDMARK_COUNT = 21
COORDS = 3
MAX_PROCESSED_FRAMES_PER_VIDEO = 48


def main() -> int:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[3]
    output_dir = script_path.parent / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = repo_root / "procedure1" / "artifacts" / "clean_manifest.csv"
    rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8", newline="")))

    sequences = []
    masks = []
    summary_rows = []
    model_path = repo_root / "public" / "models" / "hand_landmarker.task"
    timestamp_start_ms = 0
    with create_hand_landmarker(model_path) as hand_landmarker:
        for index, row in enumerate(rows, start=1):
            result = extract_video_skeleton(repo_root / row["video_path"], hand_landmarker, timestamp_start_ms)
            timestamp_start_ms = result["next_timestamp_ms"]
            sequences.append(result["sequence"])
            masks.append(result["mask"])
            summary_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "user_id": row["user_id"],
                    "label": row["label"],
                    "video_path": row["video_path"],
                    "frame_count": result["frame_count"],
                    "processed_frame_count": result["processed_frame_count"],
                    "valid_frame_count": result["valid_frame_count"],
                    "valid_frame_ratio": round_float(result["valid_frame_ratio"]),
                    **skeleton_summary_features(result["sequence"], result["mask"]),
                }
            )
            if index % 25 == 0:
                print(f"Processed {index}/{len(rows)} videos", flush=True)

    np.savez_compressed(
        output_dir / "skeleton_landmarks_128.npz",
        X=np.asarray(sequences, dtype=np.float32),
        masks=np.asarray(masks, dtype=np.float32),
        sample_ids=np.asarray([row["sample_id"] for row in rows]),
        labels=np.asarray([row["label"] for row in rows]),
        users=np.asarray([row["user_id"] for row in rows]),
    )
    write_csv(output_dir / "skeleton_features.csv", summary_rows)
    with (output_dir / "skeleton_feature_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "samples": len(rows),
                "tensor_shape": [len(rows), TARGET_FRAMES, LANDMARK_COUNT, COORDS],
                "max_processed_frames_per_video": MAX_PROCESSED_FRAMES_PER_VIDEO,
                "feature_rows": len(summary_rows),
                "mean_valid_frame_ratio": round_float(float(np.mean([row["valid_frame_ratio"] for row in summary_rows]))),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")

    print(f"Wrote skeleton feature artifacts to {output_dir}")
    return 0


def create_hand_landmarker(model_path: Path) -> Any:
    base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def extract_video_skeleton(video_path: Path, hand_landmarker: Any, timestamp_start_ms: int) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return empty_result(timestamp_start_ms)

    frame_index = 0
    processed_frame_count = 0
    declared_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target_indices = frame_indices_to_process(declared_frame_count)
    valid_indices = []
    valid_landmarks = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if target_indices and frame_index not in target_indices:
            frame_index += 1
            continue
        if not target_indices and processed_frame_count >= MAX_PROCESSED_FRAMES_PER_VIDEO:
            break
        processed_frame_count += 1
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        timestamp_ms = timestamp_start_ms + frame_index * 33
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        if result.hand_landmarks:
            landmarks = normalize_landmarks(result.hand_landmarks[0])
            valid_indices.append(frame_index)
            valid_landmarks.append(landmarks.reshape(-1))
        frame_index += 1
    cap.release()
    frame_count = declared_frame_count or frame_index

    if not valid_landmarks:
        return {
            "sequence": np.zeros((TARGET_FRAMES, LANDMARK_COUNT, COORDS), dtype=np.float32),
            "mask": np.zeros((TARGET_FRAMES,), dtype=np.float32),
            "frame_count": frame_count,
            "processed_frame_count": processed_frame_count,
            "valid_frame_count": 0,
            "valid_frame_ratio": 0.0,
            "next_timestamp_ms": timestamp_start_ms + max(1, frame_count + 1) * 33,
        }

    sequence = resample_landmarks(np.asarray(valid_indices, dtype=np.float64), np.asarray(valid_landmarks))
    mask = np.ones((TARGET_FRAMES,), dtype=np.float32)
    return {
        "sequence": sequence.reshape((TARGET_FRAMES, LANDMARK_COUNT, COORDS)).astype(np.float32),
        "mask": mask,
        "frame_count": frame_count,
        "processed_frame_count": processed_frame_count,
        "valid_frame_count": len(valid_landmarks),
        "valid_frame_ratio": len(valid_landmarks) / max(1, processed_frame_count),
        "next_timestamp_ms": timestamp_start_ms + max(1, frame_count + 1) * 33,
    }


def empty_result(timestamp_start_ms: int = 0) -> dict[str, Any]:
    return {
        "sequence": np.zeros((TARGET_FRAMES, LANDMARK_COUNT, COORDS), dtype=np.float32),
        "mask": np.zeros((TARGET_FRAMES,), dtype=np.float32),
        "frame_count": 0,
        "processed_frame_count": 0,
        "valid_frame_count": 0,
        "valid_frame_ratio": 0.0,
        "next_timestamp_ms": timestamp_start_ms + 33,
    }


def frame_indices_to_process(frame_count: int) -> set[int]:
    if frame_count <= 0:
        return set()
    count = min(MAX_PROCESSED_FRAMES_PER_VIDEO, frame_count)
    return {int(round(value)) for value in np.linspace(0, frame_count - 1, count)}


def normalize_landmarks(hand_landmarks: Any) -> np.ndarray:
    landmarks = hand_landmarks.landmark if hasattr(hand_landmarks, "landmark") else hand_landmarks
    coords = np.asarray([[landmark.x, landmark.y, landmark.z] for landmark in landmarks], dtype=np.float64)
    wrist = coords[0].copy()
    palm_size = max(
        float(np.linalg.norm(coords[5, :2] - wrist[:2])),
        float(np.linalg.norm(coords[9, :2] - wrist[:2])),
        float(np.linalg.norm(coords[17, :2] - wrist[:2])),
        1e-6,
    )
    return (coords - wrist) / palm_size


def resample_landmarks(valid_indices: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    if len(landmarks) == 1:
        return np.repeat(landmarks, TARGET_FRAMES, axis=0)

    source_t = (valid_indices - valid_indices.min()) / max(1e-9, valid_indices.max() - valid_indices.min())
    target_t = np.linspace(0.0, 1.0, TARGET_FRAMES)
    output = np.zeros((TARGET_FRAMES, landmarks.shape[1]), dtype=np.float64)
    for feature_index in range(landmarks.shape[1]):
        output[:, feature_index] = np.interp(target_t, source_t, landmarks[:, feature_index])
    return output


def skeleton_summary_features(sequence: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    if mask.sum() == 0:
        return {name: 0.0 for name in skeleton_feature_names()}

    flat = sequence.reshape((TARGET_FRAMES, -1))
    centroid = sequence.mean(axis=1)
    centroid_steps = np.linalg.norm(np.diff(centroid, axis=0), axis=1)
    index_tip = sequence[:, 8, :]
    index_steps = np.linalg.norm(np.diff(index_tip, axis=0), axis=1)
    values: dict[str, float] = {
        "skeleton_centroid_path_length": float(centroid_steps.sum()),
        "index_tip_path_length": float(index_steps.sum()),
        "index_tip_step_mean": float(index_steps.mean()) if len(index_steps) else 0.0,
        "index_tip_step_max": float(index_steps.max()) if len(index_steps) else 0.0,
        "hand_spread_mean": float(np.linalg.norm(sequence[:, 5, :2] - sequence[:, 17, :2], axis=1).mean()),
        "hand_spread_std": float(np.linalg.norm(sequence[:, 5, :2] - sequence[:, 17, :2], axis=1).std()),
    }
    means = flat.mean(axis=0)
    stds = flat.std(axis=0)
    mins = flat.min(axis=0)
    maxs = flat.max(axis=0)
    for index in range(flat.shape[1]):
        values[f"landmark_{index:02d}_mean"] = float(means[index])
        values[f"landmark_{index:02d}_std"] = float(stds[index])
        values[f"landmark_{index:02d}_min"] = float(mins[index])
        values[f"landmark_{index:02d}_max"] = float(maxs[index])
    return {key: round_float(value) for key, value in values.items()}


def skeleton_feature_names() -> list[str]:
    names = [
        "skeleton_centroid_path_length",
        "index_tip_path_length",
        "index_tip_step_mean",
        "index_tip_step_max",
        "hand_spread_mean",
        "hand_spread_std",
    ]
    for index in range(LANDMARK_COUNT * COORDS):
        names.extend(
            [
                f"landmark_{index:02d}_mean",
                f"landmark_{index:02d}_std",
                f"landmark_{index:02d}_min",
                f"landmark_{index:02d}_max",
            ]
        )
    return names


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def round_float(value: float, places: int = 6) -> float:
    return round(float(value), places)


if __name__ == "__main__":
    raise SystemExit(main())
