"""
Evaluation runner for RoomShift AI on the real 3RScan dataset.

Evaluates the spatial-change detection engine against the official 3RScan
ground-truth annotations on the validation split.

Calculates:
- Precision, Recall, F1 score (overall and per-class: moved, removed, added)
- False Positives and False Negatives breakdown
- Displacement error for moved objects
- Execution performance statistics
"""
import os
import sys
import json
import logging
import argparse
from typing import List, Dict, Any

import numpy as np

from .rscan_adapter import RScanDataset, RScanScenePair, RScanObjectChange
from ..change_detection import detect_changes
from ..object_tracking import match_objects_across_recordings
from ..config import ChangeDetectionConfig
from ..types import ChangeType, SpatialChange, EvaluationResult

logger = logging.getLogger(__name__)


def evaluate_3rscan_validation(
    data_dir: str = "data/external/3rscan",
    max_scenes: int = 20,
    position_threshold: float = 0.25,
    match_distance_threshold: float = 2.5,
) -> Dict[str, Any]:
    """
    Run evaluation across 3RScan validation scenes.
    """
    dataset = RScanDataset(data_dir=data_dir)
    if not dataset.is_loaded:
        raise RuntimeError(f"Could not load 3RScan dataset from {data_dir}")

    val_pairs = dataset.get_scene_pairs(split="validation")
    logger.info(f"Loaded {len(val_pairs)} 3RScan validation scene pairs")

    config = ChangeDetectionConfig(
        position_threshold=position_threshold,
        match_distance_threshold=match_distance_threshold,
        geometric_comparison=False,  # evaluate object-level semantic changes
        min_confidence=0.2,
    )

    results_per_pair = []
    total_tp = 0
    total_fp = 0
    total_fn = 0

    per_class_stats = {
        "moved": {"tp": 0, "fp": 0, "fn": 0},
        "removed": {"tp": 0, "fp": 0, "fn": 0},
        "added": {"tp": 0, "fp": 0, "fn": 0},
    }

    displacement_errors = []

    evaluated_count = 0

    for pair in val_pairs:
        # Only evaluate pairs with documented changes
        if not pair.changes:
            continue

        if evaluated_count >= max_scenes:
            break
        evaluated_count += 1

        # Convert pair to RoomShift AI representations
        recon_before, recon_after, alignment, tracked_before, tracked_after = (
            dataset.to_roomshift_reconstructions(pair)
        )

        # Match objects across recordings using RoomShift AI's matching logic
        matches = match_objects_across_recordings(
            tracked_before,
            tracked_after,
            alignment.transform,
            position_threshold=config.match_distance_threshold,
        )

        # Run RoomShift AI's spatial change detection
        predicted_changes = detect_changes(
            recon_before,
            recon_after,
            tracked_before,
            tracked_after,
            matches,
            alignment,
            config,
        )

        # Ground truth changes for this pair
        gt_changes = pair.changes

        # Evaluate predictions against ground truth
        matched_gt = set()
        pair_tp = 0
        pair_fp = 0

        for pred in predicted_changes:
            if pred.change_type == ChangeType.UNCERTAIN:
                continue

            match_found = False
            for i, gt in enumerate(gt_changes):
                if i in matched_gt:
                    continue

                if pred.object_label == gt.label and pred.change_type == gt.change_type:
                    matched_gt.add(i)
                    match_found = True
                    pair_tp += 1

                    cat = pred.change_type.value
                    if cat in per_class_stats:
                        per_class_stats[cat]["tp"] += 1

                    # Displacement error for moved objects
                    if pred.change_type == ChangeType.MOVED and pred.displacement is not None:
                        err = abs(pred.displacement - gt.displacement)
                        displacement_errors.append(err)

                    break

            if not match_found:
                pair_fp += 1
                cat = pred.change_type.value
                if cat in per_class_stats:
                    per_class_stats[cat]["fp"] += 1

        pair_fn = len(gt_changes) - len(matched_gt)
        for i, gt in enumerate(gt_changes):
            if i not in matched_gt:
                cat = gt.change_type.value
                if cat in per_class_stats:
                    per_class_stats[cat]["fn"] += 1

        total_tp += pair_tp
        total_fp += pair_fp
        total_fn += pair_fn

        results_per_pair.append({
            "scene_id": pair.scene_id,
            "ref_id": pair.reference_id,
            "rescan_id": pair.rescan_id,
            "gt_count": len(gt_changes),
            "pred_count": len([p for p in predicted_changes if p.change_type != ChangeType.UNCERTAIN]),
            "tp": pair_tp,
            "fp": pair_fp,
            "fn": pair_fn,
        })

    # Overall metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Per-class metrics
    class_metrics = {}
    for cat, counts in per_class_stats.items():
        c_tp = counts["tp"]
        c_fp = counts["fp"]
        c_fn = counts["fn"]
        c_prec = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
        c_rec = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
        c_f1 = 2 * c_prec * c_rec / (c_prec + c_rec) if (c_prec + c_rec) > 0 else 0.0
        class_metrics[cat] = {
            "precision": round(c_prec, 4),
            "recall": round(c_rec, 4),
            "f1": round(c_f1, 4),
            "tp": c_tp,
            "fp": c_fp,
            "fn": c_fn,
        }

    disp_mae = float(np.mean(displacement_errors)) if displacement_errors else 0.0
    disp_median = float(np.median(displacement_errors)) if displacement_errors else 0.0

    summary = {
        "dataset": "3RScan",
        "split": "validation",
        "scenes_evaluated": evaluated_count,
        "total_true_positives": total_tp,
        "total_false_positives": total_fp,
        "total_false_negatives": total_fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "displacement_mae_meters": round(disp_mae, 4),
        "displacement_median_error_meters": round(disp_median, 4),
        "per_class": class_metrics,
        "scene_results": results_per_pair,
    }

    return summary


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Evaluate RoomShift AI on 3RScan")
    parser.add_argument("--data-dir", default="data/external/3rscan", help="Path to 3RScan data")
    parser.add_argument("--max-scenes", type=int, default=25, help="Number of validation scenes to evaluate")
    parser.add_argument("--output", default="outputs/3rscan_evaluation.json", help="Output path for results JSON")
    args = parser.parse_args()

    results = evaluate_3rscan_validation(
        data_dir=args.data_dir,
        max_scenes=args.max_scenes,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("ROOMSHIFT AI — 3RSCAN VALIDATION EVALUATION RESULTS")
    print("=" * 60)
    print(f"Dataset:              3RScan (TUM CAMPAR, ICCV 2019)")
    print(f"Split:                Validation")
    print(f"Scenes Evaluated:     {results['scenes_evaluated']}")
    print(f"Precision:            {results['precision']:.3f}")
    print(f"Recall:               {results['recall']:.3f}")
    print(f"F1 Score:             {results['f1_score']:.3f}")
    print(f"TP / FP / FN:         {results['total_true_positives']} / {results['total_false_positives']} / {results['total_false_negatives']}")
    if results['displacement_mae_meters'] > 0:
        print(f"Displacement MAE:     {results['displacement_mae_meters']:.3f} m")
    print("\nPer-Class Breakdown:")
    for cat, m in results['per_class'].items():
        print(f"  {cat.upper():10s} - P: {m['precision']:.3f}, R: {m['recall']:.3f}, F1: {m['f1']:.3f} (TP={m['tp']}, FP={m['fp']}, FN={m['fn']})")
    print("=" * 60)
    print(f"Report saved to: {os.path.abspath(args.output)}\n")


if __name__ == "__main__":
    main()
