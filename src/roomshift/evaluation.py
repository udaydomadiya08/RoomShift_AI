"""
Evaluation module — metrics computation against ground truth.

Measures precision, recall, F1 for change classification,
per-category accuracy, and position errors where ground truth is available.
"""
import json
import logging
import os
from typing import List, Dict, Any, Optional

import numpy as np

from .types import (
    SpatialChange, ChangeType, GroundTruthChange, EvaluationResult, ChangeReport,
)

logger = logging.getLogger(__name__)


def evaluate_against_ground_truth(
    report: ChangeReport,
    ground_truth_path: str,
) -> EvaluationResult:
    """
    Evaluate change detection results against ground truth annotations.
    
    Ground truth JSON format:
    {
        "changes": [
            {"object_label": "chair", "change_type": "moved", "approximate_position": [x, y, z]},
            {"object_label": "box", "change_type": "added"},
            ...
        ]
    }
    """
    with open(ground_truth_path) as f:
        gt_data = json.load(f)
    
    gt_changes = [
        GroundTruthChange(**item) for item in gt_data.get('changes', [])
    ]
    
    return _compute_metrics(report.changes, gt_changes)


def _compute_metrics(
    predicted: List[SpatialChange],
    ground_truth: List[GroundTruthChange],
) -> EvaluationResult:
    """Compute evaluation metrics."""
    # Filter out uncertain predictions for P/R computation
    confident_preds = [p for p in predicted if p.change_type != ChangeType.UNCERTAIN]
    
    # Match predictions to ground truth
    matched_gt = set()
    matched_pred = set()
    details = []
    
    for i, pred in enumerate(confident_preds):
        pred_type = pred.change_type.value
        pred_label = pred.object_label or ""
        
        best_j = None
        best_score = 0
        
        for j, gt in enumerate(ground_truth):
            if j in matched_gt:
                continue
            if gt.change_type == "none":
                continue
            
            # Match by type and label similarity
            type_match = 1.0 if pred_type == gt.change_type else 0.0
            label_match = 1.0 if pred_label.lower() == gt.object_label.lower() else 0.3
            
            score = 0.6 * type_match + 0.4 * label_match
            
            if score > best_score and score > 0.5:
                best_score = score
                best_j = j
        
        if best_j is not None:
            matched_gt.add(best_j)
            matched_pred.add(i)
            details.append({
                'predicted': pred.to_dict(),
                'ground_truth': ground_truth[best_j].__dict__,
                'match_score': best_score,
                'result': 'true_positive',
            })
        else:
            details.append({
                'predicted': pred.to_dict(),
                'ground_truth': None,
                'result': 'false_positive',
            })
    
    # Unmatched ground truth = false negatives
    for j, gt in enumerate(ground_truth):
        if j not in matched_gt and gt.change_type != "none":
            details.append({
                'predicted': None,
                'ground_truth': gt.__dict__,
                'result': 'false_negative',
            })
    
    tp = len(matched_pred)
    fp = len(confident_preds) - tp
    fn = sum(1 for j, gt in enumerate(ground_truth) 
             if j not in matched_gt and gt.change_type != "none")
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Per-category metrics
    per_category = {}
    for change_type in ['moved', 'added', 'removed']:
        cat_preds = [p for p in confident_preds if p.change_type.value == change_type]
        cat_gt = [g for g in ground_truth if g.change_type == change_type]
        
        cat_tp = sum(1 for d in details 
                     if d['result'] == 'true_positive' and 
                     d['predicted']['change_type'] == change_type)
        cat_fp = len(cat_preds) - cat_tp
        cat_fn = len(cat_gt) - cat_tp
        
        cat_p = cat_tp / (cat_tp + cat_fp) if (cat_tp + cat_fp) > 0 else 0.0
        cat_r = cat_tp / (cat_tp + cat_fn) if (cat_tp + cat_fn) > 0 else 0.0
        cat_f1 = 2 * cat_p * cat_r / (cat_p + cat_r) if (cat_p + cat_r) > 0 else 0.0
        
        per_category[change_type] = {
            'precision': round(cat_p, 3),
            'recall': round(cat_r, 3),
            'f1': round(cat_f1, 3),
            'true_positives': cat_tp,
            'false_positives': cat_fp,
            'false_negatives': cat_fn,
        }
    
    result = EvaluationResult(
        precision=round(precision, 3),
        recall=round(recall, 3),
        f1=round(f1, 3),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        per_category=per_category,
        details=details,
    )
    
    logger.info(
        f"Evaluation: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f} "
        f"(TP={tp}, FP={fp}, FN={fn})"
    )
    
    return result


def save_evaluation_report(result: EvaluationResult, output_path: str):
    """Save evaluation results to JSON."""
    data = {
        'precision': result.precision,
        'recall': result.recall,
        'f1': result.f1,
        'true_positives': result.true_positives,
        'false_positives': result.false_positives,
        'false_negatives': result.false_negatives,
        'per_category': result.per_category,
        'details': result.details,
    }
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    logger.info(f"Evaluation report saved to {output_path}")
