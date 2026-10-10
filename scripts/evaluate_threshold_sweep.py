"""
Sweep match position thresholds on 3RScan validation scenes.
"""
import logging
from roomshift.adapters.evaluate_3rscan import evaluate_3rscan_validation

logging.basicConfig(level=logging.WARNING)

def sweep():
    print(f"{'Threshold':>10} | {'Precision':>10} | {'Recall':>10} | {'F1':>10} | {'Moved TP':>10} | {'Moved FN':>10} | {'Disp MAE':>10}")
    print("-" * 80)
    for thresh in [0.3, 0.6, 1.0, 1.5, 2.5]:
        res = evaluate_3rscan_validation(max_scenes=15, position_threshold=thresh)
        moved = res['per_class']['moved']
        mae = res['displacement_mae_meters']
        print(f"{thresh:>10.2f} | {res['precision']:>10.3f} | {res['recall']:>10.3f} | {res['f1_score']:>10.3f} | {moved['tp']:>10d} | {moved['fn']:>10d} | {mae:>9.3f}m")

if __name__ == '__main__':
    sweep()
