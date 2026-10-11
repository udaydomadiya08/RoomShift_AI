"""
Trace exact matching and change detection for Scene 095821f7 -> 2e369567.
"""
from roomshift.adapters.rscan_adapter import RScanDataset
from roomshift.object_tracking import match_objects_across_recordings
from roomshift.change_detection import detect_changes
from roomshift.config import ChangeDetectionConfig
import numpy as np

dataset = RScanDataset(data_dir='data/external/3rscan')
pair = dataset.get_scene_pair_by_id('095821f7-e2c2-2de1-9568-b9ce59920e29', '2e369567-e133-204c-909a-c5da44bb58df')

recon_before, recon_after, alignment, tracked_before, tracked_after = dataset.to_roomshift_reconstructions(pair)

print("Tracked before count:", len(tracked_before))
print("Tracked after count:", len(tracked_after))

# Check ground truth
print("\nGround truth changes in 3RScan:")
for c in pair.changes:
    print(f"  GT ID {c.instance_id}: {c.label} -> {c.change_type.value} (disp={c.displacement:.3f}m)")

# Run matching
matches = match_objects_across_recordings(
    tracked_before, tracked_after,
    alignment.transform,
    position_threshold=2.5,
)

print(f"\nMatched count: {len(matches)}")
matched_before_ids = set()
matched_after_ids = set()
for b, a, s in matches:
    matched_before_ids.add(b.track_id)
    matched_after_ids.add(a.track_id)
    R = alignment.transform[:3, :3]
    t = alignment.transform[:3, 3]
    pos_after_aligned = R @ a.mean_position_3d + t
    disp = np.linalg.norm(b.mean_position_3d - pos_after_aligned)
    print(f"  Match {b.track_id} ({b.label}) <-> {a.track_id} ({a.label}): score={s:.2f}, displacement={disp:.3f}m")

unmatched_b = [b for b in tracked_before if b.track_id not in matched_before_ids]
unmatched_a = [a for a in tracked_after if a.track_id not in matched_after_ids]

print("\nUnmatched before:")
for b in unmatched_b:
    print(f"  {b.track_id} ({b.label}) pos={b.mean_position_3d}")

print("\nUnmatched after:")
for a in unmatched_a:
    print(f"  {a.track_id} ({a.label}) pos={a.mean_position_3d}")
