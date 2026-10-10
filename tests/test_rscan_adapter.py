"""
Test suite for the 3RScan Dataset Adapter and Real-Data Evaluation.
"""
import os
import sys
import numpy as np
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from roomshift.adapters.rscan_adapter import RScanDataset, RScanScenePair, RScanObjectChange
from roomshift.adapters.evaluate_3rscan import evaluate_3rscan_validation
from roomshift.types import ChangeType, GroundTruthChange, SceneReconstruction


class TestRScanAdapter:
    @pytest.fixture
    def dataset(self):
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'external', '3rscan')
        if not os.path.exists(os.path.join(data_dir, "3RScan.json")):
            pytest.skip("3RScan.json not available for test")
        return RScanDataset(data_dir=data_dir)

    def test_load_dataset(self, dataset):
        assert dataset.is_loaded
        assert len(dataset.scenes_data) == 478
        assert len(dataset.objects_data) > 0

    def test_list_scenes_by_split(self, dataset):
        val_scenes = dataset.list_scenes(split="validation")
        assert len(val_scenes) == 47
        train_scenes = dataset.list_scenes(split="train")
        assert len(train_scenes) == 385
        test_scenes = dataset.list_scenes(split="test")
        assert len(test_scenes) == 46

    def test_extract_scene_pairs(self, dataset):
        val_pairs = dataset.get_scene_pairs(split="validation")
        assert len(val_pairs) > 50
        first = val_pairs[0]
        assert isinstance(first, RScanScenePair)
        assert first.global_transform.shape == (4, 4)
        assert first.split == "validation"

    def test_column_major_transform_and_displacement(self, dataset):
        """Verify column-major 4x4 matrix translation matches metric displacement."""
        val_pairs = dataset.get_scene_pairs(split="validation")
        pairs_with_moves = [p for p in val_pairs if any(c.change_type == ChangeType.MOVED for c in p.changes)]
        assert len(pairs_with_moves) > 0

        first_pair = pairs_with_moves[0]
        moved_objs = [c for c in first_pair.changes if c.change_type == ChangeType.MOVED]
        assert len(moved_objs) > 0
        first_moved = moved_objs[0]

        assert first_moved.displacement >= 0.0
        assert first_moved.transform_matrix is not None
        assert first_moved.transform_matrix.shape == (4, 4)
        # Verify translation vector is in column 3 (index 3)
        trans = first_moved.transform_matrix[:3, 3]
        calc_disp = float(np.linalg.norm(trans))
        assert abs(first_moved.displacement - calc_disp) < 1e-4

    def test_ground_truth_conversion(self, dataset):
        val_pairs = dataset.get_scene_pairs(split="validation")
        pair = val_pairs[0]
        gt = dataset.to_roomshift_ground_truth(pair)
        assert isinstance(gt, list)
        if pair.changes:
            assert len(gt) == len(pair.changes)
            assert all(isinstance(g, GroundTruthChange) for g in gt)

    def test_roomshift_reconstructions_conversion(self, dataset):
        val_pairs = dataset.get_scene_pairs(split="validation")
        pairs_with_moves = [p for p in val_pairs if any(c.change_type == ChangeType.MOVED for c in p.changes)]
        pair = pairs_with_moves[0]

        recon_before, recon_after, alignment, tracked_before, tracked_after = (
            dataset.to_roomshift_reconstructions(pair)
        )

        assert isinstance(recon_before, SceneReconstruction)
        assert isinstance(recon_after, SceneReconstruction)
        assert recon_before.point_cloud.num_points > 0
        assert recon_after.point_cloud.num_points > 0
        assert alignment.transform.shape == (4, 4)
        assert len(tracked_before) > 0
        assert len(tracked_after) > 0

    def test_missing_data_directory_raises_error(self, tmp_path):
        empty_dir = str(tmp_path / "nonexistent")
        adapter = RScanDataset(data_dir=empty_dir)
        with pytest.raises(FileNotFoundError):
            adapter.load()


class TestRScanEvaluation:
    def test_evaluation_run_small(self):
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'external', '3rscan')
        if not os.path.exists(os.path.join(data_dir, "3RScan.json")):
            pytest.skip("3RScan.json not available for test")

        results = evaluate_3rscan_validation(
            data_dir=data_dir,
            max_scenes=3,
            position_threshold=0.25,
        )

        assert results["dataset"] == "3RScan"
        assert results["split"] == "validation"
        assert results["scenes_evaluated"] <= 3
        assert 0.0 <= results["precision"] <= 1.0
        assert 0.0 <= results["recall"] <= 1.0
        assert 0.0 <= results["f1_score"] <= 1.0
        assert "moved" in results["per_class"]
        assert "removed" in results["per_class"]
        assert "added" in results["per_class"]
