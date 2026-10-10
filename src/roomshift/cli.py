"""
Command-line interface for RoomShift AI.

Usage:
    python -m roomshift.cli compare --before before.mp4 --after after.mp4 --output outputs/demo
    python -m roomshift.cli evaluate --results outputs/demo/change_report.json --ground-truth gt.json
"""
import argparse
import logging
import sys
import os
import json
import traceback


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )


def cmd_compare(args):
    """Run the comparison pipeline."""
    from .pipeline import run_pipeline
    from .config import PipelineConfig, get_default_config, get_quality_config
    
    # Load config
    if args.config:
        config = PipelineConfig.from_file(args.config)
    elif args.quality:
        config = get_quality_config()
    else:
        config = get_default_config()
    
    config.verbose = args.verbose
    config.output_dir = args.output
    
    try:
        report = run_pipeline(
            before_video=args.before,
            after_video=args.after,
            output_dir=args.output,
            config=config,
        )
        
        print(f"\n✓ Pipeline completed successfully")
        print(f"  Changes detected: {len(report.changes)}")
        print(f"  Output directory: {os.path.abspath(args.output)}")
        print(f"  Time: {report.processing_time_seconds:.1f}s")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"\n✗ Input error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        if args.verbose:
            traceback.print_exc()
        else:
            print(f"\n✗ Pipeline error: {e}", file=sys.stderr)
            print("  Run with --verbose for full traceback", file=sys.stderr)
        return 2


def cmd_evaluate(args):
    """Run evaluation against ground truth."""
    from .evaluation import evaluate_against_ground_truth, save_evaluation_report
    from .types import ChangeReport
    
    try:
        # Load results
        with open(args.results) as f:
            data = json.load(f)
        
        from .types import SpatialChange, ChangeType
        changes = []
        for cd in data.get('changes', []):
            import numpy as np
            changes.append(SpatialChange(
                change_id=cd['change_id'],
                change_type=ChangeType(cd['change_type']),
                object_label=cd.get('object_label'),
                confidence=cd.get('confidence', 0),
                old_position=np.array(cd['old_position']) if cd.get('old_position') else None,
                new_position=np.array(cd['new_position']) if cd.get('new_position') else None,
                displacement=cd.get('displacement'),
                evidence=cd.get('evidence', []),
                limitations=cd.get('limitations', []),
            ))
        
        report = ChangeReport(changes=changes)
        result = evaluate_against_ground_truth(report, args.ground_truth)
        
        print(f"\nEvaluation Results:")
        print(f"  Precision: {result.precision:.3f}")
        print(f"  Recall:    {result.recall:.3f}")
        print(f"  F1 Score:  {result.f1:.3f}")
        print(f"  TP: {result.true_positives}, FP: {result.false_positives}, FN: {result.false_negatives}")
        
        if result.per_category:
            print(f"\nPer-Category:")
            for cat, metrics in result.per_category.items():
                print(f"  {cat}: P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f}")
        
        # Save evaluation
        if args.output:
            save_evaluation_report(result, args.output)
            print(f"\nFull report saved to {args.output}")
        
        return 0
        
    except Exception as e:
        if args.verbose:
            traceback.print_exc()
        else:
            print(f"\n✗ Evaluation error: {e}", file=sys.stderr)
        return 1


def cmd_evaluate_3rscan(args):
    """Run evaluation on real 3RScan validation scenes."""
    from .adapters.evaluate_3rscan import evaluate_3rscan_validation

    try:
        results = evaluate_3rscan_validation(
            data_dir=args.data_dir,
            max_scenes=args.max_scenes,
            position_threshold=args.threshold,
        )

        output_path = args.output or "outputs/3rscan_evaluation.json"
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
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
        if results.get('displacement_mae_meters', 0) > 0:
            print(f"Displacement MAE:     {results['displacement_mae_meters']:.3f} m")
        print("\nPer-Class Breakdown:")
        for cat, m in results['per_class'].items():
            print(f"  {cat.upper():10s} - P: {m['precision']:.3f}, R: {m['recall']:.3f}, F1: {m['f1']:.3f} (TP={m['tp']}, FP={m['fp']}, FN={m['fn']})")
        print("=" * 60)
        print(f"Report saved to: {os.path.abspath(output_path)}\n")
        return 0

    except Exception as e:
        if args.verbose:
            traceback.print_exc()
        else:
            print(f"\n✗ 3RScan evaluation error: {e}", file=sys.stderr)
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog='roomshift',
        description='RoomShift AI — 3D Spatial Change Detection',
    )
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare two video recordings')
    compare_parser.add_argument('--before', '-b', required=True, help='Path to before video')
    compare_parser.add_argument('--after', '-a', required=True, help='Path to after video')
    compare_parser.add_argument('--output', '-o', default='outputs/result', help='Output directory')
    compare_parser.add_argument('--config', '-c', help='Path to config JSON')
    compare_parser.add_argument('--quality', '-q', action='store_true', help='Use higher quality settings')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate results against ground truth')
    eval_parser.add_argument('--results', '-r', required=True, help='Path to change_report.json')
    eval_parser.add_argument('--ground-truth', '-g', required=True, help='Path to ground truth JSON')
    eval_parser.add_argument('--output', '-o', help='Path to save evaluation report')

    # Evaluate 3RScan command
    rscan_parser = subparsers.add_parser('evaluate-3rscan', help='Evaluate against official 3RScan dataset')
    rscan_parser.add_argument('--data-dir', default='data/external/3rscan', help='Path to 3RScan data directory')
    rscan_parser.add_argument('--max-scenes', type=int, default=20, help='Max validation scenes to evaluate')
    rscan_parser.add_argument('--threshold', type=float, default=0.25, help='Spatial change threshold in meters')
    rscan_parser.add_argument('--output', '-o', default='outputs/3rscan_evaluation.json', help='Output JSON path')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    setup_logging(args.verbose)
    
    if args.command == 'compare':
        return cmd_compare(args)
    elif args.command == 'evaluate':
        return cmd_evaluate(args)
    elif args.command == 'evaluate-3rscan':
        return cmd_evaluate_3rscan(args)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
