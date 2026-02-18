"""
Master script that runs all experiment phases sequentially.

Usage:
    python experiments/run_all.py [--genome skull] [--quick]

In --quick mode, uses smaller image sizes (64) and fewer iterations for fast testing.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import subprocess
import time


def run_script(script_path, args, phase_name):
    """Run a script as a subprocess and report results.

    Args:
        script_path: Path to the Python script.
        args: List of command-line arguments.
        phase_name: Human-readable name for logging.

    Returns:
        True if the script succeeded, False otherwise.
    """
    print(f"\n{'#'*70}")
    print(f"  {phase_name}")
    print(f"{'#'*70}")

    cmd = [sys.executable, script_path] + args
    print(f"  Command: {' '.join(cmd)}")
    print()

    start = time.time()
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    elapsed = time.time() - start

    status = "SUCCESS" if result.returncode == 0 else "FAILED"
    print(f"\n  {phase_name}: {status} ({elapsed:.1f}s)")

    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run all experiment phases")
    parser.add_argument('--genome', type=str, default='skull',
                        choices=['skull', 'butterfly', 'apple'],
                        help="Genome to use (default: skull)")
    parser.add_argument('--quick', action='store_true',
                        help="Quick mode: small images, fewer iterations")
    parser.add_argument('--output_dir', type=str, default='output',
                        help="Base output directory (default: output)")
    args = parser.parse_args()

    experiments_dir = os.path.dirname(os.path.abspath(__file__))

    # Determine parameters based on mode
    if args.quick:
        print("QUICK MODE: Using reduced parameters for fast testing")
        img_size = "64"
        phase1_sweeps = "20"
        phase1_topk = "10"
        phase2_iters = "100"
        phase3_iters = "100"
        phase4_gens = "5"
        phase4_pop = "5"
        phase4_sgd = "10"
    else:
        print("FULL MODE: Using default parameters")
        img_size = "256"
        phase1_sweeps = "200"
        phase1_topk = "60"
        phase2_iters = "10000"
        phase3_iters = "10000"
        phase4_gens = "50"
        phase4_pop = "20"
        phase4_sgd = "50"

    results = {}
    total_start = time.time()

    # Phase 1: CPPN Visualization
    results['Phase 1'] = run_script(
        os.path.join(experiments_dir, "phase1_cppn_visualization.py"),
        [
            "--genome", args.genome,
            "--img_size", img_size,
            "--n_random_sweeps", phase1_sweeps,
            "--top_k", phase1_topk,
            "--output_dir", os.path.join(args.output_dir, "phase1"),
        ],
        "Phase 1: CPPN Visualization",
    )

    # Phase 2: KAN-CPPN
    results['Phase 2'] = run_script(
        os.path.join(experiments_dir, "phase2_kan_cppn.py"),
        [
            "--genome", args.genome,
            "--n_iters", phase2_iters,
            "--img_size", img_size if not args.quick else "64",
            "--output_dir", os.path.join(args.output_dir, "phase2"),
        ],
        "Phase 2: KAN-CPPN Training",
    )

    # Phase 3: SwarmKAN
    results['Phase 3'] = run_script(
        os.path.join(experiments_dir, "phase3_swarm_kan.py"),
        [
            "--genome", args.genome,
            "--n_iters", phase3_iters,
            "--img_size", img_size if not args.quick else "64",
            "--output_dir", os.path.join(args.output_dir, "phase3"),
        ],
        "Phase 3: SwarmKAN vs KAN",
    )

    # Phase 4: NES-Memetic KAN
    results['Phase 4'] = run_script(
        os.path.join(experiments_dir, "phase4_memetic_kan.py"),
        [
            "--genome", args.genome,
            "--n_generations", phase4_gens,
            "--pop_size", phase4_pop,
            "--sgd_steps", phase4_sgd,
            "--img_size", img_size if not args.quick else "64",
            "--output_dir", os.path.join(args.output_dir, "phase4"),
        ],
        "Phase 4: NES-Memetic KAN + Weight Reset",
    )

    # Summary
    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  ALL EXPERIMENTS COMPLETE ({total_elapsed:.1f}s)")
    print(f"{'='*70}")
    for phase, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {phase}: {status}")
    print(f"\nResults saved to {args.output_dir}/")

    # Exit with error if any phase failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == '__main__':
    main()
