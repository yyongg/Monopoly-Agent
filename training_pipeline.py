"""Run the full Monopoly RL pipeline: train -> evaluate -> simulate.

Chains the three stages that are normally run by hand:

    PYTHONPATH=. python training/train_selfplay.py --timesteps ... --n-envs ... --fp-prob ...
    PYTHONPATH=. python -m validation.evaluate  <model> --episodes ... --opponent fp --plot
    PYTHONPATH=. python -m validation.simulate  <model> --games    ... --plot

The knobs that change between runs are exposed as flags -- ``--n-envs`` (train),
``--episodes`` (evaluate) and ``--games`` (simulate) -- while ``--timesteps``,
``--fp-prob`` and the shared model path round out the common case. Evaluation runs
against the hand-crafted FP-A/B/C trio (a meaningful yardstick; the trivial engine
baseline is too weak to be informative). Any stage can be skipped so you can, e.g.,
re-run just evaluation against an already-trained model.

Usage:
    python training_pipeline.py --n-envs 64 --episodes 200 --games 100
    python training_pipeline.py --skip-train            # evaluate + simulate only
"""

import argparse
import os
import subprocess
import sys
import time


def run_stage(name, cmd):
    """Run one pipeline stage as a subprocess, echoing the command it runs.

    Inherits stdout/stderr so live training/eval output streams through. Sets
    ``PYTHONPATH`` to the repo root (mirroring the manual ``PYTHONPATH=.``
    invocations) so the ``training`` / ``validation`` packages import. Raises on a
    non-zero exit so a failed stage aborts the pipeline instead of feeding a
    missing model into the next step.
    """
    repo_root = os.path.dirname(os.path.abspath(__file__))
    env = dict(os.environ)
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")

    print(f"\n{'=' * 70}\n[{name}] {' '.join(cmd)}\n{'=' * 70}", flush=True)
    start = time.time()
    subprocess.run(cmd, check=True, cwd=repo_root, env=env)
    print(f"[{name}] done in {time.time() - start:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # The three knobs the user asked to vary, one per stage.
    #
    # --n-envs is a MEMORY knob, not a speed knob. Each env is a worker *process*
    # holding its own torch runtime and its own cache of opponent snapshots, so
    # raising it costs RAM (see training/selfplay.py::memory_budget_gb, and mind
    # the job's cgroup cap). What it does NOT do is make training much faster:
    # PPO's update is serial and its cost grows with n_envs (the rollout buffer
    # is n_envs x n_steps), so throughput saturates. Measured here: 8 envs = 1134
    # steps/s, 64 envs = 1420 -- 8x the envs and the memory for 25% more speed.
    # 32 is the knee. For real speed, shrink the SERIAL half: --torch-threads and
    # --n-epochs below.
    parser.add_argument("--n-envs", type=int, default=32,
                        help="parallel envs for training. A memory knob, not a "
                             "speed one -- throughput saturates around here")
    parser.add_argument("--episodes", type=int, default=200,
                        help="episodes for evaluation")
    parser.add_argument("--games", type=int, default=100,
                        help="games for simulation")
    # Rounding out the common case.
    parser.add_argument("--timesteps", type=int, default=2_000_000,
                        help="training timesteps")
    parser.add_argument("--fp-prob", type=float, default=0.3,
                        help="probability an episode trains against the FP-A/B/C "
                             "trio (passed through to train_selfplay.py)")
    # The two knobs that actually buy wall clock, both by shrinking PPO's serial
    # update. --torch-threads is free (same maths, more cores). --n-epochs is a
    # genuine trade: fewer passes over each rollout is faster per step but reuses
    # each sample less, so it may need more steps to reach the same skill. Left
    # at SB3's 10 by default -- lower it deliberately, and watch the win rate.
    parser.add_argument("--torch-threads", type=int, default=4,
                        help="threads for the learner's gradient update "
                             "(~+18%% throughput at 4; flat beyond)")
    parser.add_argument("--n-epochs", type=int, default=10,
                        help="PPO epochs per rollout. 4 measured ~+33%% "
                             "throughput over 10, at some sample efficiency")
    parser.add_argument("--model", default="runs/monopoly_ppo",
                        help="model path: train writes here, eval/sim read it")
    # Let any stage be skipped so stages can be re-run independently.
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    parser.add_argument("--skip-simulate", action="store_true")
    args = parser.parse_args()

    py = sys.executable

    if not args.skip_train:
        run_stage("train", [
            py, "training/train_selfplay.py",
            "--timesteps", str(args.timesteps),
            "--n-envs", str(args.n_envs),
            "--fp-prob", str(args.fp_prob),
            "--torch-threads", str(args.torch_threads),
            "--n-epochs", str(args.n_epochs),
            "--save-path", args.model,
        ])

    if not args.skip_evaluate:
        # Evaluate against the hand-crafted FP-A/B/C trio -- a meaningful
        # stationary yardstick (the trivial engine baseline is too weak).
        run_stage("evaluate", [
            py, "-m", "validation.evaluate", args.model,
            "--episodes", str(args.episodes),
            "--opponent", "fp",
            "--plot",
        ])

    if not args.skip_simulate:
        run_stage("simulate", [
            py, "-m", "validation.simulate", args.model,
            "--games", str(args.games),
            "--plot",
        ])

    print("\nPipeline complete.", flush=True)


if __name__ == "__main__":
    main()
