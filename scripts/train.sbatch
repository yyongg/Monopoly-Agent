#!/bin/bash
# Submit a self-play training run as its own SLURM job.
#
#   sbatch scripts/train.sbatch                      # defaults below
#   sbatch --mem=96G --cpus-per-task=128 scripts/train.sbatch --n-envs 128
#
# Anything after the script name is passed straight through to
# training/train_selfplay.py.
#
# WHY A BATCH JOB AND NOT THE OOD SESSION:
#   The memory cap is a property of the SLURM *job*, fixed at submission and not
#   writable from inside it (the cgroup's memory.max is root-owned). An Open
#   OnDemand VS Code session is a job too -- one launched with whatever Memory
#   and Time the OOD form asked for, typically 32 GB / a few hours. That is both
#   too little RAM for a wide run and too little wall clock for a 15M-step one.
#   Requesting your own job is the only way to change either.
#
# SIZING --mem:
#   ~2.5 GB for the learner + ~0.4 GB per env worker (each is a separate process
#   paying for its own torch runtime), + ~0.02 GB per cached opponent snapshot
#   per worker. training/selfplay.py::memory_budget_gb is the source of truth,
#   and train_selfplay.py refuses to start if the request will not fit.
#
#     --n-envs 32   ->  ~19 GB   (fits the default 32 GB OOD session)
#     --n-envs 64   ->  ~36 GB   -> --mem=48G
#     --n-envs 128  ->  ~70 GB   -> --mem=96G
#     --n-envs 192  ->  ~104 GB  -> --mem=128G
#
#   Keep --cpus-per-task >= --n-envs, or the workers just timeshare and the extra
#   envs buy you nothing.

#SBATCH --job-name=monopoly-selfplay
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=runs/slurm/%x-%j.out
#SBATCH --error=runs/slurm/%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$(realpath "$0")")")}"
mkdir -p runs/slurm

if [ -d .venv ]; then
    source .venv/bin/activate
fi

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:-<none>}"
echo "cpus      : ${SLURM_CPUS_PER_TASK:-?}"
echo "mem       : ${SLURM_MEM_PER_NODE:-?} MB"
echo "args      : $*"
echo

# train_selfplay.py prints its own memory budget vs. the cgroup cap and exits
# early if the two do not reconcile, so a misprovisioned job fails in seconds
# rather than after an hour of rollouts.
srun python training/train_selfplay.py "$@"
