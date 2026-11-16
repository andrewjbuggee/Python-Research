#!/bin/bash
# ============================================================
# 12_train_configs_alpine.sh — STEP 12b of the paper-3 pipeline
# (also trains the trials of an 11b sweep: TAG=<sweep tag>; same layout,
#  training_runs/<tag>/{configs,runs.txt,results})
#
# Trains the profile + τ network (DropletProfileNetwork: 7-level r_e + σ and
# τ_c + σ from 636 HySICS reflectances + 4 angles) for every config chosen by
# 12_select_configs.py, one config per array task, on Alpine's L40 GPUs.
#
# Same trainer, seed and loss as the Aug-2026 PT sweep and the long retrain
# (shell_scripts/standalone_profile_tau_al40.sh):
#     hyper_parameter_sweep/sweep_train_profile_tau_synthetic.py
# Only the data, split sizes and epoch budget come from the new configs.
#
# Submit from lasp-CU-paper-3/ (12_select_configs.py prints this line):
#     mkdir -p logs
#     sbatch --array=0-<N-1>%6 --export=ALL,TAG=<tag>,H5_PATH=<new .h5> 12_train_configs_alpine.sh
#
# Wall time: run_004 took ~35 min per 1,500 epochs on the 42k-sample file;
# with ~2.4x the data and up to 4,500 epochs that is ~4 h if early stopping
# never triggers, so 8 h leaves ~2x margin.
# ============================================================
#SBATCH --account=ucb762_asc1
#SBATCH --partition=al40
#SBATCH --qos=gpu-normal
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40:1           # al40 requires the GRES type spelled out
#SBATCH --mem=16G                  # ~100k x 636 float32 x 4 reflectance arrays ~ 1 GB, plus loaders
#SBATCH --job-name=PT_retrain
#SBATCH --output=logs/PT_retrain_%A_%a.out
#SBATCH --error=logs/PT_retrain_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrew.buggee@colorado.edu

REPO=/projects/anbu8374/Python-Research/lasp-CU-paper-3
: "${TAG:?set TAG via --export=ALL,TAG=<tag>}"
: "${H5_PATH:?set H5_PATH via --export=ALL,H5_PATH=<new .h5>}"
RUNS_FILE="$REPO/training_runs/$TAG/runs.txt"
RESULTS_DIR="$REPO/training_runs/$TAG/results"

# Array task t trains the (t+1)-th run id listed in runs.txt
RUN_ID=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$RUNS_FILE")
if [ -z "$RUN_ID" ]; then
    echo "No run id on line $((SLURM_ARRAY_TASK_ID + 1)) of $RUNS_FILE"; exit 1
fi
CONFIG_FILE="$REPO/training_runs/$TAG/configs/run_$(printf '%03d' "$RUN_ID").json"

echo "============================================"
echo "Job/array:  $SLURM_ARRAY_JOB_ID / $SLURM_ARRAY_TASK_ID   Node: $SLURM_NODELIST   GPU: $CUDA_VISIBLE_DEVICES"
echo "Start:      $(date)"
echo "Tag:        $TAG    Run id: $RUN_ID"
echo "Config:     $CONFIG_FILE"
echo "HDF5:       $H5_PATH"
echo "Results:    $RESULTS_DIR"
echo "============================================"

# ── Python environment ──────────────────────────────────────────────────────
# 15 Nov 2026: every task of job 35753371 failed here with Lmod
# 'module(s) are unknown: "anaconda"' on c3gpu-a9-u15-1, the node where the
# same three lines worked in Aug 2026. So: try the anaconda module, then
# miniforge (CURC's conda module where anaconda is not offered), and if
# neither loads, fall back to the env's own bin/ (a conda env is a
# self-contained prefix, so its python runs without `conda activate`).
# Then stop with a clear message if python/torch are not usable, instead of
# letting all array tasks fail silently.
ENV_PREFIX=/projects/anbu8374/software/anaconda/envs/dropProfs_nn
module purge
CONDA_MODULE=""
for m in anaconda miniforge; do
    if module load "$m" 2>/dev/null && command -v conda >/dev/null 2>&1; then
        CONDA_MODULE=$m; break
    fi
done
if [ -n "$CONDA_MODULE" ]; then
    echo "Environment: module $CONDA_MODULE + conda activate $ENV_PREFIX"
    conda activate "$ENV_PREFIX"
else
    echo "Environment: no conda module available; using $ENV_PREFIX/bin directly"
    export PATH="$ENV_PREFIX/bin:$PATH"
fi
echo "MODULEPATH: $MODULEPATH"
python -c "import sys, torch; print('python', sys.executable); print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
    || { echo "ERROR: python/torch not usable from $ENV_PREFIX"; exit 2; }

for f in "$CONFIG_FILE" "$H5_PATH"; do
    [ -f "$f" ] || { echo "ERROR: not found: $f"; exit 1; }
done
mkdir -p "$RESULTS_DIR" "$REPO/logs"

cd "$REPO/hyper_parameter_sweep"
# n_epochs and early_stop_patience come from the config written by 12_select_configs.py
time python sweep_train_profile_tau_synthetic.py \
    --config-json "$CONFIG_FILE" \
    --h5-path     "$H5_PATH" \
    --output-dir  "$RESULTS_DIR"
EXIT_CODE=$?

echo "End: $(date)   exit code: $EXIT_CODE"
exit $EXIT_CODE
