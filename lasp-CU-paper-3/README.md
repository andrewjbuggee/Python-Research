# Paper 3: Physics-Informed Neural Networks for Cloud Droplet Profile Retrieval

Andrew J. Buggee, LASP / CU Boulder

## Project Structure

```
lasp-CU-paper-3/
├── README.md
├── train.py                  # SLURM-submittable training script
├── configs/
│   └── stage1_modis.yaml     # Training configuration
├── scripts/
│   └── train_alpine.sh       # SLURM submission script for Alpine
├── src/
│   ├── __init__.py
│   ├── data.py               # Data loading, HDF5 datasets, VOCALS-REx
│   └── models.py             # Network architectures and loss functions
├── notebooks/
│   ├── 01_data_preparation.ipynb       # Generate libRadtran training data
│   ├── 02_training_experimentation.ipynb  # Interactive training and tuning
│   └── 03_vocals_validation.ipynb      # Validation and figure generation
├── data/                     # Training data (HDF5, not in git)
├── checkpoints/              # Model checkpoints (not in git)
└── figures/                  # Publication figures
```

## Quick Start

1. **Generate training data** (Notebook 01)
   - Creates synthetic data for development
   - Template for libRadtran integration on Alpine

2. **Train the network** (Notebook 02 or train.py)
   - Interactive training in Jupyter for experimentation
   - `train.py` with SLURM for full training on Alpine

3. **Validate and make figures** (Notebook 03)
   - Compare with VOCALS-REx in situ data
   - Generate publication figures

## Stage 1 vs Stage 2

**Stage 1 (Direct Retrieval):** Train a supervised network on libRadtran
simulations with physics-based output penalties. This is implemented and
ready to use.

**Stage 2 (Full PINN with Emulator):** Add a differentiable forward model
emulator to the training loop. The retrieval network from Stage 1 is
reused — the emulator is an add-on. `ForwardModelEmulator` class is
defined in `models.py` but not yet trained.

## Alpine Setup

```bash
# Create conda environment
conda create -n paper3 python=3.11 pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
conda activate paper3
pip install h5py pyyaml matplotlib scipy

# Submit training job
sbatch scripts/train_alpine.sh
```
