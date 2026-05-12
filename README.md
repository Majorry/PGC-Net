# PGC-Net: Physics-Guided Convolutional Network for Physical Layer Authentication

Physics-Guided Deep Representation Learning for Robust Physical Layer Authentication in UAV Networks.

## Project Structure

```
PGC-Net/
├── train_pgcnet.py          # PGC-Net (1DCNN-PINN) training and evaluation
├── run_all_baselines.py     # Unified script for all models (ResNet18, 1DCNN, Siamese-1DCNN, KNN, SVM, PGC-Net)
├── utils/
│   └── noise.py             # AWGN noise injection utility
├── requirements.txt
└── README.md
```

## Requirements

```bash
pip install -r requirements.txt
```

## Data Preparation

Place `.mat` files under `preprocess/data/channel_data_InF_30users/` with naming convention `user_01.mat`, `user_02.mat`, etc. Each file should contain a complex matrix `H_sc` of shape `(n_subcarriers, n_frames)`.

## Usage

### Train PGC-Net only

```bash
# Run PGC-Net across all SNR levels and user counts
python train_pgcnet.py
```

This runs PGC-Net (1DCNN-PINN) on SNR ∈ {-20, -10, 0, 10, 20} dB × users ∈ {10, 20, 30}, producing CSV results and plots under `experiments/results/`.

### Run all baselines

```bash
# Run all 6 models with 300 frames, 20 epochs
python run_all_baselines.py --n_frames 300 --epochs 20

# Run all 6 models with 800 frames, 30 epochs
python run_all_baselines.py --n_frames 800 --epochs 30
```

Models included: ResNet18, 1DCNN, 1DCNN-PINN (PGC-Net), Siamese-1DCNN, KNN(raw), SVM(raw).

## Model Architecture

- **Encoder**: 1D CNN with 8 convolutional layers (4 stages: 64→128→256→512 channels), BatchNorm, ReLU, MaxPool1d, AdaptiveAvgPool1d → 512-d latent vector.
- **Classification Head**: Linear(512 → K) for K-user classification.
- **PINN Decoder**: MLP (512 → 256 → 128) reconstructing real + imaginary parts of channel frequency response.
- **Physics Constraints**:
  - Frequency-domain smoothness: penalizes rapid variation across adjacent subcarriers.
  - Delay-domain sparsity: concentrates energy in early IFFT taps.

## Training Loss

```
L_total = L_CE + α × L_physics + β × L_recon
```

where `L_physics = λ_smooth × L_smooth + λ_sparse × L_sparse`.

Default hyperparameters: α=0.1, λ_smooth=1.0, λ_sparse=0.1, N_early_taps=6.
