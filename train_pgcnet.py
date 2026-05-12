"""
1D CNN + PINN 有监督 SNR 实验
==============================
编码器: 1D CNN (3×64 输入)
PINN 解码器: latent → 64 复值子载波
物理约束: 频率平滑性 + 延迟域稀疏性
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
from utils.noise import add_awgn

# ==================== 配置 ====================
DATA_SOURCE = 'channel_data_InF_30users'
N_FRAMES = 500
BATCH_SIZE = 64
EPOCHS = 20
LR = 2e-4
TRAIN_RATIO = 0.8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SNR_LIST = [-20, -10, 0, 10, 20]
USER_LIST = [10, 20, 30]

# PINN 权重
PHYSICS_WEIGHT = 0.1
SMOOTH_WEIGHT = 1.0
SPARSE_WEIGHT = 0.1
N_EARLY_TAPS = 6

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
MODEL_NAME = '1DCNN-PINN'


# ==================== 数据加载 (1D) ====================
def load_raw_csi_complex(data_dir, n_users, n_frames):
    user_files = sorted(
        [f for f in os.listdir(data_dir) if f.startswith('user_') and f.endswith('.mat')],
        key=lambda x: int(x.split('_')[1].split('.')[0])
    )[:n_users]
    all_data, all_labels = [], []
    for uid, filename in enumerate(user_files):
        filepath = os.path.join(data_dir, filename)
        mat = sio.loadmat(filepath)
        H = mat['H_sc'].astype(np.complex64)
        H = H.T
        total = H.shape[0]
        idx = np.linspace(0, total - 1, n_frames, dtype=int)
        H = H[idx]
        all_data.append(H)
        all_labels.append(np.full(n_frames, uid, dtype=np.int64))
    return all_data, all_labels


def complex_to_1d(H_list, labels_list):
    all_X, all_y = [], []
    for H, y in zip(H_list, labels_list):
        real = np.real(H)
        imag = np.imag(H)
        mag = np.abs(H)
        data = np.stack([real, imag, mag], axis=1)
        all_X.append(data)
        all_y.append(y)
    return np.vstack(all_X), np.concatenate(all_y)


def train_test_split_per_user(X, y, train_ratio=0.8):
    n_classes = len(np.unique(y))
    train_idx, test_idx = [], []
    for c in range(n_classes):
        mask = y == c
        idx = np.where(mask)[0]
        np.random.shuffle(idx)
        n_train = int(len(idx) * train_ratio)
        train_idx.extend(idx[:n_train])
        test_idx.extend(idx[n_train:])
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def minmax_normalize_1d(train_data, test_data):
    n_channels = train_data.shape[1]
    for c in range(n_channels):
        c_min = train_data[:, c, :].min()
        c_max = train_data[:, c, :].max()
        train_data[:, c, :] = (train_data[:, c, :] - c_min) / (c_max - c_min + 1e-8)
        test_data[:, c, :] = (test_data[:, c, :] - c_min) / (c_max - c_min + 1e-8)
    return train_data, test_data


class CSIDataset1D(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# ==================== 1D CNN 编码器 ====================
class CNN1DEncoder(nn.Module):
    """1D CNN 编码器 (去掉最后的 fc)"""
    def __init__(self):
        super(CNN1DEncoder, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv5 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(256)
        self.conv6 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv7 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm1d(512)
        self.conv8 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm1d(512)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool1(self.relu(self.bn4(self.conv4(x))))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.pool2(self.relu(self.bn6(self.conv6(x))))
        x = self.relu(self.bn7(self.conv7(x)))
        x = self.pool3(self.relu(self.bn8(self.conv8(x))))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


# ==================== 1DCNN + PINN 模型 ====================
class PINN_CNN1D(nn.Module):
    """
    1D CNN + PINN 混合模型
    编码器: 1D CNN (3×64)
    分类头: FC(512 → n_users)
    PINN 解码器: FC(512 → 256 → 128) → Ĥ(f) = real64 + imag64
    """
    def __init__(self, num_classes=20):
        super(PINN_CNN1D, self).__init__()
        self.encoder = CNN1DEncoder()
        self.classifier = nn.Linear(512, num_classes)
        self.pinn_decoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
        )

    def forward(self, x):
        latent = self.encoder(x)
        logits = self.classifier(latent)
        H_pred = self.pinn_decoder(latent)
        return logits, H_pred


# ==================== 物理约束损失 ====================
def physics_loss(H_pred, smooth_w=1.0, sparse_w=0.1, n_early_taps=6):
    B = H_pred.shape[0]
    real = H_pred[:, :64]
    imag = H_pred[:, 64:]
    H = torch.complex(real, imag)

    # 频率平滑性
    diff = H[:, 1:] - H[:, :-1]
    diff_power = torch.mean(torch.abs(diff) ** 2)
    H_power = torch.mean(torch.abs(H) ** 2) + 1e-8
    smooth_loss = diff_power / H_power

    # 延迟域稀疏性
    h = torch.fft.ifft(H)
    total_power = torch.sum(torch.abs(h) ** 2, dim=1)
    late_power = torch.sum(torch.abs(h[:, n_early_taps:]) ** 2, dim=1)
    sparse_loss = torch.mean(late_power / (total_power + 1e-8))

    return smooth_w * smooth_loss + sparse_w * sparse_loss


# ==================== 训练 ====================
def train_one_epoch(model, loader, criterion, optimizer, alpha=0.1):
    model.train()
    total_ce, total_phys, correct, total = 0, 0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits, H_pred = model(inputs)
        ce_loss = criterion(logits, labels)
        phys = physics_loss(H_pred, SMOOTH_WEIGHT, SPARSE_WEIGHT, N_EARLY_TAPS)
        loss = ce_loss + alpha * phys
        loss.backward()
        optimizer.step()
        total_ce += ce_loss.item()
        total_phys += phys.item()
        _, pred = torch.max(logits, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    return total_ce / len(loader), total_phys / len(loader), 100 * correct / total


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            logits, _ = model(inputs)
            _, pred = torch.max(logits, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    return 100 * correct / total


# ==================== 单次运行 ====================
def run_one_setting(data_dir, n_users, snr_db, alpha=0.1):
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    H_list, labels_list = load_raw_csi_complex(data_dir, n_users, N_FRAMES)
    if snr_db is not None:
        for i in range(len(H_list)):
            H_list[i], _ = add_awgn(H_list[i], snr_db, seed=SEED + i)

    X, y = complex_to_1d(H_list, labels_list)
    X_train, X_test, y_train, y_test = train_test_split_per_user(X, y, TRAIN_RATIO)
    X_train, X_test = minmax_normalize_1d(X_train, X_test)

    train_loader = DataLoader(CSIDataset1D(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(CSIDataset1D(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    model = PINN_CNN1D(num_classes=n_users)
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    records, best_acc = [], 0.0
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        ce_loss, phys_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, alpha)
        test_acc = evaluate(model, test_loader)
        elapsed = time.time() - t0
        if test_acc > best_acc:
            best_acc = test_acc
        records.append({
            'epoch': epoch,
            'train_ce_loss': round(ce_loss, 4),
            'train_phys_loss': round(phys_loss, 6),
            'train_acc': round(train_acc, 2),
            'test_acc': round(test_acc, 2),
            'best_test_acc': round(best_acc, 2),
        })
        print(f"      Epoch {epoch:2d}/{EPOCHS} | "
              f"CE: {ce_loss:.4f} | Phys: {phys_loss:.6f} | "
              f"Train: {train_acc:.2f}% | Test: {test_acc:.2f}% | "
              f"Best: {best_acc:.2f}% | {elapsed:.1f}s")
    return records, best_acc


# ==================== 绘图 ====================
def plot_results(all_results, output_dir):
    snrs = sorted(set(r['snr'] for r in all_results))
    users = sorted(set(r['n_users'] for r in all_results))

    fig, axes = plt.subplots(len(snrs), len(users), figsize=(15, 20))
    fig.suptitle(f'{MODEL_NAME} under Different SNR and User Counts', fontsize=16, y=0.98)
    for i, snr in enumerate(snrs):
        for j, nu in enumerate(users):
            ax = axes[i, j]
            data = [r for r in all_results if r['snr'] == snr and r['n_users'] == nu]
            if not data:
                ax.set_visible(False); continue
            rec = data[0]['records']
            e = [r['epoch'] for r in rec]
            train_acc = [r['train_acc'] for r in rec]
            test_acc = [r['test_acc'] for r in rec]
            ax.plot(e, train_acc, 'b-', label='Train Acc', lw=1.5)
            ax.plot(e, test_acc, 'r-', label='Test Acc', lw=1.5)
            ax.set_title(f'SNR={snr}dB, Users={nu}', fontsize=11)
            ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)')
            ax.set_ylim(0, 105); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
            best = data[0]['best_acc']
            ax.axhline(y=best, color='gray', ls='--', lw=0.8, alpha=0.5)
            ax.text(EPOCHS * 0.7, best + 1, f'Best: {best:.1f}%', fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fp = os.path.join(output_dir, f'{MODEL_NAME}_snr_experiment.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"\nPlot saved to {fp}")

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for snr in snrs:
        xv, yv = [], []
        for nu in sorted(users):
            d = [r for r in all_results if r['snr'] == snr and r['n_users'] == nu]
            if d: xv.append(nu); yv.append(d[0]['best_acc'])
        ax2.plot(xv, yv, 'o-', label=f'SNR={snr}dB', lw=2, ms=8)
    ax2.set_xlabel('Number of Users'); ax2.set_ylabel('Best Test Accuracy (%)')
    ax2.set_title(f'{MODEL_NAME} - Best Test Accuracy vs User Count')
    ax2.set_xticks(sorted(users)); ax2.grid(True, alpha=0.3); ax2.legend()
    fig2.tight_layout()
    fp2 = os.path.join(output_dir, f'{MODEL_NAME}_snr_summary.png')
    fig2.savefig(fp2, dpi=150, bbox_inches='tight'); plt.close(fig2)
    print(f"Summary plot saved to {fp2}")


# ==================== 主流程 ====================
def main():
    print("=" * 70)
    print(f"{MODEL_NAME} SNR Experiment")
    print(f"  Device: {DEVICE}")
    print(f"  Input: 3×64 (1D frequency axis)")
    print(f"  Physics weight α: {PHYSICS_WEIGHT}")
    print(f"  Total runs: {len(SNR_LIST) * len(USER_LIST)}")
    print("=" * 70)

    data_dir = os.path.join(PROJECT_ROOT, 'preprocess', 'data', DATA_SOURCE)
    if not os.path.isdir(data_dir):
        print(f"ERROR: Data dir not found: {data_dir}"); sys.exit(1)

    all_results = []
    total_runs = len(SNR_LIST) * len(USER_LIST)
    run_idx = 0

    for snr in SNR_LIST:
        for nu in USER_LIST:
            run_idx += 1
            print(f"\n{'─' * 60}")
            print(f"[{run_idx}/{total_runs}] SNR={snr}dB, Users={nu}")
            print(f"{'─' * 60}")
            t_start = time.time()
            records, best_acc = run_one_setting(data_dir, nu, snr, alpha=PHYSICS_WEIGHT)
            elapsed = time.time() - t_start
            all_results.append({'snr': snr, 'n_users': nu, 'best_acc': best_acc, 'records': records})
            print(f"  >>> Best Test Acc: {best_acc:.2f}% | Time: {elapsed:.1f}s")

    # CSV
    csv_rows = []
    for r in all_results:
        for rec in r['records']:
            csv_rows.append({
                'model': MODEL_NAME, 'snr': r['snr'], 'n_users': r['n_users'],
                'epoch': rec['epoch'], 'train_ce_loss': rec['train_ce_loss'],
                'train_phys_loss': rec['train_phys_loss'],
                'train_acc': rec['train_acc'], 'test_acc': rec['test_acc'],
            })
    df = pd.DataFrame(csv_rows)
    df.to_csv(os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_results.csv'), index=False)

    plot_results(all_results, OUTPUT_DIR)

    print(f"\n{'=' * 70}")
    print(f"{MODEL_NAME} - Summary: Best Test Accuracy (%)")
    print(f"{'=' * 70}")
    header = f"{'SNR\\Users':>10}" + "".join(f"{nu:>8}" for nu in USER_LIST)
    print(header); print("-" * len(header))
    for snr in SNR_LIST:
        row = f"{snr:>5}dB"
        for nu in USER_LIST:
            d = [r for r in all_results if r['snr'] == snr and r['n_users'] == nu]
            row += f"{d[0]['best_acc']:>8.1f}" if d else f"{0:>8.1f}"
        print(row)
    print("=" * 70)


if __name__ == '__main__':
    main()
