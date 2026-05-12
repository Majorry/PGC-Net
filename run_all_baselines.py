"""
统一全模型 SNR 实验脚本
=======================
在指定 n_frames/epochs 下，跑完 9 个模型 × 5 SNR × 3 用户数。

用法:
    python run_all_models.py --n_frames 300 --epochs 20
    python run_all_models.py --n_frames 800 --epochs 30
"""
import os, sys, time, warnings, argparse
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
from utils.noise import add_awgn

# ==================== 固定配置 ====================
DATA_SOURCE = 'channel_data_InF_30users'
BATCH_SIZE = 64
LR = 2e-4
TRAIN_RATIO = 0.8
SNR_LIST = [-20, -10, 0, 10, 20]
USER_LIST = [10, 20, 30]
SEED = 42

KNN_NEIGHBORS = 5
CONTRASTIVE_WEIGHT = 0.1
MARGIN = 1.0
PHYSICS_WEIGHT = 0.1
SMOOTH_WEIGHT = 1.0
SPARSE_WEIGHT = 0.1
N_EARLY_TAPS = 6

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==================== 数据加载 (1D, 3×64) ====================
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
    """复值 CSI → (N, 3, 64)"""
    all_X, all_y = [], []
    for H, y in zip(H_list, labels_list):
        real = np.real(H)
        imag = np.imag(H)
        mag = np.abs(H)
        data = np.stack([real, imag, mag], axis=1)
        all_X.append(data)
        all_y.append(y)
    return np.vstack(all_X), np.concatenate(all_y)


def complex_to_2d(H_list, labels_list):
    """复值 CSI → (N, 3, 8, 8) 用于 ResNet18"""
    all_X, all_y = [], []
    for H, y in zip(H_list, labels_list):
        real = np.real(H)
        imag = np.imag(H)
        mag = np.abs(H)
        real_2d = real.reshape(-1, 1, 8, 8)
        imag_2d = imag.reshape(-1, 1, 8, 8)
        mag_2d = mag.reshape(-1, 1, 8, 8)
        data = np.concatenate([real_2d, imag_2d, mag_2d], axis=1)
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
    """逐通道 Min-Max (3×64 格式)"""
    n_channels = train_data.shape[1]
    for c in range(n_channels):
        c_min = train_data[:, c, :].min()
        c_max = train_data[:, c, :].max()
        train_data[:, c, :] = (train_data[:, c, :] - c_min) / (c_max - c_min + 1e-8)
        test_data[:, c, :] = (test_data[:, c, :] - c_min) / (c_max - c_min + 1e-8)
    return train_data, test_data


def minmax_normalize_2d(train_data, test_data):
    """逐通道 Min-Max (3×8×8 格式)"""
    n_channels = train_data.shape[1]
    for c in range(n_channels):
        c_min = train_data[:, c, :, :].min()
        c_max = train_data[:, c, :, :].max()
        train_data[:, c, :, :] = (train_data[:, c, :, :] - c_min) / (c_max - c_min + 1e-8)
        test_data[:, c, :, :] = (test_data[:, c, :, :] - c_min) / (c_max - c_min + 1e-8)
    return train_data, test_data


# ==================== Dataset ====================
class CSIDataset1D(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class CSIDataset2D(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# ==================== 模型定义 ====================

# ----- ResNet18Small (3×8×8) -----
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet18Small(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=1)
        self.layer3 = self._make_layer(128, 256, 2, stride=1)
        self.layer4 = self._make_layer(256, 512, 2, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    def _make_layer(self, in_planes, planes, num_blocks, stride):
        layers = [BasicBlock(in_planes, planes, stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(planes, planes, 1))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)




def physics_loss(H_pred, smooth_w=1.0, sparse_w=0.1, n_early_taps=6):
    B = H_pred.shape[0]
    real = H_pred[:, :64]
    imag = H_pred[:, 64:]
    H = torch.complex(real, imag)
    diff = H[:, 1:] - H[:, :-1]
    diff_power = torch.mean(torch.abs(diff) ** 2)
    H_power = torch.mean(torch.abs(H) ** 2) + 1e-8
    smooth_loss = diff_power / H_power
    h = torch.fft.ifft(H)
    total_power = torch.sum(torch.abs(h) ** 2, dim=1)
    late_power = torch.sum(torch.abs(h[:, n_early_taps:]) ** 2, dim=1)
    sparse_loss = torch.mean(late_power / (total_power + 1e-8))
    return smooth_w * smooth_loss + sparse_w * sparse_loss


# ----- 1DCNN (3×64) -----
class CNN1D(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(2)
        self.conv5 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(256)
        self.conv6 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(2)
        self.conv7 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm1d(512)
        self.conv8 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm1d(512)
        self.pool3 = nn.MaxPool1d(2)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)
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
        return self.fc(x)


# ----- 1DCNN-PINN (3×64) -----
class CNN1DEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(2)
        self.conv5 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(256)
        self.conv6 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(2)
        self.conv7 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm1d(512)
        self.conv8 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm1d(512)
        self.pool3 = nn.MaxPool1d(2)
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
        return x.view(x.size(0), -1)


class PINN_CNN1D(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
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


# ----- Siamese-1DCNN (3×64) -----
class SiameseEncoder1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(2)
        self.conv5 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(256)
        self.conv6 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(2)
        self.conv7 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm1d(512)
        self.conv8 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm1d(512)
        self.pool3 = nn.MaxPool1d(2)
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
        return x.view(x.size(0), -1)


class Siamese1DCNN(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        self.encoder = SiameseEncoder1D()
        self.classifier = nn.Linear(512, num_classes)
    def forward_single(self, x):
        return self.classifier(self.encoder(x))
    def forward_pair(self, x1, x2):
        e1, e2 = self.encoder(x1), self.encoder(x2)
        return self.classifier(e1), self.classifier(e2), e1, e2


def contrastive_loss(e1, e2, labels, margin=1.0):
    dist = torch.sqrt(torch.sum((e1 - e2) ** 2, dim=1) + 1e-8)
    loss = labels * torch.clamp(margin - dist, min=0) ** 2 + (1 - labels) * dist ** 2
    return torch.mean(loss), torch.mean(dist)


class SiameseTrainDataset(Dataset):
    """构造同类/异类对，返回用户标签"""
    def __init__(self, data, labels, n_pairs_per_epoch=2000):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = labels
        self.unqiue = np.unique(labels)
        self.n = n_pairs_per_epoch
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        if np.random.rand() < 0.5:
            c = np.random.choice(self.unqiue)
            mask = np.where(self.labels == c)[0]
            i, j = np.random.choice(mask, 2, replace=False)
            pair_label = 0.0
        else:
            c1, c2 = np.random.choice(self.unqiue, 2, replace=False)
            mask1 = np.where(self.labels == c1)[0]
            mask2 = np.where(self.labels == c2)[0]
            i, j = np.random.choice(mask1), np.random.choice(mask2)
            pair_label = 1.0
        return (self.data[i], self.labels[i],
                self.data[j], self.labels[j],
                torch.tensor(pair_label, dtype=torch.float32))


# ==================== 训练函数 ====================

def train_epoch(model, loader, criterion, optimizer):
    """通用单 epoch 训练（标准分类模型）"""
    model.train()
    total_loss, correct, total = 0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, pred = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    return total_loss / len(loader), 100 * correct / total


def train_pinn_epoch(model, loader, criterion, optimizer, alpha=0.1):
    """PINN 模型训练（分类 + 物理约束）"""
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


def train_siamese_epoch_v2(model, loader, optimizer, criterion_ce, alpha=CONTRASTIVE_WEIGHT, margin=MARGIN):
    model.train()
    total_ce, total_ct, correct, total = 0, 0, 0, 0
    for x1, y1, x2, y2, p_label in loader:
        x1, x2 = x1.to(DEVICE), x2.to(DEVICE)
        y1, y2 = y1.to(DEVICE), y2.to(DEVICE)
        p_label = p_label.to(DEVICE)
        optimizer.zero_grad()
        logits1, logits2, e1, e2 = model.forward_pair(x1, x2)
        ce1 = criterion_ce(logits1, y1)
        ce2 = criterion_ce(logits2, y2)
        ce_loss = (ce1 + ce2) / 2
        ct_loss_val, avg_dist = contrastive_loss(e1, e2, p_label, margin)
        loss = ce_loss + alpha * ct_loss_val
        loss.backward()
        optimizer.step()
        total_ce += ce_loss.item()
        total_ct += ct_loss_val.item()
        _, pred = torch.max(logits1, 1)
        total += y1.size(0)
        correct += (pred == y1).sum().item()
    return total_ce / len(loader), total_ct / len(loader), 100 * correct / total


def compute_metrics(all_labels, all_preds):
    """从预测结果计算 F1(%), TPR(%), FPR(%)"""
    f1 = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)[2]
    cm = confusion_matrix(all_labels, all_preds)
    n = cm.shape[0]
    fp = np.sum(cm, axis=0) - np.diag(cm)
    fn = np.sum(cm, axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = np.sum(cm) - (fp + fn + tp)
    tpr_per = tp / (tp + fn + 1e-15)
    fpr_per = fp / (fp + tn + 1e-15)
    return f1 * 100, np.mean(tpr_per) * 100, np.mean(fpr_per) * 100


def evaluate(model, loader):
    """返回 (accuracy%, f1%, tpr%, fpr%)"""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            _, pred = torch.max(outputs, 1)
            all_preds.append(pred.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = 100 * np.mean(all_preds == all_labels)
    f1, tpr, fpr = compute_metrics(all_labels, all_preds)
    return acc, f1, tpr, fpr


def evaluate_pinn(model, loader):
    """PINN 模型: 返回 (accuracy%, f1%, tpr%, fpr%)"""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits, _ = model(x)
            _, pred = torch.max(logits, 1)
            all_preds.append(pred.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = 100 * np.mean(all_preds == all_labels)
    f1, tpr, fpr = compute_metrics(all_labels, all_preds)
    return acc, f1, tpr, fpr


def evaluate_siamese(model, loader):
    """Siamese 模型: 返回 (accuracy%, f1%, tpr%, fpr%)"""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model.forward_single(x)
            _, pred = torch.max(logits, 1)
            all_preds.append(pred.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = 100 * np.mean(all_preds == all_labels)
    f1, tpr, fpr = compute_metrics(all_labels, all_preds)
    return acc, f1, tpr, fpr


# ==================== KNN/SVM ====================
def run_knn_svm(X_train, y_train, X_test, y_test, n_classes):
    """训练 KNN(raw)/SVM(raw) 并返回 {name: (acc, f1, tpr, fpr)}"""
    B, C, L = X_train.shape
    X_flat = X_train.reshape(B, -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    n_neigh = min(KNN_NEIGHBORS, n_classes)

    results = {}
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_flat)
    X_te = scaler.transform(X_test_flat)

    for name, Cls in [('KNN(raw)', KNeighborsClassifier), ('SVM(raw)', SVC)]:
        kwargs = {'n_neighbors': n_neigh} if name == 'KNN(raw)' else {'kernel': 'rbf', 'decision_function_shape': 'ovr', 'random_state': SEED}
        m = Cls(**kwargs).fit(X_tr, y_train)
        yp = m.predict(X_te)
        acc = 100 * np.mean(yp == y_test)
        f1, tpr, fpr = compute_metrics(y_test, yp)
        results[name] = (acc, f1, tpr, fpr)

    return results


# ==================== 单次运行（一个模型 × 一个 setting）====================
def run_model(model_name, data_dir, n_users, snr_db, n_frames, epochs):
    """运行单个模型在指定配置下的实验"""
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # 加载数据
    H_list, labels_list = load_raw_csi_complex(data_dir, n_users, n_frames)
    if snr_db is not None:
        for i in range(len(H_list)):
            H_list[i], _ = add_awgn(H_list[i], snr_db, seed=SEED + i)

    # 根据模型选择数据格式
    is_2d = (model_name == 'ResNet18')

    if is_2d:
        X, y = complex_to_2d(H_list, labels_list)
        X_train, X_test, y_train, y_test = train_test_split_per_user(X, y, TRAIN_RATIO)
        X_train, X_test = minmax_normalize_2d(X_train, X_test)
        DatasetClass = CSIDataset2D
    else:
        X, y = complex_to_1d(H_list, labels_list)
        X_train, X_test, y_train, y_test = train_test_split_per_user(X, y, TRAIN_RATIO)
        X_train, X_test = minmax_normalize_1d(X_train, X_test)
        DatasetClass = CSIDataset1D

    train_loader = DataLoader(DatasetClass(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(DatasetClass(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    # ---- KNN/SVM 系（不需要训练循环）----
    if model_name in ('KNN(raw)', 'SVM(raw)'):
        knn_svm_results = run_knn_svm(X_train, y_train, X_test, y_test, n_users)
        return knn_svm_results[model_name]  # (acc, f1, tpr, fpr)

    # ---- 深度学习模型 ----
    if model_name == 'ResNet18':
        model = ResNet18Small(num_classes=n_users).to(DEVICE)
    elif model_name == '1DCNN':
        model = CNN1D(num_classes=n_users).to(DEVICE)
    elif model_name == '1DCNN-PINN':
        model = PINN_CNN1D(num_classes=n_users).to(DEVICE)
    elif model_name == 'Siamese-1DCNN':
        model = Siamese1DCNN(num_classes=n_users).to(DEVICE)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        t0 = time.time()

        if model_name == '1DCNN-PINN':
            ce_loss, phys_loss, train_acc = train_pinn_epoch(
                model, train_loader, criterion, optimizer, alpha=PHYSICS_WEIGHT
            )
        elif model_name == 'Siamese-1DCNN':
            n_pairs = min(2000, len(X_train) * 2)
            siamese_train = SiameseTrainDataset(X_train, y_train, n_pairs)
            pair_loader = DataLoader(siamese_train, batch_size=BATCH_SIZE, shuffle=True,
                                      num_workers=0, pin_memory=False)
            ce_loss, ct_loss, train_acc = train_siamese_epoch_v2(
                model, pair_loader, optimizer, criterion,
                alpha=CONTRASTIVE_WEIGHT, margin=MARGIN
            )
        else:
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)

        # 测试
        if model_name == 'Siamese-1DCNN':
            test_acc, test_f1, test_tpr, test_fpr = evaluate_siamese(model, test_loader)
        elif model_name == '1DCNN-PINN':
            test_acc, test_f1, test_tpr, test_fpr = evaluate_pinn(model, test_loader)
        else:
            test_acc, test_f1, test_tpr, test_fpr = evaluate(model, test_loader)

        elapsed = time.time() - t0

        if test_acc > best_acc:
            best_acc = test_acc
            best_f1 = test_f1
            best_tpr = test_tpr
            best_fpr = test_fpr

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"      Epoch {epoch:3d}/{epochs} | "
                  f"Train: {train_acc:.2f}% | Test: {test_acc:.2f}% | "
                  f"Best: {best_acc:.2f}% | {elapsed:.1f}s")

    return best_acc, best_f1, best_tpr, best_fpr


# ==================== 主流程 ====================
def main():
    parser = argparse.ArgumentParser(description='Run all 9 models SNR experiment')
    parser.add_argument('--n_frames', type=int, default=300, help='Frames per user')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    args = parser.parse_args()

    N_FRAMES = args.n_frames
    EPOCHS = args.epochs
    MODEL_NAME_TAG = f'AllModels_{N_FRAMES}frames'

    MODELS = ['ResNet18', '1DCNN', '1DCNN-PINN',
              'KNN(raw)', 'SVM(raw)', 'Siamese-1DCNN']

    print("=" * 70)
    print(f"Unified All-Models SNR Experiment ({N_FRAMES} frames/user)")
    print(f"  Device: {DEVICE}")
    print(f"  Models: {len(MODELS)} ({', '.join(MODELS)})")
    print(f"  SNR: {SNR_LIST}")
    print(f"  Users: {USER_LIST}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Total runs: {len(MODELS) * len(SNR_LIST) * len(USER_LIST)}")
    print("=" * 70)

    data_dir = os.path.join(PROJECT_ROOT, 'preprocess', 'data', DATA_SOURCE)
    if not os.path.isdir(data_dir):
        print(f"ERROR: Data dir not found: {data_dir}")
        sys.exit(1)

    all_rows = []
    total_runs = len(MODELS) * len(SNR_LIST) * len(USER_LIST)
    run_idx = 0

    for model_name in MODELS:
        for snr in SNR_LIST:
            for nu in USER_LIST:
                run_idx += 1
                print(f"\n{'─' * 60}")
                print(f"[{run_idx}/{total_runs}] {model_name} | SNR={snr}dB | Users={nu}")
                print(f"{'─' * 60}")

                t_start = time.time()
                acc, f1, tpr, fpr = run_model(model_name, data_dir, nu, snr, N_FRAMES, EPOCHS)
                elapsed = time.time() - t_start

                all_rows.append({
                    'model': model_name,
                    'snr': snr,
                    'n_users': nu,
                    'n_frames': N_FRAMES,
                    'best_test_acc': round(acc, 2),
                    'best_f1': round(f1, 2),
                    'best_tpr': round(tpr, 2),
                    'best_fpr': round(fpr, 2),
                })

                print(f"  >>> Best Test Acc: {acc:.2f}% | F1: {f1:.2f}% | TPR: {tpr:.2f}% | FPR: {fpr:.2f}% | Time: {elapsed:.1f}s")

    # ---- 保存 CSV ----
    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(OUTPUT_DIR, f'{MODEL_NAME_TAG}_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # ---- 汇总表 ----
    print(f"\n{'=' * 70}")
    print(f"Summary: Best Test Accuracy (%) — {N_FRAMES} frames/user")
    print(f"{'=' * 70}")
    for model_name in MODELS:
        print(f"\n--- {model_name} ---")
        header = f"{'SNR\\Users':>10}" + "".join(f"{nu:>8}" for nu in USER_LIST)
        print(header)
        print("-" * len(header))
        for snr in SNR_LIST:
            row = f"{snr:>5}dB"
            for nu in USER_LIST:
                vals = [r['best_test_acc'] for r in all_rows
                        if r['model'] == model_name and r['snr'] == snr and r['n_users'] == nu]
                row += f"{vals[0]:>8.1f}" if vals else f"{'':>8}"
            print(row)

    # ---- 合并到总 CSV ----
    merge_to_master(csv_path)

    print(f"\n{'=' * 70}")
    print(f"Done! All results for {N_FRAMES} frames complete.")
    print(f"{'=' * 70}")


def merge_to_master(new_csv_path):
    """将新结果合并到 master_summary.csv"""
    master_path = os.path.join(OUTPUT_DIR, 'master_summary.csv')

    new_df = pd.read_csv(new_csv_path)

    if os.path.exists(master_path):
        old_df = pd.read_csv(master_path)
        # 补齐旧数据缺失的列
        for col in ['best_f1', 'best_tpr', 'best_fpr']:
            if col not in old_df.columns:
                old_df[col] = None
        merged = pd.concat([old_df, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=['model', 'snr', 'n_users', 'n_frames'], keep='last')
    else:
        merged = new_df

    # 按 model, n_frames, snr, n_users 排序
    merged = merged.sort_values(['model', 'n_frames', 'snr', 'n_users']).reset_index(drop=True)
    merged.to_csv(master_path, index=False)
    print(f"Master CSV updated: {master_path}")


if __name__ == '__main__':
    main()
