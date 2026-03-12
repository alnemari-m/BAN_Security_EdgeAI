"""
Synchronization Sensitivity Analysis for PhysioKey
====================================================
Measures how EER degrades when paired signal windows are offset in time.
Uses the PTB-XL dataset with a single trained model (fold 0 of 5-fold CV).

Offsets tested: 0, ±10, ±25, ±50, ±100, ±200 samples at 256 Hz
  = 0, ±39ms, ±98ms, ±195ms, ±391ms, ±781ms
"""
import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, filtfilt
from sklearn.metrics import roc_curve, auc
import wfdb

# Reuse components from main simulation
sys.path.insert(0, os.path.dirname(__file__))

DEVICE = torch.device('cpu')

# ---- Model (same as run_simulation.py) ----
class PhysioKeyCNN(nn.Module):
    def __init__(self, embed_dim=32):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm1d(8)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(16)
        self.conv3 = nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, embed_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.mean(dim=2)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data)


def cosine_similarity(z1, z2):
    z1_norm = z1 / (np.linalg.norm(z1, axis=1, keepdims=True) + 1e-8)
    z2_norm = z2 / (np.linalg.norm(z2, axis=1, keepdims=True) + 1e-8)
    return np.sum(z1_norm * z2_norm, axis=1)


def compute_eer(intra_scores, inter_scores):
    labels = np.concatenate([np.ones(len(intra_scores)), np.zeros(len(inter_scores))])
    scores = np.concatenate([intra_scores, inter_scores])
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    roc_auc = auc(fpr, tpr)
    return eer, thresholds[idx], roc_auc


# NT-Xent + decorrelation + alignment loss
def contrastive_loss(z1, z2, temperature=0.05, decorr_lambda=0.05, align_lambda=0.5):
    batch_size = z1.shape[0]
    z1_norm = F.normalize(z1, dim=1)
    z2_norm = F.normalize(z2, dim=1)
    sim_pos = (z1_norm * z2_norm).sum(dim=1) / temperature
    z_all = torch.cat([z1_norm, z2_norm], dim=0)
    sim_matrix = torch.mm(z_all, z_all.t()) / temperature
    mask = torch.eye(2 * batch_size, device=z1.device).bool()
    sim_matrix.masked_fill_(mask, -1e9)
    pos_sim = torch.cat([sim_pos, sim_pos], dim=0)
    logsumexp = torch.logsumexp(sim_matrix, dim=1)
    contrastive = (-pos_sim + logsumexp).mean()

    # Decorrelation
    z_centered = z1 - z1.mean(dim=0)
    cov = (z_centered.t() @ z_centered) / (batch_size - 1)
    std = z_centered.std(dim=0) + 1e-4
    corr = cov / (std.unsqueeze(0) * std.unsqueeze(1))
    decorr = (corr - torch.eye(z1.shape[1], device=z1.device)).pow(2).sum()

    # Alignment
    alignment = (z1 - z2).pow(2).sum(dim=1).mean()

    return contrastive + decorr_lambda * decorr + align_lambda * alignment


def load_ptbxl_raw(data_dir, target_fs=256, max_patients=500, seed=42):
    """Load PTB-XL as raw continuous signals (not windowed) for flexible offsetting."""
    np.random.seed(seed)
    records = []
    for root, dirs, files in os.walk(data_dir):
        for f in sorted(files):
            if f.endswith('_hr.hea') and not f.startswith('.'):
                records.append(os.path.join(root, f.replace('.hea', '')))

    np.random.shuffle(records)
    records = records[:max_patients]

    patient_signals = []  # list of (lead1_continuous, lead2_continuous, patient_id)

    for rec_path in records:
        try:
            record = wfdb.rdrecord(rec_path)
            sig = record.p_signal
            fs = record.fs
            sig_names = [s.upper() for s in record.sig_name]

            lead1_idx = next((i for i, n in enumerate(sig_names) if n in ('I', 'MLI')), None)
            lead2_idx = next((i for i, n in enumerate(sig_names) if n in ('II', 'MLII')), None)
            if lead1_idx is None or lead2_idx is None:
                continue

            lead1 = sig[:, lead1_idx].astype(np.float64)
            lead2 = sig[:, lead2_idx].astype(np.float64)

            if np.any(np.isnan(lead1)) or np.any(np.isnan(lead2)):
                continue

            # Resample
            if fs != target_fs:
                n_target = int(len(lead1) * target_fs / fs)
                lead1 = np.interp(np.linspace(0, len(lead1)-1, n_target), np.arange(len(lead1)), lead1)
                lead2 = np.interp(np.linspace(0, len(lead2)-1, n_target), np.arange(len(lead2)), lead2)

            # Bandpass
            lead1 = bandpass_filter(lead1, 0.5, 40.0, target_fs)
            lead2 = bandpass_filter(lead2, 0.5, 40.0, target_fs)

            # Normalize
            if np.std(lead1) < 1e-6 or np.std(lead2) < 1e-6:
                continue
            lead1 = (lead1 - np.mean(lead1)) / np.std(lead1)
            lead2 = (lead2 - np.mean(lead2)) / np.std(lead2)

            patient_signals.append((lead1.astype(np.float32), lead2.astype(np.float32), rec_path))

        except Exception:
            continue

    print(f"  Loaded {len(patient_signals)} patients with continuous signals")
    return patient_signals


def extract_offset_windows(patient_signals, window_len=128, offset=0, max_per_patient=50):
    """Extract paired windows with a given sample offset between lead1 and lead2.

    lead1 windows start at position w*window_len
    lead2 windows start at position w*window_len + offset
    """
    windows_s1, windows_s2, pids = [], [], []

    for lead1, lead2, pid in patient_signals:
        count = 0
        n_windows = (min(len(lead1), len(lead2)) - abs(offset)) // window_len
        for w in range(n_windows):
            if count >= max_per_patient:
                break
            start1 = w * window_len
            start2 = w * window_len + offset

            if start2 < 0 or start2 + window_len > len(lead2):
                continue
            if start1 + window_len > len(lead1):
                continue

            w1 = lead1[start1:start1 + window_len]
            w2 = lead2[start2:start2 + window_len]

            if np.std(w1) < 0.1 or np.std(w2) < 0.1:
                continue

            windows_s1.append(w1)
            windows_s2.append(w2)
            pids.append(pid)
            count += 1

    return (np.array(windows_s1, dtype=np.float32),
            np.array(windows_s2, dtype=np.float32),
            np.array(pids))


def train_model(windows_s1, windows_s2, patient_ids, config_override=None):
    """Train a PhysioKeyCNN on aligned (offset=0) windows."""
    from torch.utils.data import Dataset, DataLoader

    class PairDataset(Dataset):
        def __init__(self, w1, w2, pids):
            self.w1 = torch.FloatTensor(w1).unsqueeze(1)
            self.w2 = torch.FloatTensor(w2).unsqueeze(1)
            self.pids = pids

        def __len__(self):
            return len(self.w1)

        def __getitem__(self, idx):
            return self.w1[idx], self.w2[idx], 0

    model = PhysioKeyCNN(embed_dim=32).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

    dataset = PairDataset(windows_s1, windows_s2, patient_ids)
    loader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)

    model.train()
    for epoch in range(100):  # 100 epochs for speed
        for x1, x2, _ in loader:
            x1, x2 = x1.to(DEVICE), x2.to(DEVICE)
            z1 = model(x1)
            z2 = model(x2)
            loss = contrastive_loss(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % 25 == 0:
            print(f"    Epoch {epoch+1}/100, loss: {loss.item():.4f}")

    return model


def evaluate_at_offset(model, patient_signals, offset, window_len=128):
    """Evaluate EER at a given offset."""
    windows_s1, windows_s2, pids = extract_offset_windows(
        patient_signals, window_len=window_len, offset=offset
    )

    if len(windows_s1) < 100:
        return None

    model.eval()
    with torch.no_grad():
        z1_list, z2_list = [], []
        bs = 512
        t_s1 = torch.FloatTensor(windows_s1).unsqueeze(1)
        t_s2 = torch.FloatTensor(windows_s2).unsqueeze(1)
        for i in range(0, len(t_s1), bs):
            z1_list.append(model(t_s1[i:i+bs]).numpy())
            z2_list.append(model(t_s2[i:i+bs]).numpy())
        z1_all = np.concatenate(z1_list)
        z2_all = np.concatenate(z2_list)

    # Intra-body: same patient, paired windows
    intra_cos = cosine_similarity(z1_all, z2_all)

    # Inter-body: different patients
    unique_pids = list(set(pids))
    inter_cos = []
    n_inter = min(10000, len(z1_all) * 10)
    for _ in range(n_inter):
        p1, p2 = np.random.choice(len(unique_pids), 2, replace=False)
        pid1, pid2 = unique_pids[p1], unique_pids[p2]
        idx1 = np.where(pids == pid1)[0]
        idx2 = np.where(pids == pid2)[0]
        i1, i2 = np.random.choice(idx1), np.random.choice(idx2)
        cos = cosine_similarity(z1_all[i1:i1+1], z2_all[i2:i2+1])[0]
        inter_cos.append(cos)
    inter_cos = np.array(inter_cos)

    eer, thresh, roc_auc = compute_eer(intra_cos, inter_cos)

    # BDR at offset
    all_z = np.concatenate([z1_all, z2_all])
    z_min = all_z.min(axis=0)
    z_max = all_z.max(axis=0)
    z_range = z_max - z_min
    z_range[z_range < 1e-8] = 1.0

    def quantize_1bit(z):
        normalized = (z - z_min) / z_range
        return (normalized > 0.5).astype(np.int8)

    bits1 = quantize_1bit(z1_all)
    bits2 = quantize_1bit(z2_all)
    hd = np.mean(bits1 != bits2, axis=1)
    bdr = float(np.mean(hd))

    return {
        'offset_samples': int(offset),
        'offset_ms': round(offset / 256 * 1000, 1),
        'eer': round(float(eer) * 100, 1),
        'auc': round(float(roc_auc), 3),
        'bdr': round(bdr, 3),
        'intra_cos_mean': round(float(np.mean(intra_cos)), 3),
        'inter_cos_mean': round(float(np.mean(inter_cos)), 3),
        'n_pairs': len(windows_s1),
    }


def main():
    print("=" * 60)
    print("PhysioKey Synchronization Sensitivity Analysis")
    print("=" * 60)

    data_dir = os.path.join(os.path.dirname(__file__), 'ptbxl_data')

    # Step 1: Load raw continuous signals
    print("\n[1/3] Loading PTB-XL continuous signals...")
    patient_signals = load_ptbxl_raw(data_dir, max_patients=500)

    # Step 2: Split into train/test (80/20 by patient)
    np.random.seed(42)
    np.random.shuffle(patient_signals)
    n_train = int(0.8 * len(patient_signals))
    train_signals = patient_signals[:n_train]
    test_signals = patient_signals[n_train:]
    print(f"  Train: {len(train_signals)} patients, Test: {len(test_signals)} patients")

    # Extract aligned windows for training
    train_s1, train_s2, train_pids = extract_offset_windows(train_signals, offset=0)
    print(f"  Training windows: {len(train_s1)}")

    # Step 3: Train model on aligned data
    print("\n[2/3] Training model on aligned (offset=0) windows...")
    model = train_model(train_s1, train_s2, train_pids)

    # Step 4: Evaluate at various offsets
    print("\n[3/3] Evaluating at different synchronization offsets...")
    offsets = [0, 10, 13, 25, 50, 100, 200]  # samples at 256 Hz
    results = []

    for offset in offsets:
        offset_ms = round(offset / 256 * 1000, 1)
        print(f"\n  Offset: {offset} samples ({offset_ms} ms)...")
        r = evaluate_at_offset(model, test_signals, offset)
        if r:
            results.append(r)
            print(f"    EER: {r['eer']}%, AUC: {r['auc']}, BDR: {r['bdr']}, "
                  f"Intra cos: {r['intra_cos_mean']}, n={r['n_pairs']}")

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Offset (samples)':<18} {'Offset (ms)':<12} {'EER (%)':<10} {'AUC':<8} {'BDR':<8} {'Intra cos':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['offset_samples']:<18} {r['offset_ms']:<12} {r['eer']:<10} {r['auc']:<8} {r['bdr']:<8} {r['intra_cos_mean']:<10}")

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), 'results', 'sync_sensitivity_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
