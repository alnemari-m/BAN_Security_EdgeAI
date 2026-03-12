"""
PhysioKey Simulation Pipeline
==============================
Downloads PTB-XL ECG data from PhysioNet, trains the 1D-CNN feature extractor,
runs key agreement simulations, and computes all metrics for the paper.

Uses Lead I (limb) and Lead II (chest) from the same patient as two
"different sensors" on the same body — they share cardiovascular origin
but have different waveform morphologies, closely modeling ECG-PPG pairing.

Author: Mohammed Alnemari
"""

import os
import sys
import json
import hashlib
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, filtfilt
from sklearn.metrics import roc_curve, auc
import wfdb

# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    'data_dir': os.path.join(os.path.dirname(__file__), 'ptbxl_data'),
    'results_dir': os.path.join(os.path.dirname(__file__), 'results'),
    'sample_rate': 256,       # Target Hz (PTB-XL native: 500Hz, we downsample)
    'window_len': 128,        # 0.5 second at 256 Hz (ablation showed best EER)
    'embed_dim': 32,          # Embedding dimension
    'n_bits': 64,             # Key length (2 bits per dimension * 32 dims)
    'n_bits_per_dim': 2,      # 2-bit quantization for lower BDR
    'bch_t': 10,              # BCH error correction capability
    'batch_size': 128,        # Smaller batches for better gradients
    'epochs': 150,            # Training epochs
    'lr': 5e-4,               # Lower LR for stability
    'temperature': 0.05,      # Lower temp for sharper contrastive loss
    'decorr_lambda': 0.05,    # Decorrelation
    'align_lambda': 0.5,      # Alignment loss weight
    'n_patients_train': 200,
    'n_patients_test': 100,
    'max_records': 500,       # Max records to download (reduced for speed)
    'seed': 42,
    # BIDMC dataset
    'bidmc_data_dir': os.path.join(os.path.dirname(__file__), 'bidmc_data'),
    'ppg_lowcut': 0.5,        # PPG bandpass low cutoff (Hz)
    'ppg_highcut': 8.0,       # PPG bandpass high cutoff (Hz)
    'n_cv_folds': 5,          # Number of cross-validation folds
}

os.makedirs(CONFIG['results_dir'], exist_ok=True)
np.random.seed(CONFIG['seed'])
torch.manual_seed(CONFIG['seed'])

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")


# ============================================================
# 1. DATA DOWNLOAD & PREPROCESSING
# ============================================================
def download_ptbxl(data_dir, max_records=500):
    """Download PTB-XL records from PhysioNet in bulk."""
    os.makedirs(data_dir, exist_ok=True)

    # Check if already downloaded
    marker = os.path.join(data_dir, '.download_complete')
    if os.path.exists(marker):
        print("PTB-XL data already downloaded.")
        return

    print("Downloading PTB-XL records from PhysioNet (bulk)...")

    # Get record list
    try:
        record_list = wfdb.get_record_list('ptb-xl', records='all')
        print(f"  Total available records: {len(record_list)}")
    except Exception as e:
        print(f"Could not fetch record list: {e}")
        # Build record list manually for high-res records
        record_list = []
        for i in range(1, max_records + 1):
            folder = f"{(i-1)//1000:02d}000"
            record_list.append(f"records500/{folder}/{i:05d}_hr")

    # Filter for high-resolution (500Hz) records only
    # Ensure proper filtering — records must start with 'records500/'
    hr_records = [r for r in record_list if r.startswith('records500/') and '_hr' in r]
    if not hr_records:
        hr_records = [r for r in record_list if '_hr' in r and 'records500' in r]
    if not hr_records:
        hr_records = record_list  # Fallback

    # Limit to max_records
    hr_records = hr_records[:max_records]
    print(f"  Downloading {len(hr_records)} high-res records...")

    # Download in one bulk call
    try:
        wfdb.dl_database('ptb-xl', data_dir, records=hr_records)
        downloaded = len(hr_records)
    except Exception as e:
        print(f"  Bulk download failed: {e}")
        print("  Falling back to batch download...")
        # Fall back to batch download in groups of 50
        downloaded = 0
        batch_size = 50
        for i in range(0, len(hr_records), batch_size):
            batch = hr_records[i:i+batch_size]
            try:
                wfdb.dl_database('ptb-xl', data_dir, records=batch)
                downloaded += len(batch)
                print(f"  Downloaded {downloaded}/{len(hr_records)} records...")
            except Exception as e2:
                print(f"  Batch {i//batch_size} failed: {e2}")
                # Try individual records in this batch
                for rec in batch:
                    try:
                        wfdb.dl_database('ptb-xl', data_dir, records=[rec])
                        downloaded += 1
                    except Exception:
                        pass

    print(f"Downloaded {downloaded} records total")

    with open(marker, 'w') as f:
        f.write(f"downloaded={downloaded}\n")


def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    """Apply Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 0.001)
    high = min(highcut / nyq, 0.999)
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def load_and_preprocess(data_dir, max_records=2000):
    """Load PTB-XL records, extract Lead I and Lead II, preprocess."""
    print("Loading and preprocessing PTB-XL data...")

    # Find all records
    records = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.endswith('_hr.hea'):
                rec_path = os.path.join(root, f.replace('.hea', ''))
                records.append(rec_path)

    if not records:
        # Try direct listing
        for i in range(1, max_records + 1):
            folder = f"{(i-1)//1000:02d}000"
            rec_path = os.path.join(data_dir, folder, f"{i:05d}_hr")
            if os.path.exists(rec_path + '.hea'):
                records.append(rec_path)

    records = records[:max_records]
    print(f"  Found {len(records)} records")

    all_windows_lead1 = []  # "Sensor 1" (Lead I)
    all_windows_lead2 = []  # "Sensor 2" (Lead II)
    patient_ids = []

    target_fs = CONFIG['sample_rate']
    window_len = CONFIG['window_len']

    for rec_path in records:
        try:
            record = wfdb.rdrecord(rec_path)
            sig = record.p_signal  # shape: (n_samples, n_leads)
            fs = record.fs
            sig_names = [s.upper().strip() for s in record.sig_name]

            # Find Lead I and Lead II indices
            lead1_idx = None
            lead2_idx = None
            for idx, name in enumerate(sig_names):
                if name in ('I', 'LEAD I', 'ECG_I'):
                    lead1_idx = idx
                elif name in ('II', 'LEAD II', 'ECG_II'):
                    lead2_idx = idx

            if lead1_idx is None or lead2_idx is None:
                # Fallback: use first two leads
                if sig.shape[1] >= 2:
                    lead1_idx = 0
                    lead2_idx = 1
                else:
                    continue

            lead1 = sig[:, lead1_idx].astype(np.float64)
            lead2 = sig[:, lead2_idx].astype(np.float64)

            # Remove NaNs
            if np.any(np.isnan(lead1)) or np.any(np.isnan(lead2)):
                continue

            # Resample from fs to target_fs
            if fs != target_fs:
                n_target = int(len(lead1) * target_fs / fs)
                lead1 = np.interp(
                    np.linspace(0, len(lead1)-1, n_target),
                    np.arange(len(lead1)), lead1
                )
                lead2 = np.interp(
                    np.linspace(0, len(lead2)-1, n_target),
                    np.arange(len(lead2)), lead2
                )

            # Bandpass filter (0.5 - 40 Hz for ECG)
            try:
                lead1 = bandpass_filter(lead1, 0.5, 40.0, target_fs)
                lead2 = bandpass_filter(lead2, 0.5, 40.0, target_fs)
            except Exception:
                continue

            # Normalize to zero mean, unit variance
            if np.std(lead1) < 1e-6 or np.std(lead2) < 1e-6:
                continue
            lead1 = (lead1 - np.mean(lead1)) / np.std(lead1)
            lead2 = (lead2 - np.mean(lead2)) / np.std(lead2)

            # Segment into windows
            n_windows = len(lead1) // window_len
            for w in range(n_windows):
                start = w * window_len
                end = start + window_len
                w1 = lead1[start:end]
                w2 = lead2[start:end]

                # Skip low-quality windows
                if np.std(w1) < 0.1 or np.std(w2) < 0.1:
                    continue

                all_windows_lead1.append(w1)
                all_windows_lead2.append(w2)
                patient_ids.append(rec_path)

        except Exception as e:
            continue

    all_windows_lead1 = np.array(all_windows_lead1, dtype=np.float32)
    all_windows_lead2 = np.array(all_windows_lead2, dtype=np.float32)
    patient_ids = np.array(patient_ids)

    print(f"  Extracted {len(all_windows_lead1)} window pairs from "
          f"{len(set(patient_ids))} patients")

    return all_windows_lead1, all_windows_lead2, patient_ids


def download_bidmc(data_dir):
    """Download BIDMC PPG and Respiration Dataset from PhysioNet.

    53 recordings x 8 minutes from ICU patients.
    Synchronized ECG (Lead II) + PPG (plethysmograph) at 125 Hz.
    Open access: physionet.org/content/bidmc/1.0.0/
    """
    os.makedirs(data_dir, exist_ok=True)

    marker = os.path.join(data_dir, '.download_complete')
    if os.path.exists(marker):
        print("BIDMC data already downloaded.")
        return

    print("Downloading BIDMC dataset from PhysioNet...")
    try:
        wfdb.dl_database('bidmc', data_dir)
        print("  BIDMC download complete.")
    except Exception as e:
        print(f"  Bulk download failed: {e}")
        print("  Trying individual records...")
        downloaded = 0
        for i in range(1, 54):
            rec_name = f"bidmc{i:02d}"
            try:
                wfdb.dl_database('bidmc', data_dir, records=[rec_name])
                downloaded += 1
            except Exception:
                pass
        print(f"  Downloaded {downloaded}/53 records.")

    with open(marker, 'w') as f:
        f.write("downloaded=bidmc\n")


def load_and_preprocess_bidmc(data_dir):
    """Load BIDMC records, extract ECG (II) and PPG (PLETH), preprocess.

    Returns (windows_ecg, windows_ppg, patient_ids) in the same format
    as the PTB-XL loader for pipeline compatibility.
    """
    print("Loading and preprocessing BIDMC data...")

    target_fs = CONFIG['sample_rate']  # 256 Hz
    window_len = CONFIG['window_len']  # 256 samples

    all_windows_ecg = []
    all_windows_ppg = []
    patient_ids = []

    for i in range(1, 54):
        rec_name = f"bidmc{i:02d}"
        rec_path = os.path.join(data_dir, rec_name)

        if not os.path.exists(rec_path + '.hea'):
            continue

        try:
            record = wfdb.rdrecord(rec_path)
            sig = record.p_signal
            fs = record.fs
            # Clean signal names: strip whitespace, commas, quotes
            sig_names = [s.strip().strip(',').strip().upper() for s in record.sig_name]

            # Find ECG (II) and PPG (PLETH) channel indices
            ecg_idx = None
            ppg_idx = None
            for idx, name in enumerate(sig_names):
                if name in ('II', 'LEAD II', 'ECG'):
                    ecg_idx = idx
                elif name in ('PLETH', 'PPG'):
                    ppg_idx = idx

            if ecg_idx is None or ppg_idx is None:
                continue

            ecg = sig[:, ecg_idx].astype(np.float64)
            ppg = sig[:, ppg_idx].astype(np.float64)

            if np.any(np.isnan(ecg)) or np.any(np.isnan(ppg)):
                continue

            # Resample from 125 Hz to 256 Hz
            if fs != target_fs:
                n_target = int(len(ecg) * target_fs / fs)
                ecg = np.interp(
                    np.linspace(0, len(ecg) - 1, n_target),
                    np.arange(len(ecg)), ecg
                )
                ppg = np.interp(
                    np.linspace(0, len(ppg) - 1, n_target),
                    np.arange(len(ppg)), ppg
                )

            # Bandpass filter: ECG 0.5-40 Hz, PPG 0.5-8 Hz
            try:
                ecg = bandpass_filter(ecg, 0.5, 40.0, target_fs)
                ppg = bandpass_filter(ppg, CONFIG['ppg_lowcut'],
                                      CONFIG['ppg_highcut'], target_fs)
            except Exception:
                continue

            # Normalize
            if np.std(ecg) < 1e-6 or np.std(ppg) < 1e-6:
                continue
            ecg = (ecg - np.mean(ecg)) / np.std(ecg)
            ppg = (ppg - np.mean(ppg)) / np.std(ppg)

            # Segment into windows (cap at 50 per patient to keep runtime manageable)
            n_windows = min(len(ecg) // window_len, 50)
            pid = rec_name  # e.g., "bidmc01"
            for w in range(n_windows):
                start = w * window_len
                end = start + window_len
                w_ecg = ecg[start:end]
                w_ppg = ppg[start:end]

                if np.std(w_ecg) < 0.1 or np.std(w_ppg) < 0.1:
                    continue

                all_windows_ecg.append(w_ecg)
                all_windows_ppg.append(w_ppg)
                patient_ids.append(pid)

        except Exception:
            continue

    all_windows_ecg = np.array(all_windows_ecg, dtype=np.float32)
    all_windows_ppg = np.array(all_windows_ppg, dtype=np.float32)
    patient_ids = np.array(patient_ids)

    print(f"  Extracted {len(all_windows_ecg)} ECG-PPG window pairs from "
          f"{len(set(patient_ids))} patients")

    return all_windows_ecg, all_windows_ppg, patient_ids


# ============================================================
# 2. DATASET & MODEL
# ============================================================
class PhysioKeyDataset(Dataset):
    """Dataset for contrastive training of PhysioKey embeddings."""

    def __init__(self, windows_s1, windows_s2, patient_ids, mode='train'):
        self.windows_s1 = torch.FloatTensor(windows_s1).unsqueeze(1)  # (N, 1, 256)
        self.windows_s2 = torch.FloatTensor(windows_s2).unsqueeze(1)
        self.patient_ids = patient_ids

        # Create patient-to-indices mapping
        self.patient_to_idx = {}
        for i, pid in enumerate(patient_ids):
            if pid not in self.patient_to_idx:
                self.patient_to_idx[pid] = []
            self.patient_to_idx[pid].append(i)

        self.unique_patients = list(self.patient_to_idx.keys())

    def __len__(self):
        return len(self.windows_s1)

    def __getitem__(self, idx):
        return self.windows_s1[idx], self.windows_s2[idx], idx


class PhysioKeyCNN(nn.Module):
    """1D-CNN feature extractor for PhysioKey.

    Architecture:
    Input(Nx1) -> Conv1D(8,k=5,s=2) -> BN -> Conv1D(16,k=3,s=2) -> BN
    -> Conv1D(32,k=3,s=2) -> BN -> GAP -> Dense(64) -> Dense(32)
    Uses BatchNorm for training stability and GAP for variable-length input.
    """

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
        x = x.mean(dim=2)           # GAP -> (batch, 32)
        x = F.relu(self.fc1(x))     # -> (batch, 64)
        x = self.fc2(x)             # -> (batch, 32)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# 3. TRAINING
# ============================================================
def contrastive_loss(z1, z2, temperature=0.07):
    """NT-Xent contrastive loss between paired embeddings."""
    batch_size = z1.shape[0]

    # Normalize
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Concatenate
    z = torch.cat([z1, z2], dim=0)  # (2B, d)

    # Similarity matrix
    sim = torch.mm(z, z.t()) / temperature  # (2B, 2B)

    # Mask out self-similarity
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim.masked_fill_(mask, -1e9)

    # Positive pairs: (i, i+B) and (i+B, i)
    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size),
        torch.arange(0, batch_size)
    ]).to(z.device)

    loss = F.cross_entropy(sim, labels)
    return loss


def decorrelation_loss(z):
    """Encourage decorrelated embedding dimensions."""
    z_centered = z - z.mean(dim=0)
    cov = (z_centered.t() @ z_centered) / (z.shape[0] - 1)

    # Normalize to correlation matrix
    std = torch.sqrt(torch.diag(cov) + 1e-8)
    corr = cov / (std.unsqueeze(0) * std.unsqueeze(1))

    # Loss: deviation from identity
    identity = torch.eye(corr.shape[0], device=corr.device)
    loss = ((corr - identity) ** 2).mean()
    return loss


class DualEncoderCNN(nn.Module):
    """Dual-encoder for cross-modal (ECG+PPG) key agreement.

    Two modality-specific 1D-CNN encoders with a shared projection head
    that maps both modalities to a common embedding space.
    Total: ~8448 params (~8.3 KB INT8, ~33.3 KB with TFLM runtime)
    """

    def __init__(self, embed_dim=32):
        super().__init__()
        # ECG encoder
        self.ecg_conv1 = nn.Conv1d(1, 8, kernel_size=5, stride=2, padding=2)
        self.ecg_bn1 = nn.BatchNorm1d(8)
        self.ecg_conv2 = nn.Conv1d(8, 16, kernel_size=3, stride=2, padding=1)
        self.ecg_bn2 = nn.BatchNorm1d(16)
        self.ecg_conv3 = nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1)
        self.ecg_bn3 = nn.BatchNorm1d(32)

        # PPG encoder
        self.ppg_conv1 = nn.Conv1d(1, 8, kernel_size=5, stride=2, padding=2)
        self.ppg_bn1 = nn.BatchNorm1d(8)
        self.ppg_conv2 = nn.Conv1d(8, 16, kernel_size=3, stride=2, padding=1)
        self.ppg_bn2 = nn.BatchNorm1d(16)
        self.ppg_conv3 = nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1)
        self.ppg_bn3 = nn.BatchNorm1d(32)

        # Shared projection head
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, embed_dim)

    def encode_ecg(self, x):
        x = F.relu(self.ecg_bn1(self.ecg_conv1(x)))
        x = F.relu(self.ecg_bn2(self.ecg_conv2(x)))
        x = F.relu(self.ecg_bn3(self.ecg_conv3(x)))
        return x.mean(dim=2)

    def encode_ppg(self, x):
        x = F.relu(self.ppg_bn1(self.ppg_conv1(x)))
        x = F.relu(self.ppg_bn2(self.ppg_conv2(x)))
        x = F.relu(self.ppg_bn3(self.ppg_conv3(x)))
        return x.mean(dim=2)

    def project(self, h):
        return self.fc2(F.relu(self.fc1(h)))

    def forward(self, x_ecg, x_ppg):
        z_ecg = self.project(self.encode_ecg(x_ecg))
        z_ppg = self.project(self.encode_ppg(x_ppg))
        return z_ecg, z_ppg

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


def train_dual_encoder(model, train_loader, config):
    """Train the DualEncoderCNN with the same loss as PhysioKeyCNN."""
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs']
    )
    model.train()
    history = []

    for epoch in range(config['epochs']):
        total_loss = 0
        n_batches = 0

        for s1, s2, _ in train_loader:
            s1, s2 = s1.to(DEVICE), s2.to(DEVICE)
            z1, z2 = model(s1, s2)

            loss_c = contrastive_loss(z1, z2, config['temperature'])
            loss_d = decorrelation_loss(torch.cat([z1, z2], dim=0))
            z1_n = F.normalize(z1, dim=1)
            z2_n = F.normalize(z2, dim=1)
            loss_align = 1.0 - (z1_n * z2_n).sum(dim=1).mean()
            loss = (loss_c
                    + config['decorr_lambda'] * loss_d
                    + config.get('align_lambda', 0.5) * loss_align)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        history.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{config['epochs']}: loss={avg_loss:.4f}")

    return history


def evaluate_dual_encoder_fold(model, windows_ecg, windows_ppg, patient_ids, config):
    """Evaluate a trained dual-encoder on a test fold."""
    model.eval()
    test_ecg = torch.FloatTensor(windows_ecg).unsqueeze(1).to(DEVICE)
    test_ppg = torch.FloatTensor(windows_ppg).unsqueeze(1).to(DEVICE)

    with torch.no_grad():
        z1_list, z2_list = [], []
        bs = 512
        for i in range(0, len(test_ecg), bs):
            z_ecg, z_ppg = model(test_ecg[i:i+bs], test_ppg[i:i+bs])
            z1_list.append(z_ecg.cpu().numpy())
            z2_list.append(z_ppg.cpu().numpy())
        z1_all = np.concatenate(z1_list, axis=0)
        z2_all = np.concatenate(z2_list, axis=0)

    intra_cos = cosine_similarity(z1_all, z2_all)
    unique_pids = list(set(patient_ids))
    n_inter = min(10000, len(z1_all) * 10)
    inter_cos = []
    for _ in range(n_inter):
        p1, p2 = np.random.choice(len(unique_pids), 2, replace=False)
        pid1, pid2 = unique_pids[p1], unique_pids[p2]
        idx1 = np.where(patient_ids == pid1)[0]
        idx2 = np.where(patient_ids == pid2)[0]
        i1, i2 = np.random.choice(idx1), np.random.choice(idx2)
        cos = cosine_similarity(z1_all[i1:i1+1], z2_all[i2:i2+1])[0]
        inter_cos.append(cos)
    inter_cos = np.array(inter_cos)

    eer, eer_thresh, roc_auc = compute_eer(intra_cos, inter_cos)

    embed_dim = z1_all.shape[1]
    all_z = np.concatenate([z1_all, z2_all], axis=0)
    bdr_1bit = bdr_2bit = 0.5
    for bpd in [1, 2]:
        bits_all, bounds = quantize_embedding(all_z, n_bits_per_dim=bpd)
        bits_s1, _ = quantize_embedding(z1_all, n_bits_per_dim=bpd, boundaries=bounds)
        bits_s2, _ = quantize_embedding(z2_all, n_bits_per_dim=bpd, boundaries=bounds)
        hd = hamming_distance(bits_s1, bits_s2)
        n_key_bits = embed_dim * bpd
        bdr = hd / n_key_bits
        if bpd == 1:
            bdr_1bit = float(bdr.mean())
        else:
            bdr_2bit = float(bdr.mean())

    # Key agreement sweep
    ka_results = {}
    for bpd in [1, 2]:
        n_key_bits = embed_dim * bpd
        bits_all, bounds = quantize_embedding(all_z, n_bits_per_dim=bpd)
        bits_s1, _ = quantize_embedding(z1_all, n_bits_per_dim=bpd, boundaries=bounds)
        bits_s2, _ = quantize_embedding(z2_all, n_bits_per_dim=bpd, boundaries=bounds)
        hd = hamming_distance(bits_s1, bits_s2)
        bch_n = get_bch_block_length(n_key_bits)
        for code in BCH_CODES.get(bch_n, []):
            key = f"{bpd}b_BCH({bch_n},{code['k']},{code['d']})"
            ka_results[key] = float(np.mean(hd <= code['t']))

    # NIST on debiased keys
    bits_s1_2b, _ = quantize_embedding(z1_all, n_bits_per_dim=2,
                                        boundaries=quantize_embedding(all_z, n_bits_per_dim=2)[1])
    keys_raw = [bits_s1_2b[i].copy() for i in range(min(500, len(bits_s1_2b)))]
    nist_perkey = run_nist_tests(keys_raw) if keys_raw else {}
    keys_debiased = [von_neumann_debias(k) for k in keys_raw]
    keys_debiased = [k for k in keys_debiased if len(k) >= 16]
    nist_debiased = run_nist_tests(keys_debiased) if keys_debiased else {}

    nist_concat_debiased = {}
    if len(keys_debiased) >= 16:
        concat_seqs = []
        buf = np.array([], dtype=np.uint8)
        for k in keys_debiased:
            buf = np.concatenate([buf, k])
            while len(buf) >= 1024:
                concat_seqs.append(buf[:1024].copy())
                buf = buf[1024:]
        if concat_seqs:
            nist_concat_debiased = run_nist_tests(concat_seqs, n_keys=len(concat_seqs))

    return {
        'intra_cos_mean': float(intra_cos.mean()),
        'inter_cos_mean': float(inter_cos.mean()),
        'eer': float(eer),
        'roc_auc': float(roc_auc),
        'bdr_1bit': bdr_1bit,
        'bdr_2bit': bdr_2bit,
        'key_agreement': ka_results,
        'nist_perkey': nist_perkey,
        'nist_debiased': nist_debiased,
        'nist_concat_debiased': nist_concat_debiased,
        'n_test_windows': len(z1_all),
        'n_test_patients': len(unique_pids),
        'n_debiased_keys': len(keys_debiased),
        'mean_debiased_len': float(np.mean([len(k) for k in keys_debiased])) if keys_debiased else 0,
    }


def run_dual_encoder_kfold(windows_ecg, windows_ppg, patient_ids, config, n_folds=5):
    """Run patient-level k-fold CV with dual-encoder on BIDMC."""
    unique_patients = np.array(list(set(patient_ids)))
    np.random.shuffle(unique_patients)
    fold_size = len(unique_patients) // n_folds
    fold_results = []

    for fold in range(n_folds):
        print(f"\n  --- BIDMC Dual-Encoder Fold {fold+1}/{n_folds} ---")
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else len(unique_patients)
        test_pats = set(unique_patients[test_start:test_end])
        train_pats = set(unique_patients) - test_pats

        train_mask = np.array([p in train_pats for p in patient_ids])
        test_mask = np.array([p in test_pats for p in patient_ids])

        print(f"    Train: {train_mask.sum()} windows from {len(train_pats)} patients")
        print(f"    Test:  {test_mask.sum()} windows from {len(test_pats)} patients")

        if train_mask.sum() < 50 or test_mask.sum() < 20:
            print(f"    Skipping fold {fold+1}: insufficient data")
            continue

        train_dataset = PhysioKeyDataset(
            windows_ecg[train_mask], windows_ppg[train_mask],
            patient_ids[train_mask], mode='train'
        )
        train_loader = DataLoader(
            train_dataset, batch_size=min(config['batch_size'], train_mask.sum()),
            shuffle=True, drop_last=True
        )

        model = DualEncoderCNN(embed_dim=config['embed_dim']).to(DEVICE)
        train_dual_encoder(model, train_loader, config)

        fold_metric = evaluate_dual_encoder_fold(
            model, windows_ecg[test_mask], windows_ppg[test_mask],
            patient_ids[test_mask], config
        )
        fold_metric['fold'] = fold + 1
        fold_results.append(fold_metric)
        print(f"    EER={fold_metric['eer']*100:.1f}%, "
              f"AUC={fold_metric['roc_auc']:.3f}, "
              f"BDR(1b)={fold_metric['bdr_1bit']:.3f}")

    if not fold_results:
        return {'error': 'No folds completed'}

    scalar_keys = ['intra_cos_mean', 'inter_cos_mean', 'eer', 'roc_auc',
                    'bdr_1bit', 'bdr_2bit']
    agg = {}
    for key in scalar_keys:
        vals = [f[key] for f in fold_results]
        agg[f'{key}_mean'] = float(np.mean(vals))
        agg[f'{key}_std'] = float(np.std(vals))

    ka_keys = set()
    for f in fold_results:
        ka_keys.update(f['key_agreement'].keys())
    ka_agg = {}
    for ka_key in ka_keys:
        vals = [f['key_agreement'].get(ka_key, 0) for f in fold_results]
        ka_agg[ka_key] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}

    return {
        'n_folds': len(fold_results),
        'per_fold': fold_results,
        'aggregated': agg,
        'key_agreement_aggregated': ka_agg,
        'dataset': 'BIDMC-DualEncoder',
    }


def train_model(model, train_loader, config):
    """Train the PhysioKey 1D-CNN."""
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs']
    )

    model.train()
    history = []

    for epoch in range(config['epochs']):
        total_loss = 0
        n_batches = 0

        for s1, s2, _ in train_loader:
            s1, s2 = s1.to(DEVICE), s2.to(DEVICE)

            z1 = model(s1)
            z2 = model(s2)

            # Combined loss
            loss_c = contrastive_loss(z1, z2, config['temperature'])
            loss_d = decorrelation_loss(torch.cat([z1, z2], dim=0))

            # Alignment loss: directly maximize cosine similarity for pairs
            z1_n = F.normalize(z1, dim=1)
            z2_n = F.normalize(z2, dim=1)
            loss_align = 1.0 - (z1_n * z2_n).sum(dim=1).mean()

            loss = loss_c + config['decorr_lambda'] * loss_d + config.get('align_lambda', 0.5) * loss_align

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        history.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{config['epochs']}: loss={avg_loss:.4f}")

    return history


# ============================================================
# 3b. CROSS-VALIDATION & ABLATION
# ============================================================
def evaluate_fold(model, windows_s1, windows_s2, patient_ids, config):
    """Evaluate a trained model on a test fold. Returns dict of metrics."""
    model.eval()
    test_s1 = torch.FloatTensor(windows_s1).unsqueeze(1).to(DEVICE)
    test_s2 = torch.FloatTensor(windows_s2).unsqueeze(1).to(DEVICE)

    with torch.no_grad():
        z1_list, z2_list = [], []
        bs = 512
        for i in range(0, len(test_s1), bs):
            z1_list.append(model(test_s1[i:i+bs]).cpu().numpy())
            z2_list.append(model(test_s2[i:i+bs]).cpu().numpy())
        z1_all = np.concatenate(z1_list, axis=0)
        z2_all = np.concatenate(z2_list, axis=0)

    # Cosine similarity
    intra_cos = cosine_similarity(z1_all, z2_all)

    unique_pids = list(set(patient_ids))
    n_inter = min(10000, len(z1_all) * 10)
    inter_cos = []
    for _ in range(n_inter):
        p1, p2 = np.random.choice(len(unique_pids), 2, replace=False)
        pid1, pid2 = unique_pids[p1], unique_pids[p2]
        idx1 = np.where(patient_ids == pid1)[0]
        idx2 = np.where(patient_ids == pid2)[0]
        i1, i2 = np.random.choice(idx1), np.random.choice(idx2)
        cos = cosine_similarity(z1_all[i1:i1+1], z2_all[i2:i2+1])[0]
        inter_cos.append(cos)
    inter_cos = np.array(inter_cos)

    # EER and AUC
    eer, eer_thresh, roc_auc = compute_eer(intra_cos, inter_cos)

    # FAR/FRR at fixed thresholds
    far_frr_thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    far_frr = compute_far_frr(intra_cos, inter_cos, far_frr_thresholds)

    # Quantization and BDR
    embed_dim = z1_all.shape[1]  # Use actual embedding dim, not hardcoded 32
    all_z = np.concatenate([z1_all, z2_all], axis=0)
    for bpd in [1, 2]:
        bits_all, bounds = quantize_embedding(all_z, n_bits_per_dim=bpd)
        bits_s1, _ = quantize_embedding(z1_all, n_bits_per_dim=bpd, boundaries=bounds)
        bits_s2, _ = quantize_embedding(z2_all, n_bits_per_dim=bpd, boundaries=bounds)
        hd = hamming_distance(bits_s1, bits_s2)
        n_key_bits = embed_dim * bpd
        bdr = hd / n_key_bits
        if bpd == 1:
            bdr_1bit = float(bdr.mean())
        else:
            bdr_2bit = float(bdr.mean())

    # Key agreement sweep
    ka_results = {}
    for bpd in [1, 2]:
        n_key_bits = embed_dim * bpd
        bits_all, bounds = quantize_embedding(all_z, n_bits_per_dim=bpd)
        bits_s1, _ = quantize_embedding(z1_all, n_bits_per_dim=bpd, boundaries=bounds)
        bits_s2, _ = quantize_embedding(z2_all, n_bits_per_dim=bpd, boundaries=bounds)
        hd = hamming_distance(bits_s1, bits_s2)
        bch_n = get_bch_block_length(n_key_bits)
        for code in BCH_CODES.get(bch_n, []):
            key = f"{bpd}b_BCH({bch_n},{code['k']},{code['d']})"
            ka_results[key] = float(np.mean(hd <= code['t']))

    # NIST tests: per-key (individual 64-bit keys)
    bits_all_2b, bounds_2b = quantize_embedding(all_z, n_bits_per_dim=2)
    bits_s1_2b, _ = quantize_embedding(z1_all, n_bits_per_dim=2, boundaries=bounds_2b)
    keys = [bits_s1_2b[i].copy() for i in range(min(500, len(bits_s1_2b)))]
    nist_perkey = run_nist_tests(keys) if keys else {}

    # NIST tests: concatenated (16 keys → 1024-bit sequences)
    nist_concat = {}
    if len(keys) >= 16:
        concat_seqs = []
        for i in range(0, len(keys) - 15, 16):
            seq = np.concatenate(keys[i:i+16])
            concat_seqs.append(seq)
        if concat_seqs:
            nist_concat = run_nist_tests(concat_seqs, n_keys=len(concat_seqs))

    # Von Neumann debiased NIST tests
    keys_debiased = [von_neumann_debias(k) for k in keys]
    keys_debiased = [k for k in keys_debiased if len(k) >= 16]
    nist_debiased = run_nist_tests(keys_debiased) if keys_debiased else {}

    # Concatenated debiased keys
    nist_concat_debiased = {}
    if len(keys_debiased) >= 16:
        target_len = 1024
        concat_seqs_db = []
        buf = np.array([], dtype=np.uint8)
        for k in keys_debiased:
            buf = np.concatenate([buf, k])
            while len(buf) >= target_len:
                concat_seqs_db.append(buf[:target_len].copy())
                buf = buf[target_len:]
        if concat_seqs_db:
            nist_concat_debiased = run_nist_tests(concat_seqs_db, n_keys=len(concat_seqs_db))

    return {
        'intra_cos_mean': float(intra_cos.mean()),
        'intra_cos_std': float(intra_cos.std()),
        'inter_cos_mean': float(inter_cos.mean()),
        'inter_cos_std': float(inter_cos.std()),
        'eer': float(eer),
        'roc_auc': float(roc_auc),
        'bdr_1bit': bdr_1bit,
        'bdr_2bit': bdr_2bit,
        'far_frr': far_frr,
        'key_agreement': ka_results,
        'nist_perkey': nist_perkey,
        'nist_concat': nist_concat,
        'nist_debiased': nist_debiased,
        'nist_concat_debiased': nist_concat_debiased,
        'n_debiased_keys': len(keys_debiased),
        'mean_debiased_len': float(np.mean([len(k) for k in keys_debiased])) if keys_debiased else 0,
        'n_test_windows': len(z1_all),
        'n_test_patients': len(unique_pids),
    }


def run_kfold_evaluation(windows_s1, windows_s2, patient_ids, config,
                         n_folds=5, dataset_name=''):
    """Run patient-level k-fold cross-validation.

    Fresh model per fold, train from scratch.
    Returns per-fold results + mean +/- std aggregations.
    """
    unique_patients = np.array(list(set(patient_ids)))
    np.random.shuffle(unique_patients)

    fold_size = len(unique_patients) // n_folds
    fold_results = []

    for fold in range(n_folds):
        print(f"\n  --- {dataset_name} Fold {fold+1}/{n_folds} ---")

        # Patient-level split
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else len(unique_patients)
        test_pats = set(unique_patients[test_start:test_end])
        train_pats = set(unique_patients) - test_pats

        train_mask = np.array([p in train_pats for p in patient_ids])
        test_mask = np.array([p in test_pats for p in patient_ids])

        print(f"    Train: {train_mask.sum()} windows from {len(train_pats)} patients")
        print(f"    Test:  {test_mask.sum()} windows from {len(test_pats)} patients")

        if train_mask.sum() < 50 or test_mask.sum() < 20:
            print(f"    Skipping fold {fold+1}: insufficient data")
            continue

        # Create dataset and loader
        train_dataset = PhysioKeyDataset(
            windows_s1[train_mask], windows_s2[train_mask],
            patient_ids[train_mask], mode='train'
        )
        train_loader = DataLoader(
            train_dataset, batch_size=min(config['batch_size'], train_mask.sum()),
            shuffle=True, drop_last=True
        )

        # Fresh model
        model = PhysioKeyCNN(embed_dim=config['embed_dim']).to(DEVICE)
        train_model(model, train_loader, config)

        # Evaluate
        fold_metric = evaluate_fold(
            model, windows_s1[test_mask], windows_s2[test_mask],
            patient_ids[test_mask], config
        )
        fold_metric['fold'] = fold + 1
        fold_results.append(fold_metric)

        print(f"    EER={fold_metric['eer']*100:.1f}%, "
              f"AUC={fold_metric['roc_auc']:.3f}, "
              f"BDR(1b)={fold_metric['bdr_1bit']:.3f}")

    if not fold_results:
        return {'error': 'No folds completed'}

    # Aggregate: mean +/- std across folds
    scalar_keys = ['intra_cos_mean', 'inter_cos_mean', 'eer', 'roc_auc',
                    'bdr_1bit', 'bdr_2bit']
    agg = {}
    for key in scalar_keys:
        vals = [f[key] for f in fold_results]
        agg[f'{key}_mean'] = float(np.mean(vals))
        agg[f'{key}_std'] = float(np.std(vals))

    # Aggregate FAR/FRR across folds
    far_frr_agg = []
    if fold_results and 'far_frr' in fold_results[0]:
        n_thresholds = len(fold_results[0]['far_frr'])
        for ti in range(n_thresholds):
            tau = fold_results[0]['far_frr'][ti]['threshold']
            fars = [f['far_frr'][ti]['FAR'] for f in fold_results]
            frrs = [f['far_frr'][ti]['FRR'] for f in fold_results]
            bers = [f['far_frr'][ti]['BER'] for f in fold_results]
            far_frr_agg.append({
                'threshold': tau,
                'FAR_mean': float(np.mean(fars)),
                'FAR_std': float(np.std(fars)),
                'FRR_mean': float(np.mean(frrs)),
                'FRR_std': float(np.std(frrs)),
                'BER_mean': float(np.mean(bers)),
                'BER_std': float(np.std(bers)),
            })

    # Aggregate key agreement results
    ka_keys = set()
    for f in fold_results:
        ka_keys.update(f['key_agreement'].keys())
    ka_agg = {}
    for ka_key in ka_keys:
        vals = [f['key_agreement'].get(ka_key, 0) for f in fold_results]
        ka_agg[ka_key] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
        }

    return {
        'n_folds': len(fold_results),
        'per_fold': fold_results,
        'aggregated': agg,
        'far_frr_aggregated': far_frr_agg,
        'key_agreement_aggregated': ka_agg,
        'dataset': dataset_name,
    }


def run_ablation_studies(windows_s1, windows_s2, patient_ids, config):
    """Run ablation studies on PTB-XL dataset.

    Uses 3-fold CV with 80 epochs per variant for speed.
    Studies: loss components, embedding dim, window length.
    """
    print("\n" + "=" * 60)
    print("ABLATION STUDIES (PTB-XL)")
    print("=" * 60)

    ablation_config = dict(config)
    ablation_config['epochs'] = 80
    ablation_folds = 3

    results = {}

    # --- Loss Ablation ---
    print("\n  [Ablation 1/3] Loss components...")
    loss_variants = {
        'contrastive_only': {'decorr_lambda': 0.0, 'align_lambda': 0.0},
        'contrastive_align': {'decorr_lambda': 0.0, 'align_lambda': 0.5},
        'contrastive_decorr': {'decorr_lambda': 0.05, 'align_lambda': 0.0},
        'full': {'decorr_lambda': 0.05, 'align_lambda': 0.5},
    }
    loss_results = {}
    for name, overrides in loss_variants.items():
        print(f"\n    Variant: {name}")
        var_config = dict(ablation_config)
        var_config.update(overrides)
        res = run_kfold_evaluation(
            windows_s1, windows_s2, patient_ids, var_config,
            n_folds=ablation_folds, dataset_name=f'loss_{name}'
        )
        loss_results[name] = res.get('aggregated', {})
        if 'aggregated' in res:
            print(f"      EER={res['aggregated'].get('eer_mean', 0)*100:.1f}% "
                  f"+/- {res['aggregated'].get('eer_std', 0)*100:.1f}%")
    results['loss_ablation'] = loss_results

    # --- Embedding Dimension Ablation ---
    print("\n  [Ablation 2/3] Embedding dimensions...")
    dim_results = {}
    for dim in [16, 32, 64]:
        print(f"\n    Embed dim: {dim}")
        var_config = dict(ablation_config)
        var_config['embed_dim'] = dim
        var_config['n_bits'] = dim * var_config['n_bits_per_dim']
        res = run_kfold_evaluation(
            windows_s1, windows_s2, patient_ids, var_config,
            n_folds=ablation_folds, dataset_name=f'dim_{dim}'
        )
        dim_results[str(dim)] = res.get('aggregated', {})
        if 'aggregated' in res:
            print(f"      EER={res['aggregated'].get('eer_mean', 0)*100:.1f}%")
    results['dim_ablation'] = dim_results

    # --- Window Length Ablation ---
    print("\n  [Ablation 3/3] Window lengths...")
    wl_results = {}
    for wl in [128, 256, 512]:
        print(f"\n    Window length: {wl}")
        # Re-window the data
        var_config = dict(ablation_config)
        var_config['window_len'] = wl

        if wl == config['window_len']:
            # Use original windows (same as default config)
            wl_s1, wl_s2, wl_pids = windows_s1, windows_s2, patient_ids
        else:
            # Re-segment: concatenate windows per patient, then re-window
            unique_pats = list(set(patient_ids))
            wl_s1_list, wl_s2_list, wl_pids_list = [], [], []
            for pat in unique_pats:
                mask = patient_ids == pat
                s1_cat = windows_s1[mask].reshape(-1)
                s2_cat = windows_s2[mask].reshape(-1)
                n_win = len(s1_cat) // wl
                for w in range(n_win):
                    w1 = s1_cat[w * wl:(w + 1) * wl]
                    w2 = s2_cat[w * wl:(w + 1) * wl]
                    if np.std(w1) >= 0.1 and np.std(w2) >= 0.1:
                        wl_s1_list.append(w1)
                        wl_s2_list.append(w2)
                        wl_pids_list.append(pat)
            if not wl_s1_list:
                continue
            wl_s1 = np.array(wl_s1_list, dtype=np.float32)
            wl_s2 = np.array(wl_s2_list, dtype=np.float32)
            wl_pids = np.array(wl_pids_list)

        res = run_kfold_evaluation(
            wl_s1, wl_s2, wl_pids, var_config,
            n_folds=ablation_folds, dataset_name=f'wl_{wl}'
        )
        wl_results[str(wl)] = res.get('aggregated', {})
        if 'aggregated' in res:
            print(f"      EER={res['aggregated'].get('eer_mean', 0)*100:.1f}%")
    results['window_ablation'] = wl_results

    return results


# ============================================================
# 4. KEY AGREEMENT SIMULATION
# ============================================================
def quantize_embedding(z, n_bits_per_dim=CONFIG['n_bits_per_dim'], boundaries=None):
    """Multi-level Gray code quantization.

    z: (N, d) numpy array of embeddings
    Returns: (N, d * n_bits_per_dim) binary array
    """
    d = z.shape[1]
    n_levels = 2 ** n_bits_per_dim  # 16 levels

    if boundaries is None:
        # Compute quantization boundaries from data
        boundaries = []
        for dim in range(d):
            vals = z[:, dim]
            percentiles = np.linspace(0, 100, n_levels + 1)
            bounds = np.percentile(vals, percentiles)
            boundaries.append(bounds)

    # Quantize each dimension
    bits = np.zeros((z.shape[0], d * n_bits_per_dim), dtype=np.uint8)

    for dim in range(d):
        bounds = boundaries[dim]
        # Digitize: which bin does each value fall into
        level = np.digitize(z[:, dim], bounds[1:-1])  # 0 to n_levels-1
        level = np.clip(level, 0, n_levels - 1)

        # Gray code encoding
        gray = level ^ (level >> 1)

        # Convert to bits
        for bit_idx in range(n_bits_per_dim):
            bits[:, dim * n_bits_per_dim + bit_idx] = (gray >> (n_bits_per_dim - 1 - bit_idx)) & 1

    return bits, boundaries


def hamming_distance(a, b):
    """Compute Hamming distance between binary arrays."""
    return np.sum(a != b, axis=-1)


def cosine_similarity(z1, z2):
    """Row-wise cosine similarity."""
    norm1 = np.linalg.norm(z1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(z2, axis=1, keepdims=True)
    return np.sum(z1 * z2, axis=1) / (norm1.squeeze() * norm2.squeeze() + 1e-8)


def compute_far_frr(intra_scores, inter_scores, thresholds):
    """Compute FAR and FRR at given thresholds."""
    results = []
    for tau in thresholds:
        # FAR: fraction of inter-body pairs accepted (score >= tau)
        far = np.mean(inter_scores >= tau)
        # FRR: fraction of intra-body pairs rejected (score < tau)
        frr = np.mean(intra_scores < tau)
        ber = (far + frr) / 2
        results.append({'threshold': tau, 'FAR': far, 'FRR': frr, 'BER': ber})
    return results


def compute_eer(intra_scores, inter_scores):
    """Compute Equal Error Rate."""
    # Labels: 1 for intra-body (positive), 0 for inter-body (negative)
    labels = np.concatenate([np.ones(len(intra_scores)), np.zeros(len(inter_scores))])
    scores = np.concatenate([intra_scores, inter_scores])

    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr

    # Find where FPR = FNR
    idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    eer_threshold = thresholds[idx]

    roc_auc = auc(fpr, tpr)

    return eer, eer_threshold, roc_auc


def min_entropy_per_dim(bits, n_bits_per_dim=CONFIG['n_bits_per_dim']):
    """Compute per-dimension min-entropy."""
    d = bits.shape[1] // n_bits_per_dim
    entropies = []

    for dim in range(d):
        dim_bits = bits[:, dim*n_bits_per_dim:(dim+1)*n_bits_per_dim]
        # Convert to integer
        vals = np.zeros(dim_bits.shape[0], dtype=int)
        for b in range(n_bits_per_dim):
            vals = vals * 2 + dim_bits[:, b]

        # Compute max probability
        unique, counts = np.unique(vals, return_counts=True)
        probs = counts / counts.sum()
        max_prob = probs.max()

        h_inf = -np.log2(max_prob)
        entropies.append(h_inf)

    return np.array(entropies)


def nist_frequency_test(bits_sequence):
    """NIST Monobit Frequency Test."""
    n = len(bits_sequence)
    s = np.sum(2 * bits_sequence.astype(float) - 1)
    s_obs = abs(s) / np.sqrt(n)
    from scipy.special import erfc
    p_value = erfc(s_obs / np.sqrt(2))
    return p_value


def nist_runs_test(bits_sequence):
    """NIST Runs Test."""
    n = len(bits_sequence)
    pi = np.mean(bits_sequence)

    # Pre-test
    if abs(pi - 0.5) >= 2 / np.sqrt(n):
        return 0.0

    # Count runs
    runs = 1
    for i in range(1, n):
        if bits_sequence[i] != bits_sequence[i-1]:
            runs += 1

    # Compute p-value
    from scipy.special import erfc
    num = abs(runs - 2*n*pi*(1-pi))
    den = 2*np.sqrt(2*n)*pi*(1-pi)
    if den < 1e-10:
        return 0.0
    p_value = erfc(num / den)
    return p_value


def nist_block_frequency_test(bits_sequence, block_size=8):
    """NIST Block Frequency Test."""
    n = len(bits_sequence)
    n_blocks = n // block_size
    if n_blocks == 0:
        return 0.0

    chi_sq = 0
    for i in range(n_blocks):
        block = bits_sequence[i*block_size:(i+1)*block_size]
        pi_i = np.mean(block)
        chi_sq += (pi_i - 0.5) ** 2

    chi_sq *= 4 * block_size

    from scipy.special import gammaincc
    p_value = gammaincc(n_blocks / 2, chi_sq / 2)
    return p_value


def nist_serial_test(bits_sequence, m=2):
    """NIST Serial Test — overlapping pattern uniformity."""
    n = len(bits_sequence)
    if n < m + 1:
        return 0.0

    def count_patterns(seq, block_len):
        counts = {}
        for i in range(len(seq)):
            pattern = tuple(seq[i:i+block_len] if i + block_len <= len(seq)
                          else np.concatenate([seq[i:], seq[:block_len - (len(seq) - i)]]))
            counts[pattern] = counts.get(pattern, 0) + 1
        return counts

    def psi_sq(seq, block_len):
        counts = count_patterns(seq, block_len)
        total = sum(v**2 for v in counts.values())
        return (2**block_len / len(seq)) * total - len(seq)

    psi_m = psi_sq(bits_sequence, m)
    psi_m1 = psi_sq(bits_sequence, m - 1) if m >= 2 else 0
    psi_m2 = psi_sq(bits_sequence, m - 2) if m >= 3 else 0

    delta1 = psi_m - psi_m1
    delta2 = psi_m - 2 * psi_m1 + psi_m2

    from scipy.special import gammaincc
    p1 = gammaincc(2**(m-2), delta1 / 2) if delta1 > 0 else 1.0
    p2 = gammaincc(2**(m-3), delta2 / 2) if m >= 3 and delta2 > 0 else 1.0

    return min(p1, p2)


def nist_approximate_entropy_test(bits_sequence, m=2):
    """NIST Approximate Entropy Test — entropy estimation."""
    n = len(bits_sequence)
    if n < m + 1:
        return 0.0

    def phi(seq, block_len):
        counts = {}
        extended = np.concatenate([seq, seq[:block_len - 1]])
        for i in range(n):
            pattern = tuple(extended[i:i+block_len])
            counts[pattern] = counts.get(pattern, 0) + 1
        c_vals = np.array(list(counts.values())) / n
        return np.sum(c_vals * np.log(c_vals + 1e-15))

    phi_m = phi(bits_sequence, m)
    phi_m1 = phi(bits_sequence, m + 1)

    ap_en = phi_m - phi_m1
    chi_sq = 2 * n * (np.log(2) - ap_en)

    from scipy.special import gammaincc
    p_value = gammaincc(2**(m-1), chi_sq / 2)
    return p_value


def nist_cumulative_sums_test(bits_sequence):
    """NIST Cumulative Sums Test — random walk max deviation."""
    n = len(bits_sequence)
    if n < 10:
        return 0.0

    # Convert to +1/-1
    walk = 2 * bits_sequence.astype(float) - 1
    cumsum = np.cumsum(walk)
    z = np.max(np.abs(cumsum))

    # Compute p-value
    k_max = int(np.floor((n / z + 1) / 4)) + 1
    from scipy.stats import norm
    p_sum = 0
    for k in range(-k_max, k_max + 1):
        p_sum += (norm.cdf((4 * k + 1) * z / np.sqrt(n)) -
                  norm.cdf((4 * k - 1) * z / np.sqrt(n)))
    p_value = 1 - p_sum
    return max(0.0, min(1.0, p_value))


def von_neumann_debias(bits):
    """Von Neumann debiasing: remove bias from a bit sequence.

    Process consecutive non-overlapping pairs:
      (0,1) -> output 0;  (1,0) -> output 1;  same -> discard
    Expected output length: ~N/4 for input of N bits.
    """
    output = []
    for i in range(0, len(bits) - 1, 2):
        if bits[i] != bits[i + 1]:
            output.append(bits[i])
    return np.array(output, dtype=np.uint8)


def run_nist_tests(keys, n_keys=1000):
    """Run subset of NIST SP 800-22 tests on generated keys."""
    results = {}
    test_funcs = {
        'Frequency (Monobit)': nist_frequency_test,
        'Runs': nist_runs_test,
        'Block Frequency': lambda b: nist_block_frequency_test(b, 8),
        'Serial': lambda b: nist_serial_test(b, 2),
        'Approximate Entropy': lambda b: nist_approximate_entropy_test(b, 2),
        'Cumulative Sums': nist_cumulative_sums_test,
    }

    for test_name, test_func in test_funcs.items():
        p_values = []
        for key in keys[:n_keys]:
            p = test_func(key)
            p_values.append(p)

        p_values = np.array(p_values)
        pass_rate = np.mean(p_values >= 0.01)
        mean_p = np.mean(p_values)

        results[test_name] = {
            'pass_rate': float(pass_rate),
            'mean_p_value': float(mean_p),
        }

    return results


# ============================================================
# BCH CODE PARAMETER TABLES (real codes over GF(2^m))
# ============================================================
# BCH(n, k, d) where n = 2^m - 1, t = floor((d-1)/2) errors corrected
BCH_CODES = {
    31: [  # GF(2^5), n=31
        {'t': 1, 'k': 26, 'd': 3},
        {'t': 2, 'k': 21, 'd': 5},
        {'t': 3, 'k': 16, 'd': 7},
        {'t': 5, 'k': 11, 'd': 11},
        {'t': 7, 'k': 6, 'd': 15},
    ],
    63: [  # GF(2^6), n=63
        {'t': 1, 'k': 57, 'd': 3},
        {'t': 2, 'k': 51, 'd': 5},
        {'t': 3, 'k': 45, 'd': 7},
        {'t': 4, 'k': 39, 'd': 9},
        {'t': 5, 'k': 36, 'd': 11},
        {'t': 6, 'k': 30, 'd': 13},
        {'t': 7, 'k': 24, 'd': 15},
        {'t': 10, 'k': 18, 'd': 21},
        {'t': 11, 'k': 16, 'd': 23},
        {'t': 13, 'k': 10, 'd': 27},
        {'t': 15, 'k': 7, 'd': 31},
    ],
    127: [  # GF(2^7), n=127
        {'t': 1, 'k': 120, 'd': 3},
        {'t': 2, 'k': 113, 'd': 5},
        {'t': 3, 'k': 106, 'd': 7},
        {'t': 5, 'k': 92, 'd': 11},
        {'t': 7, 'k': 78, 'd': 15},
        {'t': 10, 'k': 64, 'd': 21},
        {'t': 13, 'k': 50, 'd': 27},
        {'t': 15, 'k': 43, 'd': 31},
        {'t': 21, 'k': 29, 'd': 43},
        {'t': 23, 'k': 22, 'd': 47},
        {'t': 27, 'k': 15, 'd': 55},
        {'t': 31, 'k': 8, 'd': 63},
    ],
}


def get_bch_block_length(n_bits):
    """Find the smallest BCH block length >= n_bits."""
    for n in sorted(BCH_CODES.keys()):
        if n >= n_bits:
            return n
    return max(BCH_CODES.keys())


def majority_vote_bdr(per_bit_bdr, n_rounds):
    """Compute BDR after majority voting over n_rounds (must be odd).

    For each bit position, take majority across R rounds.
    New BDR = P(majority of R Bernoulli(p) trials are 1).
    """
    from scipy.special import comb
    p = per_bit_bdr
    # P(majority wrong) = sum_{k=ceil(R/2)}^{R} C(R,k) * p^k * (1-p)^(R-k)
    threshold = n_rounds // 2 + 1
    bdr_new = 0.0
    for k in range(threshold, n_rounds + 1):
        bdr_new += comb(n_rounds, k, exact=True) * (p ** k) * ((1 - p) ** (n_rounds - k))
    return bdr_new


def estimate_cortex_m4_performance(model):
    """Analytical resource estimates for ARM Cortex-M4."""
    n_params = model.count_parameters()

    # INT8: 1 byte per parameter
    model_size_kb = n_params / 1024

    # TFLM runtime overhead: ~25 KB
    total_flash_kb = model_size_kb + 25.0

    # SRAM: tensor arena (largest intermediate activation)
    # Depends on input window length
    wl = CONFIG['window_len']
    conv1_out = wl // 2   # stride 2
    conv2_out = conv1_out // 2
    conv3_out = conv2_out // 2
    max_activation = max(conv1_out * 8, conv2_out * 16, conv3_out * 32)
    sram_kb = round(max_activation / 1024 + 4.0, 1)  # activation + IO buffers

    # MACs computation (depends on window length)
    mac_conv1 = conv1_out * 8 * (5 * 1)
    mac_conv2 = conv2_out * 16 * (3 * 8)
    mac_conv3 = conv3_out * 32 * (3 * 16)
    mac_dense1 = 32 * 64
    mac_dense2 = 64 * CONFIG['embed_dim']
    total_macs = mac_conv1 + mac_conv2 + mac_conv3 + mac_dense1 + mac_dense2

    # Cortex-M4 @ 168 MHz with CMSIS-NN: ~40 MMAC/s
    mmacs_per_sec = 40.0
    inference_ms = (total_macs / 1e6) / mmacs_per_sec * 1000

    # BCH + quantization: ~1.2 ms
    bch_ms = 1.2
    # SHA-256: ~0.3 ms
    sha_ms = 0.3

    total_ms = inference_ms + bch_ms + sha_ms

    # Energy: Cortex-M4 @ 168 MHz: ~48.2 mW active
    power_mw = 168 * 0.287  # 0.287 mW/MHz
    energy_mj = power_mw * total_ms / 1000

    return {
        'n_params': int(n_params),
        'model_size_kb': round(model_size_kb, 1),
        'total_flash_kb': round(total_flash_kb, 1),
        'sram_kb': round(sram_kb, 1),
        'total_macs': int(total_macs),
        'inference_ms': round(inference_ms, 1),
        'bch_ms': round(bch_ms, 1),
        'sha_ms': round(sha_ms, 1),
        'total_latency_ms': round(total_ms, 1),
        'power_mw': round(power_mw, 1),
        'energy_per_agreement_mj': round(energy_mj, 3),
        'energy_both_sensors_mj': round(2 * energy_mj, 3),
    }


# ============================================================
# 5. MAIN PIPELINE
# ============================================================
def main():
    print("=" * 60)
    print("PhysioKey Simulation Pipeline (Revised)")
    print("  - Dual dataset: PTB-XL + BIDMC (ECG+PPG)")
    print("  - 5-fold cross-validation")
    print("  - Ablation studies")
    print("=" * 60)

    start_time = time.time()

    # --- Step 1: Download both datasets ---
    print("\n[1/7] Downloading datasets...")
    download_ptbxl(CONFIG['data_dir'], CONFIG['max_records'])
    download_bidmc(CONFIG['bidmc_data_dir'])

    # --- Step 2: Load & preprocess both datasets ---
    print("\n[2/7] Loading and preprocessing...")
    ptbxl_s1, ptbxl_s2, ptbxl_pids = load_and_preprocess(
        CONFIG['data_dir'], CONFIG['max_records']
    )
    bidmc_s1, bidmc_s2, bidmc_pids = load_and_preprocess_bidmc(
        CONFIG['bidmc_data_dir']
    )

    if len(ptbxl_s1) < 100:
        print("ERROR: Not enough PTB-XL data.")
        sys.exit(1)

    print(f"\n  PTB-XL: {len(ptbxl_s1)} windows, "
          f"{len(set(ptbxl_pids))} patients")
    print(f"  BIDMC:  {len(bidmc_s1)} windows, "
          f"{len(set(bidmc_pids))} patients")

    # --- Step 3: PTB-XL 5-fold cross-validation ---
    print("\n[3/7] PTB-XL 5-fold cross-validation...")
    ptbxl_cv = run_kfold_evaluation(
        ptbxl_s1, ptbxl_s2, ptbxl_pids, CONFIG,
        n_folds=CONFIG['n_cv_folds'], dataset_name='PTB-XL'
    )

    if 'aggregated' in ptbxl_cv:
        agg = ptbxl_cv['aggregated']
        print(f"\n  PTB-XL CV Summary:")
        print(f"    EER: {agg['eer_mean']*100:.1f}% +/- {agg['eer_std']*100:.1f}%")
        print(f"    AUC: {agg['roc_auc_mean']:.3f} +/- {agg['roc_auc_std']:.3f}")
        print(f"    BDR(1b): {agg['bdr_1bit_mean']:.3f} +/- {agg['bdr_1bit_std']:.3f}")

    # --- Step 4: BIDMC 5-fold cross-validation ---
    print("\n[4/7] BIDMC 5-fold cross-validation...")
    bidmc_cv = run_kfold_evaluation(
        bidmc_s1, bidmc_s2, bidmc_pids, CONFIG,
        n_folds=CONFIG['n_cv_folds'], dataset_name='BIDMC'
    )

    if 'aggregated' in bidmc_cv:
        agg = bidmc_cv['aggregated']
        print(f"\n  BIDMC CV Summary:")
        print(f"    EER: {agg['eer_mean']*100:.1f}% +/- {agg['eer_std']*100:.1f}%")
        print(f"    AUC: {agg['roc_auc_mean']:.3f} +/- {agg['roc_auc_std']:.3f}")
        print(f"    BDR(1b): {agg['bdr_1bit_mean']:.3f} +/- {agg['bdr_1bit_std']:.3f}")

    # --- Step 4b: BIDMC Dual-Encoder CV ---
    print("\n[4b/8] Running BIDMC dual-encoder CV...")
    if len(bidmc_s1) >= 20:
        bidmc_dual_cv = run_dual_encoder_kfold(
            bidmc_s1, bidmc_s2, bidmc_pids, CONFIG, n_folds=5
        )
    else:
        print("  Skipping dual-encoder: insufficient BIDMC data")
        bidmc_dual_cv = {'error': 'Insufficient data'}

    if isinstance(bidmc_dual_cv, dict) and 'aggregated' in bidmc_dual_cv:
        agg = bidmc_dual_cv['aggregated']
        print(f"\n  BIDMC Dual-Encoder CV Summary:")
        print(f"    EER: {agg['eer_mean']*100:.1f}% +/- {agg['eer_std']*100:.1f}%")
        print(f"    AUC: {agg['roc_auc_mean']:.3f} +/- {agg['roc_auc_std']:.3f}")
        print(f"    BDR(1b): {agg['bdr_1bit_mean']:.3f} +/- {agg['bdr_1bit_std']:.3f}")

    # --- Step 5: Ablation studies (PTB-XL — where model works) ---
    print("\n[5/8] Running ablation studies on PTB-XL...")
    if len(ptbxl_s1) >= 100:
        ablation_results = run_ablation_studies(
            ptbxl_s1, ptbxl_s2, ptbxl_pids, CONFIG
        )
    else:
        print("  Skipping ablation: insufficient PTB-XL data")
        ablation_results = {'error': 'Insufficient data'}

    # --- Step 6: Hardware resource estimates ---
    print("\n[6/8] Computing resource estimates...")
    model = PhysioKeyCNN(embed_dim=CONFIG['embed_dim']).to(DEVICE)
    hw_results = estimate_cortex_m4_performance(model)

    # Hybrid PhysioKey+ECDH estimates
    ecdh_latency_ms = 142.0
    ecdh_energy_mj = 6.84
    hkdf_ms = 0.5
    hkdf_energy = 0.024
    hybrid_latency = hw_results['total_latency_ms'] + ecdh_latency_ms + hkdf_ms
    hybrid_energy = hw_results['energy_per_agreement_mj'] + ecdh_energy_mj + hkdf_energy

    hw_results['hybrid_latency_ms'] = round(hybrid_latency, 1)
    hw_results['hybrid_energy_mj'] = round(hybrid_energy, 3)

    for k, v in hw_results.items():
        print(f"  {k}: {v}")

    # --- Step 7: Save all results ---
    print("\n[7/8] Saving results...")

    elapsed = time.time() - start_time

    results = {
        'ptbxl_cv': ptbxl_cv,
        'bidmc_cv': bidmc_cv,
        'bidmc_dual_cv': bidmc_dual_cv,
        'ablation': ablation_results,
        'hardware': hw_results,
        'config': {k: v for k, v in CONFIG.items()
                   if not k.endswith('_dir')},
        'runtime_seconds': round(elapsed, 1),
    }

    results_path = os.path.join(CONFIG['results_dir'], 'simulation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print(f"Total runtime: {elapsed/60:.1f} minutes")

    # ============================================================
    # SUMMARY FOR PAPER
    # ============================================================
    print("\n" + "=" * 60)
    print("SUMMARY FOR PAPER")
    print("=" * 60)

    for name, cv_res in [('PTB-XL', ptbxl_cv), ('BIDMC', bidmc_cv),
                         ('BIDMC Dual-Encoder', bidmc_dual_cv)]:
        if not isinstance(cv_res, dict) or 'aggregated' not in cv_res:
            continue
        a = cv_res['aggregated']
        print(f"\n{name} ({cv_res['n_folds']}-fold CV):")
        print(f"  Intra-body cosine: {a['intra_cos_mean_mean']:.3f} "
              f"+/- {a['intra_cos_mean_std']:.3f}")
        print(f"  Inter-body cosine: {a['inter_cos_mean_mean']:.3f} "
              f"+/- {a['inter_cos_mean_std']:.3f}")
        print(f"  EER: {a['eer_mean']*100:.1f}% +/- {a['eer_std']*100:.1f}%")
        print(f"  ROC AUC: {a['roc_auc_mean']:.3f} +/- {a['roc_auc_std']:.3f}")
        print(f"  BDR (1-bit): {a['bdr_1bit_mean']:.3f} +/- {a['bdr_1bit_std']:.3f}")
        print(f"  BDR (2-bit): {a['bdr_2bit_mean']:.3f} +/- {a['bdr_2bit_std']:.3f}")

    print(f"\nHardware (Cortex-M4):")
    print(f"  Parameters: {hw_results['n_params']}")
    print(f"  Flash: {hw_results['total_flash_kb']:.1f} KB")
    print(f"  Inference: {hw_results['inference_ms']:.1f} ms")
    print(f"  PhysioKey standalone: {hw_results['total_latency_ms']:.1f} ms, "
          f"{hw_results['energy_per_agreement_mj']:.3f} mJ")
    print(f"  Hybrid PhysioKey+ECDH: {hw_results['hybrid_latency_ms']:.1f} ms, "
          f"{hw_results['hybrid_energy_mj']:.3f} mJ")

    # Print FAR/FRR table for paper
    if 'far_frr_aggregated' in ptbxl_cv and ptbxl_cv['far_frr_aggregated']:
        print(f"\nFAR/FRR (PTB-XL, {ptbxl_cv['n_folds']}-fold CV mean +/- std):")
        print(f"  {'Threshold':>10s} {'FAR':>16s} {'FRR':>16s} {'BER':>16s}")
        for entry in ptbxl_cv['far_frr_aggregated']:
            print(f"  {entry['threshold']:>10.2f} "
                  f"{entry['FAR_mean']:.3f}+/-{entry['FAR_std']:.3f} "
                  f"{entry['FRR_mean']:.3f}+/-{entry['FRR_std']:.3f} "
                  f"{entry['BER_mean']:.3f}+/-{entry['BER_std']:.3f}")

    if 'loss_ablation' in ablation_results:
        print(f"\nAblation (loss components):")
        for name, res in ablation_results['loss_ablation'].items():
            if res:
                print(f"  {name}: EER={res.get('eer_mean', 0)*100:.1f}%")


if __name__ == '__main__':
    main()
