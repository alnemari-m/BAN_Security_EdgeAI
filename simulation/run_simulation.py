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
    'window_len': 256,        # 1 second at 256 Hz
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


def run_nist_tests(keys, n_keys=1000):
    """Run subset of NIST SP 800-22 tests on generated keys."""
    results = {}
    test_funcs = {
        'Frequency (Monobit)': nist_frequency_test,
        'Runs': nist_runs_test,
        'Block Frequency': lambda b: nist_block_frequency_test(b, 8),
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
    # Conv1: 128*8 = 1024, Conv2: 64*16=1024, Conv3: 32*32=1024
    # Max activation: ~4 KB, plus IO buffers ~4 KB
    sram_kb = 8.2

    # MACs computation
    # Conv1: 128 * 8 * (5*1) = 5120
    # Conv2: 64 * 16 * (3*8) = 24576
    # Conv3: 32 * 32 * (3*16) = 49152
    # Dense1: 32 * 64 = 2048
    # Dense2: 64 * 32 = 2048
    total_macs = 5120 + 24576 + 49152 + 2048 + 2048

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
    print("PhysioKey Simulation Pipeline")
    print("=" * 60)

    # --- Step 1: Download data ---
    print("\n[1/6] Downloading PTB-XL data...")
    download_ptbxl(CONFIG['data_dir'], CONFIG['max_records'])

    # --- Step 2: Load & preprocess ---
    print("\n[2/6] Loading and preprocessing...")
    windows_s1, windows_s2, patient_ids = load_and_preprocess(
        CONFIG['data_dir'], CONFIG['max_records']
    )

    if len(windows_s1) < 100:
        print("ERROR: Not enough data. Trying alternative approach...")
        sys.exit(1)

    # Split patients into train/test
    unique_patients = list(set(patient_ids))
    np.random.shuffle(unique_patients)

    n_train = min(CONFIG['n_patients_train'], len(unique_patients) // 2)
    n_test = min(CONFIG['n_patients_test'], len(unique_patients) - n_train)

    train_patients = set(unique_patients[:n_train])
    test_patients = set(unique_patients[n_train:n_train + n_test])

    train_mask = np.array([p in train_patients for p in patient_ids])
    test_mask = np.array([p in test_patients for p in patient_ids])

    print(f"  Train: {train_mask.sum()} windows from {n_train} patients")
    print(f"  Test:  {test_mask.sum()} windows from {n_test} patients")

    # Create datasets
    train_dataset = PhysioKeyDataset(
        windows_s1[train_mask], windows_s2[train_mask],
        patient_ids[train_mask], mode='train'
    )
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG['batch_size'],
        shuffle=True, drop_last=True
    )

    # --- Step 3: Train model ---
    print(f"\n[3/6] Training 1D-CNN ({CONFIG['epochs']} epochs)...")
    model = PhysioKeyCNN(embed_dim=CONFIG['embed_dim']).to(DEVICE)
    print(f"  Parameters: {model.count_parameters()}")

    history = train_model(model, train_loader, CONFIG)

    # Save model for reproducibility
    model_path = os.path.join(CONFIG['results_dir'], 'physiokey_model.pt')
    torch.save(model.state_dict(), model_path)
    print(f"  Model saved to {model_path}")

    # --- Step 4: Extract embeddings on test set ---
    print("\n[4/6] Extracting test embeddings...")
    model.eval()

    test_s1 = torch.FloatTensor(windows_s1[test_mask]).unsqueeze(1).to(DEVICE)
    test_s2 = torch.FloatTensor(windows_s2[test_mask]).unsqueeze(1).to(DEVICE)
    test_pids = patient_ids[test_mask]

    with torch.no_grad():
        # Process in batches to avoid OOM
        z1_list, z2_list = [], []
        bs = 512
        for i in range(0, len(test_s1), bs):
            z1_list.append(model(test_s1[i:i+bs]).cpu().numpy())
            z2_list.append(model(test_s2[i:i+bs]).cpu().numpy())

        z1_all = np.concatenate(z1_list, axis=0)
        z2_all = np.concatenate(z2_list, axis=0)

    print(f"  Embeddings shape: {z1_all.shape}")

    # Save embeddings for reproducibility
    embed_path = os.path.join(CONFIG['results_dir'], 'embeddings.npz')
    np.savez(embed_path, z1=z1_all, z2=z2_all, patient_ids=test_pids)
    print(f"  Embeddings saved to {embed_path}")

    # --- Step 5: Compute metrics ---
    print("\n[5/6] Computing metrics...")

    # 5a. Cosine similarity: intra-body vs inter-body
    print("  Computing cosine similarities...")

    # Intra-body: same patient, Lead I vs Lead II
    intra_cos = cosine_similarity(z1_all, z2_all)

    # Inter-body: different patients
    n_inter = min(50000, len(z1_all) * 10)
    inter_cos = []
    unique_test_pids = list(set(test_pids))

    for _ in range(n_inter):
        # Pick two different patients
        p1, p2 = np.random.choice(len(unique_test_pids), 2, replace=False)
        pid1, pid2 = unique_test_pids[p1], unique_test_pids[p2]

        idx1 = np.where(test_pids == pid1)[0]
        idx2 = np.where(test_pids == pid2)[0]

        i1 = np.random.choice(idx1)
        i2 = np.random.choice(idx2)

        cos = cosine_similarity(
            z1_all[i1:i1+1], z2_all[i2:i2+1]
        )[0]
        inter_cos.append(cos)

    inter_cos = np.array(inter_cos)

    print(f"  Intra-body cosine: mean={intra_cos.mean():.4f}, std={intra_cos.std():.4f}")
    print(f"  Inter-body cosine: mean={inter_cos.mean():.4f}, std={inter_cos.std():.4f}")

    # 5b. FAR/FRR at various thresholds
    thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    far_frr_results = compute_far_frr(intra_cos, inter_cos, thresholds)

    print("\n  Threshold | FAR          | FRR          | BER")
    print("  " + "-" * 55)
    for r in far_frr_results:
        print(f"  {r['threshold']:.2f}      | {r['FAR']:.6e} | {r['FRR']:.6e} | {r['BER']:.6e}")

    # 5c. EER
    eer, eer_thresh, roc_auc = compute_eer(intra_cos, inter_cos)
    print(f"\n  EER: {eer:.6f} ({eer*100:.3f}%) at threshold {eer_thresh:.4f}")
    print(f"  ROC AUC: {roc_auc:.6f}")

    # 5d. Quantization and entropy
    print("\n  Computing quantization and entropy...")
    all_z = np.concatenate([z1_all, z2_all], axis=0)
    bits_all, boundaries = quantize_embedding(all_z, n_bits_per_dim=CONFIG['n_bits_per_dim'])

    bits_s1, _ = quantize_embedding(z1_all, n_bits_per_dim=CONFIG['n_bits_per_dim'], boundaries=boundaries)
    bits_s2, _ = quantize_embedding(z2_all, n_bits_per_dim=CONFIG['n_bits_per_dim'], boundaries=boundaries)

    # Per-dimension min-entropy
    dim_entropies = min_entropy_per_dim(bits_all, n_bits_per_dim=CONFIG['n_bits_per_dim'])
    total_min_entropy = dim_entropies.sum()

    print(f"  Per-dim min-entropy: mean={dim_entropies.mean():.4f}, "
          f"min={dim_entropies.min():.4f}, max={dim_entropies.max():.4f}")
    print(f"  Total min-entropy: {total_min_entropy:.1f} bits")

    # 5e. Bit Disagreement Rate for intra-body
    intra_hd = hamming_distance(bits_s1, bits_s2)
    bdr = intra_hd / CONFIG['n_bits']
    print(f"  Intra-body BDR: mean={bdr.mean():.4f}, std={bdr.std():.4f}")
    print(f"  Intra-body Hamming dist: mean={intra_hd.mean():.1f}, "
          f"median={np.median(intra_hd):.0f}")

    # Key agreement success rate (BCH can correct up to t errors)
    ka_success = np.mean(intra_hd <= CONFIG['bch_t'])
    print(f"  Key agreement success rate (BCH t={CONFIG['bch_t']}): {ka_success:.4f}")

    # Multi-round key agreement: use majority vote across R rounds
    # Each round uses a different time window; majority vote on each bit
    print("\n  Computing multi-round key agreement (R=3, 5)...")
    for n_rounds in [3, 5]:
        multi_success = 0
        multi_total = 0
        for pid in unique_test_pids:
            idx = np.where(test_pids == pid)[0]
            if len(idx) >= n_rounds:
                # Collect multiple rounds of quantized bits
                rounds_s1 = bits_s1[idx[:n_rounds]]  # (R, n_bits)
                rounds_s2 = bits_s2[idx[:n_rounds]]
                # Majority vote
                vote_s1 = (rounds_s1.sum(axis=0) > n_rounds / 2).astype(np.uint8)
                vote_s2 = (rounds_s2.sum(axis=0) > n_rounds / 2).astype(np.uint8)
                hd = np.sum(vote_s1 != vote_s2)
                if hd <= CONFIG['bch_t']:
                    multi_success += 1
                multi_total += 1
        if multi_total > 0:
            mr_rate = multi_success / multi_total
            print(f"    R={n_rounds}: success={mr_rate:.4f} ({multi_success}/{multi_total})")

    # 5f. Temporal drift (replay resistance)
    # Compare embeddings from different windows of same patient
    print("\n  Computing temporal drift for replay resistance...")
    drift_distances = []
    for pid in unique_test_pids[:30]:
        idx = np.where(test_pids == pid)[0]
        if len(idx) >= 4:
            # Compare window 0 vs windows 2, 3, ... (non-adjacent)
            for j in range(2, min(len(idx), 6)):
                hd = hamming_distance(bits_s1[idx[0]:idx[0]+1], bits_s1[idx[j]:idx[j]+1])[0]
                drift_distances.append(hd)

    if drift_distances:
        drift_arr = np.array(drift_distances)
        print(f"  Temporal drift HD: mean={drift_arr.mean():.1f}, "
              f"min={drift_arr.min():.0f}, max={drift_arr.max():.0f}")
        replay_fail_rate = np.mean(drift_arr > CONFIG['bch_t'])
        print(f"  Replay failure rate: {replay_fail_rate:.4f}")

    # 5g. NIST randomness tests
    # NIST SP 800-22 requires minimum 100 bits per sequence.
    # Individual keys are only 64 bits, so we concatenate multiple keys
    # from different patients to form longer sequences for meaningful testing.
    print("\n  Running NIST randomness tests...")
    print("    (Concatenating keys to reach NIST minimum sequence length)")

    # Approach 1: per-key tests (note: below NIST minimum, reported for transparency)
    keys_short = []
    for i in range(min(1000, len(bits_s1))):
        keys_short.append(bits_s1[i].copy())

    nist_results_short = run_nist_tests(keys_short)
    print("    Per-key results (64-bit sequences — below NIST minimum of 100 bits):")
    for test_name, result in nist_results_short.items():
        print(f"      {test_name}: pass_rate={result['pass_rate']:.3f}, "
              f"mean_p={result['mean_p_value']:.4f}")

    # Approach 2: concatenated sequences (>= 1000 bits each)
    concat_len = 1024  # bits per test sequence
    keys_per_seq = concat_len // CONFIG['n_bits']
    n_sequences = min(100, len(bits_s1) // keys_per_seq)
    keys_long = []
    for i in range(n_sequences):
        start = i * keys_per_seq
        seq = np.concatenate([bits_s1[start + j] for j in range(keys_per_seq)])
        keys_long.append(seq)

    nist_results_long = run_nist_tests(keys_long) if keys_long else {}
    print(f"    Concatenated results ({concat_len}-bit sequences, {n_sequences} sequences):")
    for test_name, result in nist_results_long.items():
        print(f"      {test_name}: pass_rate={result['pass_rate']:.3f}, "
              f"mean_p={result['mean_p_value']:.4f}")

    nist_results = {
        'per_key_64bit': nist_results_short,
        'concatenated_1024bit': nist_results_long,
        'note': 'Per-key tests use 64-bit sequences (below NIST SP 800-22 minimum of 100 bits). '
                'Concatenated tests use 1024-bit sequences formed from 16 consecutive keys.',
    }

    # 5h. Key agreement sweep with REAL BCH parameters
    print("\n  Key agreement parameter sweep (real BCH codes)...")
    sweep_results = []
    for bpd in [1, 2]:
        n_key_bits = 32 * bpd
        bits_sweep, bounds_sweep = quantize_embedding(all_z, n_bits_per_dim=bpd)
        bits_s1_sweep, _ = quantize_embedding(z1_all, n_bits_per_dim=bpd, boundaries=bounds_sweep)
        bits_s2_sweep, _ = quantize_embedding(z2_all, n_bits_per_dim=bpd, boundaries=bounds_sweep)
        hd_sweep = hamming_distance(bits_s1_sweep, bits_s2_sweep)
        bdr_sweep = hd_sweep / n_key_bits

        # Find the real BCH block length for this bit count
        bch_n = get_bch_block_length(n_key_bits)
        bch_codes = BCH_CODES.get(bch_n, [])

        print(f"\n    {bpd}b/dim, {n_key_bits} raw bits -> BCH({bch_n},...)")
        print(f"    BDR: mean={bdr_sweep.mean():.3f}, std={bdr_sweep.std():.3f}")

        for code in bch_codes:
            t_val = code['t']
            k_val = code['k']
            d_val = code['d']
            # Pad to BCH block length and check HD
            success = float(np.mean(hd_sweep <= t_val))
            sweep_results.append({
                'bits_per_dim': bpd,
                'total_bits': n_key_bits,
                'bch_n': bch_n,
                'bch_k': k_val,
                'bch_t': t_val,
                'bch_d': d_val,
                'effective_key_bits': k_val,
                'bdr_mean': float(bdr_sweep.mean()),
                'bdr_std': float(bdr_sweep.std()),
                'ka_success_rate': success,
            })
            print(f"      BCH({bch_n},{k_val},{d_val}), t={t_val}: "
                  f"success={success:.3f}, eff_key={k_val} bits")

    # 5i. Multi-round majority voting analysis
    print("\n  Multi-round majority voting analysis...")
    multiround_results = []
    for bpd in [1, 2]:
        n_key_bits = 32 * bpd
        bits_sweep, bounds_sweep = quantize_embedding(all_z, n_bits_per_dim=bpd)
        bdr_single = None

        for n_rounds in [1, 3, 5, 7]:
            # Compute actual multi-round BDR from data
            mr_hd_list = []
            mr_total = 0
            for pid in unique_test_pids:
                idx = np.where(test_pids == pid)[0]
                if len(idx) >= n_rounds:
                    for start in range(0, len(idx) - n_rounds + 1, n_rounds):
                        sel = idx[start:start + n_rounds]
                        rounds_s1, _ = quantize_embedding(
                            z1_all[sel], n_bits_per_dim=bpd, boundaries=bounds_sweep)
                        rounds_s2, _ = quantize_embedding(
                            z2_all[sel], n_bits_per_dim=bpd, boundaries=bounds_sweep)
                        if n_rounds == 1:
                            voted_s1 = rounds_s1[0]
                            voted_s2 = rounds_s2[0]
                        else:
                            voted_s1 = (rounds_s1.sum(axis=0) > n_rounds / 2).astype(np.uint8)
                            voted_s2 = (rounds_s2.sum(axis=0) > n_rounds / 2).astype(np.uint8)
                        hd = np.sum(voted_s1 != voted_s2)
                        mr_hd_list.append(hd)
                        mr_total += 1

            if not mr_hd_list:
                continue

            mr_hd_arr = np.array(mr_hd_list)
            mr_bdr = mr_hd_arr / n_key_bits

            if n_rounds == 1:
                bdr_single = float(mr_bdr.mean())

            # Theoretical BDR for comparison
            if bdr_single is not None and n_rounds > 1:
                theoretical_bdr = majority_vote_bdr(bdr_single, n_rounds)
            else:
                theoretical_bdr = float(mr_bdr.mean())

            bch_n = get_bch_block_length(n_key_bits)
            bch_codes = BCH_CODES.get(bch_n, [])

            print(f"\n    {bpd}b/dim, R={n_rounds}: "
                  f"actual_BDR={mr_bdr.mean():.3f}, "
                  f"theoretical_BDR={theoretical_bdr:.3f}, "
                  f"n_pairs={mr_total}")

            for code in bch_codes:
                t_val = code['t']
                k_val = code['k']
                success = float(np.mean(mr_hd_arr <= t_val))
                entry = {
                    'bits_per_dim': bpd,
                    'total_bits': n_key_bits,
                    'n_rounds': n_rounds,
                    'bch_n': bch_n,
                    'bch_k': k_val,
                    'bch_t': t_val,
                    'effective_key_bits': k_val,
                    'actual_bdr': float(mr_bdr.mean()),
                    'theoretical_bdr': theoretical_bdr,
                    'ka_success_rate': success,
                    'n_pairs': mr_total,
                }
                multiround_results.append(entry)
                if success > 0.01:  # Only print non-trivial results
                    print(f"      BCH({bch_n},{k_val},...) t={t_val}: "
                          f"success={success:.3f}, eff_key={k_val}b")

    # --- Step 6: Resource estimates ---
    print("\n[6/6] Computing resource estimates...")
    hw_results = estimate_cortex_m4_performance(model)
    for k, v in hw_results.items():
        print(f"  {k}: {v}")

    # ============================================================
    # SAVE ALL RESULTS
    # ============================================================
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    results = {
        'cosine_similarity': {
            'intra_body_mean': float(intra_cos.mean()),
            'intra_body_std': float(intra_cos.std()),
            'inter_body_mean': float(inter_cos.mean()),
            'inter_body_std': float(inter_cos.std()),
        },
        'far_frr': far_frr_results,
        'eer': {
            'value': float(eer),
            'threshold': float(eer_thresh),
            'roc_auc': float(roc_auc),
        },
        'entropy': {
            'per_dim_mean': float(dim_entropies.mean()),
            'per_dim_min': float(dim_entropies.min()),
            'per_dim_max': float(dim_entropies.max()),
            'total_min_entropy_bits': float(total_min_entropy),
            'per_dim_values': dim_entropies.tolist(),
        },
        'bit_disagreement': {
            'intra_body_bdr_mean': float(bdr.mean()),
            'intra_body_bdr_std': float(bdr.std()),
            'intra_body_hd_mean': float(intra_hd.mean()),
            'key_agreement_success_rate': float(ka_success),
        },
        'replay_resistance': {
            'temporal_drift_hd_mean': float(drift_arr.mean()) if drift_distances else 0,
            'replay_failure_rate': float(replay_fail_rate) if drift_distances else 0,
        },
        'nist_tests': nist_results,
        'hardware': hw_results,
        'dataset': {
            'n_train_patients': n_train,
            'n_test_patients': n_test,
            'n_train_windows': int(train_mask.sum()),
            'n_test_windows': int(test_mask.sum()),
            'total_patients': len(unique_patients),
        },
        'training': {
            'final_loss': float(history[-1]),
            'epochs': CONFIG['epochs'],
        },
        'key_agreement_sweep': sweep_results,
        'multiround_voting': multiround_results,
    }

    results_path = os.path.join(CONFIG['results_dir'], 'simulation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    # Print summary for paper
    print("\n" + "=" * 60)
    print("SUMMARY FOR PAPER")
    print("=" * 60)
    print(f"Intra-body cosine similarity: {intra_cos.mean():.3f} (std={intra_cos.std():.3f})")
    print(f"Inter-body cosine similarity: {inter_cos.mean():.3f} (std={inter_cos.std():.3f})")
    print(f"EER: {eer*100:.2f}%")
    print(f"ROC AUC: {roc_auc:.4f}")

    # Find best operating point (lowest BER)
    best = min(far_frr_results, key=lambda x: x['BER'])
    print(f"Best threshold: {best['threshold']:.2f}")
    print(f"  FAR = {best['FAR']:.2e}")
    print(f"  FRR = {best['FRR']:.2e}")

    print(f"Total min-entropy: {total_min_entropy:.1f} bits")
    print(f"Model parameters: {hw_results['n_params']}")
    print(f"Model flash: {hw_results['total_flash_kb']:.1f} KB")
    print(f"Inference latency: {hw_results['inference_ms']:.1f} ms")
    print(f"Energy per agreement: {hw_results['energy_both_sensors_mj']:.2f} mJ")

    # Print best multi-round results
    if multiround_results:
        print("\nBest multi-round key agreement results:")
        best_by_config = {}
        for r in multiround_results:
            key = (r['bits_per_dim'], r['n_rounds'])
            if key not in best_by_config or r['ka_success_rate'] > best_by_config[key]['ka_success_rate']:
                best_by_config[key] = r
        for (bpd, nr), r in sorted(best_by_config.items()):
            if r['ka_success_rate'] > 0.1:
                print(f"  {bpd}b/dim, R={nr}: {r['ka_success_rate']*100:.1f}% "
                      f"(BCH t={r['bch_t']}, eff_key={r['effective_key_bits']}b, "
                      f"BDR={r['actual_bdr']:.3f})")


if __name__ == '__main__':
    main()
