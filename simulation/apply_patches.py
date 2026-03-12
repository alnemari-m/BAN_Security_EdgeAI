"""Apply patches to run_simulation.py: add DualEncoderCNN + von Neumann debiasing."""
import re

SIM_PATH = r'C:\Users\maste\Academic\Research\BAN_Security_EdgeAI\simulation\run_simulation.py'

with open(SIM_PATH, 'r', encoding='utf-8') as f:
    content = f.read()

# ============================================================
# PATCH 1: Add DualEncoderCNN class + training/eval functions
# ============================================================
DUAL_ENCODER_CODE = r'''

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

'''

# Insert DualEncoderCNN after PhysioKeyCNN's count_parameters
marker1 = "    def count_parameters(self):\n        return sum(p.numel() for p in self.parameters())\n\n\n# ============================================================\n# 3. TRAINING"
assert marker1 in content, "PATCH 1 marker not found!"
content = content.replace(
    marker1,
    "    def count_parameters(self):\n        return sum(p.numel() for p in self.parameters())\n"
    + DUAL_ENCODER_CODE
    + "\n# ============================================================\n# 3. TRAINING",
    1
)
print("PATCH 1 applied: DualEncoderCNN + training/eval functions")

# ============================================================
# PATCH 2: Add von Neumann debiasing before run_nist_tests
# ============================================================
VON_NEUMANN = '''def von_neumann_debias(bits):
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


'''

marker2 = "def run_nist_tests(keys, n_keys=1000):"
assert marker2 in content, "PATCH 2 marker not found!"
content = content.replace(marker2, VON_NEUMANN + marker2, 1)
print("PATCH 2 applied: von_neumann_debias()")

# ============================================================
# PATCH 3: Add debiased NIST to evaluate_fold return dict
# ============================================================
# Add debiased testing before the return statement in evaluate_fold
old_return = "    # NIST tests: concatenated (16 keys \u2192 1024-bit sequences)\n    nist_concat = {}\n    if len(keys) >= 16:\n        concat_seqs = []\n        for i in range(0, len(keys) - 15, 16):\n            seq = np.concatenate(keys[i:i+16])\n            concat_seqs.append(seq)\n        if concat_seqs:\n            nist_concat = run_nist_tests(concat_seqs, n_keys=len(concat_seqs))\n\n    return {\n        'intra_cos_mean': float(intra_cos.mean()),"

new_return = "    # NIST tests: concatenated (16 keys \u2192 1024-bit sequences)\n    nist_concat = {}\n    if len(keys) >= 16:\n        concat_seqs = []\n        for i in range(0, len(keys) - 15, 16):\n            seq = np.concatenate(keys[i:i+16])\n            concat_seqs.append(seq)\n        if concat_seqs:\n            nist_concat = run_nist_tests(concat_seqs, n_keys=len(concat_seqs))\n\n    # Von Neumann debiased NIST tests\n    keys_debiased = [von_neumann_debias(k) for k in keys]\n    keys_debiased = [k for k in keys_debiased if len(k) >= 16]\n    nist_debiased = run_nist_tests(keys_debiased) if keys_debiased else {}\n\n    # Concatenated debiased keys\n    nist_concat_debiased = {}\n    if len(keys_debiased) >= 16:\n        target_len = 1024\n        concat_seqs_db = []\n        buf = np.array([], dtype=np.uint8)\n        for k in keys_debiased:\n            buf = np.concatenate([buf, k])\n            while len(buf) >= target_len:\n                concat_seqs_db.append(buf[:target_len].copy())\n                buf = buf[target_len:]\n        if concat_seqs_db:\n            nist_concat_debiased = run_nist_tests(concat_seqs_db, n_keys=len(concat_seqs_db))\n\n    return {\n        'intra_cos_mean': float(intra_cos.mean()),"

assert old_return in content, "PATCH 3 marker not found!"
content = content.replace(old_return, new_return, 1)
print("PATCH 3 applied: debiased NIST in evaluate_fold")

# Add debiased results to return dict
old_ret_dict = """        'nist_perkey': nist_perkey,
        'nist_concat': nist_concat,
        'n_test_windows': len(z1_all),"""

new_ret_dict = """        'nist_perkey': nist_perkey,
        'nist_concat': nist_concat,
        'nist_debiased': nist_debiased,
        'nist_concat_debiased': nist_concat_debiased,
        'n_debiased_keys': len(keys_debiased),
        'mean_debiased_len': float(np.mean([len(k) for k in keys_debiased])) if keys_debiased else 0,
        'n_test_windows': len(z1_all),"""

assert old_ret_dict in content, "PATCH 3b marker not found!"
content = content.replace(old_ret_dict, new_ret_dict, 1)
print("PATCH 3b applied: debiased results in return dict")

# ============================================================
# PATCH 4: Add dual-encoder step to main()
# ============================================================
old_step5 = """    # --- Step 5: Ablation studies (PTB-XL -- where model works) ---
    print("\\n[5/7] Running ablation studies on PTB-XL...")"""

new_step5 = """    # --- Step 4b: BIDMC Dual-Encoder 5-fold cross-validation ---
    print("\\n[4b/8] BIDMC Dual-Encoder 5-fold cross-validation...")
    bidmc_dual_cv = run_dual_encoder_kfold(
        bidmc_s1, bidmc_s2, bidmc_pids, CONFIG,
        n_folds=CONFIG['n_cv_folds']
    )

    if 'aggregated' in bidmc_dual_cv:
        agg = bidmc_dual_cv['aggregated']
        print(f"\\n  BIDMC Dual-Encoder CV Summary:")
        print(f"    EER: {agg['eer_mean']*100:.1f}% +/- {agg['eer_std']*100:.1f}%")
        print(f"    AUC: {agg['roc_auc_mean']:.3f} +/- {agg['roc_auc_std']:.3f}")
        print(f"    BDR(1b): {agg['bdr_1bit_mean']:.3f} +/- {agg['bdr_1bit_std']:.3f}")

    # --- Step 5: Ablation studies (PTB-XL -- where model works) ---
    print("\\n[5/8] Running ablation studies on PTB-XL...")"""

assert old_step5 in content, "PATCH 4 marker not found!"
content = content.replace(old_step5, new_step5, 1)
print("PATCH 4 applied: dual-encoder step in main()")

# ============================================================
# PATCH 5: Add dual-encoder to results JSON
# ============================================================
old_results = """    results = {
        'ptbxl_cv': ptbxl_cv,
        'bidmc_cv': bidmc_cv,
        'ablation': ablation_results,"""

new_results = """    results = {
        'ptbxl_cv': ptbxl_cv,
        'bidmc_cv': bidmc_cv,
        'bidmc_dual_cv': bidmc_dual_cv,
        'ablation': ablation_results,"""

assert old_results in content, "PATCH 5 marker not found!"
content = content.replace(old_results, new_results, 1)
print("PATCH 5 applied: dual-encoder in results JSON")

# ============================================================
# PATCH 6: Add dual-encoder to summary output
# ============================================================
old_summary = "    for name, cv_res in [('PTB-XL', ptbxl_cv), ('BIDMC', bidmc_cv)]:"
new_summary = """    for name, cv_res in [('PTB-XL', ptbxl_cv), ('BIDMC (shared)', bidmc_cv),
                         ('BIDMC (dual-encoder)', bidmc_dual_cv)]:"""

assert old_summary in content, "PATCH 6 marker not found!"
content = content.replace(old_summary, new_summary, 1)
print("PATCH 6 applied: dual-encoder in summary")

# ============================================================
# Write
# ============================================================
with open(SIM_PATH, 'w', encoding='utf-8') as f:
    f.write(content)

print("\nAll patches applied successfully!")
print(f"File size: {len(content)} chars, {content.count(chr(10))} lines")
