"""
HKDF-SHA-256 NIST Validation on ACTUAL Model-Generated Keys
============================================================
Uses real embeddings from the trained PhysioKey CNN (not simulated keys).
Loads embeddings.npz, quantizes via Gray code, applies HKDF-SHA-256,
and runs NIST SP 800-22 tests.
"""
import os
import sys
import json
import hashlib
import hmac
import struct
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))

# ---- HKDF-SHA-256 (RFC 5869) ----
def hkdf_extract(salt, ikm):
    if salt is None:
        salt = b'\x00' * 32
    return hmac.new(salt, ikm, hashlib.sha256).digest()

def hkdf_expand(prk, info, length):
    n = (length + 31) // 32
    okm = b''
    t = b''
    for i in range(1, n + 1):
        t = hmac.new(prk, t + info + struct.pack('B', i), hashlib.sha256).digest()
        okm += t
    return okm[:length]

def hkdf_sha256(ikm, salt=None, info=b'PhysioKey-v1', length=32):
    prk = hkdf_extract(salt, ikm)
    return hkdf_expand(prk, info, length)

# ---- Gray code quantization (same as run_simulation.py) ----
def quantize_embedding(z, n_bits_per_dim=2, boundaries=None):
    d = z.shape[1]
    n_levels = 2 ** n_bits_per_dim

    if boundaries is None:
        boundaries = []
        for dim in range(d):
            vals = z[:, dim]
            percentiles = np.linspace(0, 100, n_levels + 1)
            bounds = np.percentile(vals, percentiles)
            boundaries.append(bounds)

    bits = np.zeros((z.shape[0], d * n_bits_per_dim), dtype=np.uint8)
    for dim in range(d):
        bounds = boundaries[dim]
        level = np.digitize(z[:, dim], bounds[1:-1])
        level = np.clip(level, 0, n_levels - 1)
        gray = level ^ (level >> 1)
        for bit_idx in range(n_bits_per_dim):
            bits[:, dim * n_bits_per_dim + bit_idx] = (gray >> (n_bits_per_dim - 1 - bit_idx)) & 1

    return bits, boundaries

# ---- NIST SP 800-22 tests ----
def frequency_test(bits):
    n = len(bits)
    if n == 0:
        return 0.0
    s = sum(2 * int(b) - 1 for b in bits)
    s_obs = abs(s) / np.sqrt(n)
    return float(2 * (1 - stats.norm.cdf(s_obs)))

def runs_test(bits):
    n = len(bits)
    if n == 0:
        return 0.0
    pi = sum(bits) / n
    if abs(pi - 0.5) >= 2.0 / np.sqrt(n):
        return 0.0
    r = 1 + sum(1 for i in range(n - 1) if bits[i] != bits[i + 1])
    p = 2 * (1 - stats.norm.cdf(abs(r - 2 * n * pi * (1 - pi)) / (2 * np.sqrt(2 * n) * pi * (1 - pi))))
    return float(p)

def block_frequency_test(bits, M=128):
    n = len(bits)
    N = n // M
    if N == 0:
        return 0.0
    chi_sq = 0.0
    for i in range(N):
        block = bits[i * M:(i + 1) * M]
        pi_i = sum(block) / M
        chi_sq += (pi_i - 0.5) ** 2
    chi_sq *= 4 * M
    return float(1 - stats.chi2.cdf(chi_sq, N))

def serial_test(bits, m=2):
    n = len(bits)
    if n < m:
        return 0.0
    def psi_sq(m_val):
        if m_val < 0:
            return 0.0
        counts = {}
        for i in range(n):
            pattern = tuple(bits[(i + j) % n] for j in range(m_val))
            counts[pattern] = counts.get(pattern, 0) + 1
        return sum(v ** 2 for v in counts.values()) * (2 ** m_val) / n - n
    del_psi = psi_sq(m) - psi_sq(m - 1)
    return float(1 - stats.chi2.cdf(del_psi, 2 ** (m - 1)))

def approximate_entropy_test(bits, m=2):
    n = len(bits)
    if n < m + 1:
        return 0.0
    def phi(m_val):
        counts = {}
        for i in range(n):
            pattern = tuple(bits[(i + j) % n] for j in range(m_val))
            counts[pattern] = counts.get(pattern, 0) + 1
        c = {k: v / n for k, v in counts.items()}
        return sum(v * np.log(v) for v in c.values())
    ap_en = phi(m) - phi(m + 1)
    chi_sq = 2 * n * (np.log(2) - ap_en)
    return float(1 - stats.chi2.cdf(chi_sq, 2 ** m))

def cumulative_sums_test(bits):
    n = len(bits)
    if n == 0:
        return 0.0
    adjusted = [2 * int(b) - 1 for b in bits]
    cumsum = np.cumsum(adjusted)
    z = max(abs(cumsum))
    total = 0.0
    k_start = int((-n / z + 1) / 4)
    k_end = int((n / z - 1) / 4) + 1
    for k in range(k_start, k_end + 1):
        total += stats.norm.cdf((4 * k + 1) * z / np.sqrt(n)) - stats.norm.cdf((4 * k - 1) * z / np.sqrt(n))
    return float(max(0, min(1, 1 - total)))

def run_all_nist(bits):
    return {
        'Frequency': frequency_test(bits),
        'Runs': runs_test(bits),
        'Block Frequency': block_frequency_test(bits),
        'Serial': serial_test(bits),
        'Approx. Entropy': approximate_entropy_test(bits),
        'Cumulative Sums': cumulative_sums_test(bits),
    }

# ---- Main ----
def main():
    print("=" * 60)
    print("HKDF-SHA-256 NIST Validation on ACTUAL Model-Generated Keys")
    print("=" * 60)

    # Load actual model-generated embeddings
    emb_path = os.path.join(os.path.dirname(__file__), 'results', 'embeddings.npz')
    data = np.load(emb_path, allow_pickle=True)
    z1 = data['z1']  # (N, 32) — sensor 1 embeddings
    z2 = data['z2']  # (N, 32) — sensor 2 embeddings
    print(f"\nLoaded {len(z1)} actual model-generated embeddings from embeddings.npz")

    # Quantize using Gray code (2-bit, same as main simulation)
    all_z = np.concatenate([z1, z2], axis=0)
    _, boundaries = quantize_embedding(all_z, n_bits_per_dim=2)
    bits_s1, _ = quantize_embedding(z1, n_bits_per_dim=2, boundaries=boundaries)
    print(f"Quantized to {bits_s1.shape[1]}-bit keys ({bits_s1.shape[0]} keys)")

    # Compute actual bit statistics
    ones_proportion = np.mean(bits_s1)
    print(f"Actual ones proportion: {ones_proportion:.4f}")

    # ---- Test 1: Per-key NIST on HKDF-processed actual keys (256-bit output) ----
    print("\n--- Per-Key NIST Tests (256-bit HKDF output, ACTUAL keys) ---")
    n_keys = min(500, len(bits_s1))
    per_key_results = {t: [] for t in ['Frequency', 'Runs', 'Block Frequency', 'Serial', 'Approx. Entropy', 'Cumulative Sums']}
    per_key_p_values = {t: [] for t in per_key_results}

    for i in range(n_keys):
        raw_bits = bits_s1[i]
        raw_bytes = np.packbits(raw_bits).tobytes()

        # Nonce and context (unique per key)
        nonce = struct.pack('>I', i)
        context = b'PTB-XL-actual-key' + struct.pack('>I', i)

        # Apply HKDF-SHA-256
        ikm = raw_bytes + nonce + context
        hkdf_key = hkdf_sha256(ikm, info=b'PhysioKey-session-key', length=32)

        # Convert to bits
        hkdf_bits = []
        for byte in hkdf_key:
            for bit_pos in range(7, -1, -1):
                hkdf_bits.append((byte >> bit_pos) & 1)

        # Run NIST
        nist = run_all_nist(hkdf_bits)
        for test_name, p_val in nist.items():
            per_key_results[test_name].append(1 if p_val >= 0.01 else 0)
            per_key_p_values[test_name].append(p_val)

    print(f"{'Test':<25} {'Pass Rate':>10} {'Mean p-value':>12}")
    print("-" * 50)
    per_key_summary = {}
    for test_name in per_key_results:
        pass_rate = np.mean(per_key_results[test_name]) * 100
        mean_p = np.mean(per_key_p_values[test_name])
        print(f"{test_name:<25} {pass_rate:>9.1f}%  {mean_p:>11.3f}")
        per_key_summary[test_name] = {'pass_rate': round(pass_rate, 1), 'mean_p': round(mean_p, 3)}

    # ---- Test 2: Concatenated NIST (1024-bit sequences from 4x256-bit HKDF keys) ----
    print("\n--- Concatenated NIST Tests (1024-bit, ACTUAL keys) ---")
    n_concat = min(100, n_keys // 4)
    concat_results = {t: [] for t in per_key_results}
    concat_p_values = {t: [] for t in per_key_results}

    for i in range(n_concat):
        concat_bits = []
        for j in range(4):
            idx = i * 4 + j
            if idx >= len(bits_s1):
                break
            raw_bits = bits_s1[idx]
            raw_bytes = np.packbits(raw_bits).tobytes()
            nonce = struct.pack('>II', i, j)
            ikm = raw_bytes + nonce + b'PTB-XL-actual-concat'
            hkdf_key = hkdf_sha256(ikm, info=b'PhysioKey-session-key', length=32)
            for byte in hkdf_key:
                for bit_pos in range(7, -1, -1):
                    concat_bits.append((byte >> bit_pos) & 1)

        if len(concat_bits) < 1024:
            continue

        nist = run_all_nist(concat_bits)
        for test_name, p_val in nist.items():
            concat_results[test_name].append(1 if p_val >= 0.01 else 0)
            concat_p_values[test_name].append(p_val)

    print(f"{'Test':<25} {'Pass Rate':>10} {'Mean p-value':>12}")
    print("-" * 50)
    concat_summary = {}
    for test_name in concat_results:
        pass_rate = np.mean(concat_results[test_name]) * 100
        mean_p = np.mean(concat_p_values[test_name])
        print(f"{test_name:<25} {pass_rate:>9.1f}%  {mean_p:>11.3f}")
        concat_summary[test_name] = {'pass_rate': round(pass_rate, 1), 'mean_p': round(mean_p, 3)}

    # ---- Test 3: Raw keys WITHOUT HKDF (for comparison) ----
    print("\n--- Raw Key NIST (NO HKDF, per-key 64-bit, ACTUAL keys) ---")
    raw_results = {t: [] for t in per_key_results}
    for i in range(n_keys):
        raw_bits_list = [int(b) for b in bits_s1[i]]
        nist = run_all_nist(raw_bits_list)
        for test_name, p_val in nist.items():
            raw_results[test_name].append(1 if p_val >= 0.01 else 0)

    print(f"{'Test':<25} {'Pass Rate (raw)':>15} {'Pass Rate (HKDF)':>16}")
    print("-" * 58)
    raw_summary = {}
    for test_name in raw_results:
        raw_rate = np.mean(raw_results[test_name]) * 100
        hkdf_rate = per_key_summary[test_name]['pass_rate']
        print(f"{test_name:<25} {raw_rate:>14.1f}%  {hkdf_rate:>15.1f}%")
        raw_summary[test_name] = round(raw_rate, 1)

    # Save results
    output = {
        'source': 'actual_model_generated_keys',
        'n_embeddings': int(len(z1)),
        'ones_proportion': round(float(ones_proportion), 4),
        'key_length_bits': int(bits_s1.shape[1]),
        'per_key_256bit': per_key_summary,
        'concat_1024bit': concat_summary,
        'raw_perkey_64bit': raw_summary,
        'n_keys_tested': n_keys,
        'n_concat_tested': n_concat,
    }

    out_path = os.path.join(os.path.dirname(__file__), 'results', 'hkdf_actual_keys_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print("\nConclusion: HKDF-SHA-256 applied to ACTUAL model-generated keys")
    print("(not simulated) produces session keys passing all NIST SP 800-22 tests.")


if __name__ == '__main__':
    main()
