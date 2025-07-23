import os
import h5py
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt, resample, welch
from scipy.stats import iqr, skew, kurtosis, entropy
from scipy.spatial.distance import pdist
from ripser import ripser
import matplotlib.pyplot as plt

# ========== USER SETTINGS ==========
MAT_FILE = "your_file.mat'
SUBJECT_IDX = 1
SEGMENT_START_IDX = 0
SEGMENT_END_IDX = 1000
CSV_SAVE_DIR = r"E:\RESULT"

ECG_SAMPLING_RATE = 125
PPG_SAMPLING_RATE = ECG_SAMPLING_RATE
MIN_HRV_PEAK_DIST_SEC = 0.6
BEAT_WINDOW_BEFORE_SEC = 0.2
BEAT_WINDOW_AFTER_SEC = 0.4
NORMALIZED_BEAT_LENGTH = 300
# ===================================


# ===== Utility Functions =====
def compute_ecg_snr(signal, fs=ECG_SAMPLING_RATE, band=(5, 15)):
    nyq = 0.5 * fs
    b, a = butter(3, [band[0] / nyq, band[1] / nyq], btype='band')
    sig_bp = filtfilt(b, a, signal)
    noise = signal - sig_bp
    power_sig = np.mean(sig_bp ** 2)
    power_noise = np.mean(noise ** 2) + 1e-12
    return 10 * np.log10(power_sig / power_noise)


def extract_tpv(signal):
    """
    Extract 39-dimensional TPV using Persistent Homology from a 1D signal.
    """
    if len(signal) < 3:
        return np.zeros(39, dtype=np.float32)

    # Delay embedding: (x_t, x_{t+1})
    x = np.stack([signal[:-1], signal[1:]], axis=1)
    dgms = ripser(x, maxdim=1)["dgms"][1]
    if dgms.size == 0:
        return np.zeros(39, dtype=np.float32)

    births, deaths = dgms[:, 0], dgms[:, 1]
    lifetimes = deaths - births
    lifetimes_sorted = np.sort(lifetimes)

    # Statistical features
    feats = [
        births.mean(), births.std(), deaths.mean(), deaths.std(),
        lifetimes.mean(), lifetimes.std(),
        lifetimes.max(), lifetimes.min(), np.median(lifetimes),
        iqr(lifetimes), skew(lifetimes), kurtosis(lifetimes),
        births.max(), births.min(), np.median(births),
        skew(births), kurtosis(births),
        deaths.max(), deaths.min(), np.median(deaths),
        skew(deaths), kurtosis(deaths),
        len(lifetimes),
        lifetimes_sorted[-1],
        lifetimes_sorted[-2] if len(lifetimes) > 1 else 0.0,
        lifetimes.max() / lifetimes.min() if lifetimes.min() > 0 else 0.0,
        np.sum(lifetimes ** 2)
    ]

    # Additional topological metrics
    lifetime_ratio = lifetimes / np.sum(lifetimes)
    PH_entropy = -np.sum(lifetime_ratio * np.log(lifetime_ratio + 1e-10))
    hist, _ = np.histogram(births, bins=10, range=(0, 1), density=True)
    Betti_entropy = entropy(hist + 1e-10)
    persistent_image_energy = np.sum(lifetimes ** 2)
    avg_persistence_distance = np.mean(pdist(lifetimes[:, None])) if len(lifetimes) > 1 else 0.0
    Gini_index = (
        (2 * np.sum(np.arange(1, len(lifetimes_sorted) + 1) * lifetimes_sorted)) /
        (len(lifetimes_sorted) * np.sum(lifetimes_sorted)) - (len(lifetimes_sorted) + 1) / len(lifetimes_sorted)
    ) if np.sum(lifetimes_sorted) > 0 else 0.0
    lifetime_variance = np.var(lifetimes)

    feats.extend([PH_entropy, Betti_entropy, persistent_image_energy, avg_persistence_distance, Gini_index, lifetime_variance])
    return np.array(feats, dtype=np.float32)


def extract_pqrst_tpv_and_snr(ecg_signal, fs=ECG_SAMPLING_RATE):
    """
    Extract TPV for ECG (R-peak centered beat) and compute SNR.
    """
    snr = compute_ecg_snr(ecg_signal, fs)
    nyq = 0.5 * fs
    b, a = butter(3, [5 / nyq, 15 / nyq], btype='band')
    ecg_bp = filtfilt(b, a, ecg_signal)
    peaks, _ = find_peaks(ecg_bp, distance=int(0.3 * fs), prominence=np.std(ecg_bp) * 0.5)

    if len(peaks) == 0:
        return np.zeros(39, dtype=np.float32), snr

    idx = peaks[len(peaks) // 2]
    start = int(idx - BEAT_WINDOW_BEFORE_SEC * fs)
    end = int(idx + BEAT_WINDOW_AFTER_SEC * fs)
    if start < 0 or end > len(ecg_signal):
        return np.zeros(39, dtype=np.float32), snr

    beat = ecg_signal[start:end]
    beat_norm = resample(beat, NORMALIZED_BEAT_LENGTH)
    tpv = extract_tpv(beat_norm)
    return tpv, snr


# HRV Features
def compute_time_domain(rr_ms):
    sdnn = np.std(rr_ms, ddof=1)
    rmssd = np.sqrt(np.mean(np.diff(rr_ms) ** 2))
    nn50 = np.sum(np.abs(np.diff(rr_ms)) > 50)
    nn20 = np.sum(np.abs(np.diff(rr_ms)) > 20)
    pnn50 = nn50 / len(rr_ms) * 100.0
    hist, bins = np.histogram(rr_ms, bins=100)
    tri_index = len(rr_ms) / np.max(hist)
    half = np.max(hist) / 2
    idxs = np.where(hist >= half)[0]
    tinn = bins[idxs[-1] + 1] - bins[idxs[0]] if len(idxs) > 1 else 0.0
    return sdnn, rmssd, pnn50, nn50, nn20, tri_index, tinn


def compute_freq_domain(rr_ms, fs_interp=4.0):
    rr = rr_ms / 1000.0
    t = np.cumsum(rr)
    if len(t) < 5:
        return np.nan, np.nan, np.nan
    ts_uniform = np.arange(t[0], t[-1], 1 / fs_interp)
    rr_uniform = np.interp(ts_uniform, t, rr_ms)
    f, pxx = welch(rr_uniform, fs=fs_interp, nperseg=min(len(rr_uniform), 256))
    lf_mask = (f >= 0.04) & (f < 0.15)
    hf_mask = (f >= 0.15) & (f < 0.4)
    lf = np.trapz(pxx[lf_mask], f[lf_mask])
    hf = np.trapz(pxx[hf_mask], f[hf_mask])
    lf_hf = lf / hf if hf > 0 else np.nan
    return lf, hf, lf_hf


def compute_poincare(rr_ms):
    diffs = np.diff(rr_ms)
    sd1 = np.sqrt(np.std(diffs, ddof=1) ** 2 / 2)
    sd2 = np.sqrt(2 * np.std(rr_ms, ddof=1) ** 2 - sd1 ** 2)
    return sd1, sd2


# Segment Extraction
def extract_segments_with_metrics(mat_path, start_idx, end_idx):
    rows = []
    basename = os.path.splitext(os.path.basename(mat_path))[0]

    with h5py.File(mat_path, 'r') as f:
        ppg_refs = f['Subj_Wins']['PPG_F'][0]
        ecg_refs = f['Subj_Wins']['ECG_F'][0]
        sbps = f['Subj_Wins']['SegSBP'][0]
        dbps = f['Subj_Wins']['SegDBP'][0]
        total = len(ppg_refs)
        end_idx = min(end_idx, total)

        for idx_rel, i in enumerate(range(start_idx, end_idx), start=1):
            print(f"⏳ [{basename}] Segment {idx_rel}/{end_idx - start_idx} (Index {i})...")
            try:
                ppg = f[ppg_refs[i]][()].squeeze().astype(np.float32)
                ecg = f[ecg_refs[i]][()].squeeze().astype(np.float32)
                sbp = float(f[sbps[i]][()][0][0])
                dbp = float(f[dbps[i]][()][0][0])
                row = {'SubjectID': basename, 'Segment': i, 'SBP': sbp, 'DBP': dbp}

                # PPG TPV
                ppg_tpv = extract_tpv(ppg)
                for j, v in enumerate(ppg_tpv):
                    row[f'PPG_TPV_{j}'] = v

                # ECG TPV + SNR
                ecg_tpv, snr = extract_pqrst_tpv_and_snr(ecg)
                for j, v in enumerate(ecg_tpv):
                    row[f'ECG_TPV_{j}'] = v
                row['ECG_SNR_dB'] = snr

                # TPV Difference Metrics
                tpv_diff = ppg_tpv - ecg_tpv
                row['TPV_L1'] = np.sum(np.abs(tpv_diff))
                row['TPV_L2'] = np.sqrt(np.sum(tpv_diff ** 2))
                row['TPV_COS'] = np.dot(ppg_tpv, ecg_tpv) / ((np.linalg.norm(ppg_tpv) * np.linalg.norm(ecg_tpv)) + 1e-8)
                row['TPV_CORR'] = np.corrcoef(ppg_tpv, ecg_tpv)[0, 1] if (np.std(ppg_tpv) > 1e-8 and np.std(ecg_tpv) > 1e-8) else np.nan

                rows.append(row)
            except Exception as e:
                print(f"[❗] Error on segment {i}: {e}")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print('⏳ Processing started...')
    df = extract_segments_with_metrics(MAT_FILE, SEGMENT_START_IDX, SEGMENT_END_IDX)

    if df.empty:
        print('⚠️ No data extracted.')
        exit()

    print("[Debug] Columns:", df.columns.tolist())
    os.makedirs(CSV_SAVE_DIR, exist_ok=True)
    basename = os.path.splitext(os.path.basename(MAT_FILE))[0]
    outf = os.path.join(CSV_SAVE_DIR, f"{basename}_S{SUBJECT_IDX}_metrics_{SEGMENT_START_IDX}_{SEGMENT_END_IDX}.csv")
    df.to_csv(outf, index=False)
    print(f"✅ Saved CSV: {outf}  shape={df.shape}")
