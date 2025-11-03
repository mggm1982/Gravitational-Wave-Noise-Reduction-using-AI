import os
import sys
import numpy as np
from gwpy.timeseries import TimeSeries
from scipy.signal import welch, butter, sosfiltfilt
try:
    from scipy.signal import tukey
except ImportError:
    from scipy.signal.windows import tukey
from pycbc.waveform import get_td_waveform
from pycbc.types import TimeSeries as PTimeSeries

NOISE_FILE = '../../data/raw/H1_noise.hdf5'
OUT_DIR = '../../data/processed/'
os.makedirs(OUT_DIR, exist_ok=True)

FS = 4096
WIN_LEN = FS
N_SAMPLES = 5000
VAL_FRACTION = 0.2

PSD_SEG_SEC = 4
PSD_AVG_SEC = 256

M1_RANGE = (20.0, 80.0)
M2_RANGE = (20.0, 80.0)
SPIN_RANGE = (-0.5, 0.5)
DISTANCE_RANGE = (100, 1000)
F_LOWER = 20.0

SNR_DB_RANGE = (5.0, 20.0)
MAX_TIME_SHIFT = WIN_LEN // 2

RNG_SEED = 1337
rng = np.random.default_rng(RNG_SEED)

def bandlimit(x, fs=FS, lo=20.0, hi=1024.0, order=8):
    lo = max(1e-6, lo)
    sos = butter(order, [lo/(fs/2), hi/(fs/2)], btype='band', output='sos')
    return sosfiltfilt(sos, x)

def tapered(x, alpha=0.02):

    w = tukey(len(x), alpha=alpha)
    return x * w

def estimate_psd_from_noise(noise_arr, fs=FS, psd_total_sec=PSD_AVG_SEC, seg_sec=PSD_SEG_SEC):

    needed = int(psd_total_sec * fs)
    if len(noise_arr) < needed:
        raise ValueError(f"Need >= {psd_total_sec}s of noise; have {len(noise_arr)/fs:.1f}s")

    chunk = noise_arr[:needed]
    chunk = bandlimit(chunk, fs)
    nperseg = int(seg_sec * fs)
    f, psd = welch(chunk, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
    return f, psd

def whiten_with_psd(x, fs, f_psd, psd):

    freqs = np.fft.rfftfreq(len(x), 1.0/fs)
    psd_interp = np.interp(freqs, f_psd, psd)
    psd_interp = np.maximum(psd_interp, 1e-24)
    Xf = np.fft.rfft(x)
    Yf = Xf / np.sqrt(psd_interp + 1e-24)
    y = np.fft.irfft(Yf, n=len(x))
    return y - np.mean(y)

def random_cbc_waveform(fs=FS, win_len=WIN_LEN, debug_idx=None):

    m1 = rng.uniform(*M1_RANGE)
    m2 = rng.uniform(*M2_RANGE)
    chi1 = rng.uniform(*SPIN_RANGE)
    chi2 = rng.uniform(*SPIN_RANGE)
    distance = rng.uniform(*DISTANCE_RANGE)

    if debug_idx is not None and debug_idx < 3:
        print(f"    Waveform params: m1={m1:.1f}, m2={m2:.1f}, dist={distance:.0f}Mpc")

    hp, _ = get_td_waveform(
        approximant="IMRPhenomD",
        mass1=float(m1), mass2=float(m2),
        spin1z=float(chi1), spin2z=float(chi2),
        distance=float(distance),
        delta_t=1.0/fs, f_lower=float(F_LOWER)
    )

    sig = np.asarray(PTimeSeries(hp, delta_t=1.0/fs))
    sig = bandlimit(sig, fs)
    sig = tapered(sig)

    if len(sig) < win_len:
        sig = np.pad(sig, (win_len - len(sig), 0))
    else:
        sig = sig[-win_len:]

    return sig

def set_target_snr(signal_w, noise_w, target_snr_db):

    eps = 1e-12
    noise_rms = np.sqrt(np.mean(noise_w**2)) + eps
    signal_rms = np.sqrt(np.mean(signal_w**2)) + eps
    target_linear = 10.0 ** (target_snr_db / 20.0)
    desired_signal_rms = target_linear * noise_rms
    return desired_signal_rms / signal_rms

def main():
    print(f"Loading noise from: {NOISE_FILE}")
    noise_ts = TimeSeries.read(NOISE_FILE)
    noise = noise_ts.value
    print(f"Noise length: {len(noise)/FS:.1f} s")

    print("Estimating PSD...")
    f_psd, psd = estimate_psd_from_noise(noise, fs=FS)
    psd_len = int(PSD_AVG_SEC * FS)
    print(f"PSD estimated using {PSD_AVG_SEC}s of data.")

    X = np.zeros((N_SAMPLES, WIN_LEN), dtype=np.float32)
    y = np.zeros((N_SAMPLES, WIN_LEN), dtype=np.float32)

    for i in range(N_SAMPLES):
        if (i + 1) % 200 == 0:
            print(f"Generating sample {i+1}/{N_SAMPLES}")

        start = rng.integers(psd_len, len(noise) - WIN_LEN)
        noise_slice = noise[start:start+WIN_LEN]
        noise_slice = bandlimit(noise_slice, FS)

        sig = random_cbc_waveform(FS, WIN_LEN, debug_idx=i)

        noise_w = whiten_with_psd(noise_slice, FS, f_psd, psd)
        sig_w = whiten_with_psd(sig, FS, f_psd, psd)

        shift = rng.integers(0, MAX_TIME_SHIFT)
        sig_w = np.roll(sig_w, shift)

        snr_db = rng.uniform(*SNR_DB_RANGE)
        scale = set_target_snr(sig_w, noise_w, snr_db)
        sig_w *= scale

        x = noise_w + sig_w
        target = sig_w

        x = x / (np.max(np.abs(x)) + 1e-12)
        target = target / (np.max(np.abs(target)) + 1e-12)

        max_val = np.max(np.abs(x))
        if max_val > 0:
            x /= max_val
            target /= max_val

        if i < 5:
            diff = np.mean(np.abs(x - target))
            print(f"Sample {i}: SNR={snr_db:.1f}dB, mean abs diff={diff:.4f}")

        X[i] = x.astype(np.float32)
        y[i] = target.astype(np.float32)

    n_val = int(VAL_FRACTION * N_SAMPLES)
    X_train, X_val = X[:-n_val], X[-n_val:]
    y_train, y_val = y[:-n_val], y[-n_val:]

    X_train = X_train[..., None]
    X_val = X_val[..., None]
    y_train = y_train[..., None]
    y_val = y_val[..., None]

    np.save(os.path.join(OUT_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(OUT_DIR, 'X_val.npy'), X_val)
    np.save(os.path.join(OUT_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(OUT_DIR, 'y_val.npy'), y_val)

    print("\n✅ Saved:")
    print(f"  {OUT_DIR}/X_train.npy shape={X_train.shape}")
    print(f"  {OUT_DIR}/y_train.npy shape={y_train.shape}")
    print(f"  {OUT_DIR}/X_val.npy   shape={X_val.shape}")
    print(f"  {OUT_DIR}/y_val.npy   shape={y_val.shape}")

    diff = np.mean(np.abs(X_train - y_train))
    noise_level = np.std(X_train - y_train)
    signal_level = np.std(y_train)
    effective_snr = signal_level / noise_level if noise_level > 0 else float('inf')

    print(f"\nDataset Statistics:")
    print(f"  Mean abs diff: {diff:.4e}")
    print(f"  Noise std: {noise_level:.4e}")
    print(f"  Signal std: {signal_level:.4e}")
    print(f"  Effective SNR: {20*np.log10(effective_snr):.2f} dB")

    create_realistic_sample_plot(X_val[0], y_val[0])

def create_realistic_sample_plot(noisy, clean):

    import matplotlib.pyplot as plt
    t = np.linspace(0, 1, WIN_LEN)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    axes[0].plot(t, noisy.squeeze(), 'b', alpha=0.7, lw=0.8)
    axes[0].set_title("Noisy LIGO Input (Noise + Signal)")
    axes[0].set_ylabel("Strain")

    axes[1].plot(t, clean.squeeze(), 'r', lw=1.2)
    axes[1].set_title("Clean Target Signal")
    axes[1].set_ylabel("Strain")

    axes[2].plot(t, noisy.squeeze(), 'b', alpha=0.6, lw=0.8, label='Noisy Input')
    axes[2].plot(t, clean.squeeze(), 'r', lw=1.2, label='Clean Target')
    axes[2].legend()
    axes[2].set_title("Overlay")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Strain")

    plt.tight_layout()
    plt.savefig('../../results/realistic_ligo_sample.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Realistic sample plot saved at ../../results/realistic_ligo_sample.png")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("❌ Error:", e)
        sys.exit(1)
