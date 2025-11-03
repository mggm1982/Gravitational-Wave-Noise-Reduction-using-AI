
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.signal import correlate, welch
from scipy.stats import pearsonr
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

def calculate_frequency_domain_metrics(clean, predicted, sample_rate=4096):

    f_clean, psd_clean = welch(clean.flatten(), fs=sample_rate, nperseg=1024)
    f_pred, psd_pred = welch(predicted.flatten(), fs=sample_rate, nperseg=1024)

    freq_correlation = np.corrcoef(psd_clean, psd_pred)[0, 1]

    spectral_distortion = np.mean(
        np.abs(np.log10(psd_pred + 1e-12) - np.log10(psd_clean + 1e-12)))

    return {
        'freq_correlation': freq_correlation,
        'spectral_distortion': spectral_distortion,
        'frequencies': f_clean,
        'psd_clean': psd_clean,
        'psd_predicted': psd_pred
    }

def analyze_model_performance():

    print("🔬 Advanced Gravitational Wave Denoiser Analysis")
    print("=" * 60)

    MODEL_PATH = '../../data/models/denoiser_model.keras'
    model = load_model(MODEL_PATH)

    X_val = np.load('../../data/processed/X_val.npy')
    y_val = np.load('../../data/processed/y_val.npy')

    print(f"Loaded model and {len(X_val)} validation samples")

    y_pred = model.predict(X_val, verbose=0)

    print("\n📊 PERFORMANCE BY SIGNAL AMPLITUDE")

    signal_amplitudes = [np.max(np.abs(y_val[i])) for i in range(len(y_val))]

    amplitude_bins = np.percentile(signal_amplitudes, [0, 25, 50, 75, 100])

    for i in range(len(amplitude_bins) - 1):
        mask = (np.array(signal_amplitudes) >= amplitude_bins[i]) & \
               (np.array(signal_amplitudes) < amplitude_bins[i+1])

        if np.sum(mask) > 0:
            bin_r_squared = 1 - np.sum((y_val[mask] - y_pred[mask])**2) / \
                np.sum((y_val[mask] - np.mean(y_val[mask]))**2)
            bin_mse = mean_squared_error(
                y_val[mask].flatten(), y_pred[mask].flatten())

            print(f"Amplitude {amplitude_bins[i]:.3f}-{amplitude_bins[i+1]:.3f}: "
                  f"R²={bin_r_squared:.3f}, MSE={bin_mse:.6f} ({np.sum(mask)} samples)")

    print("\n⏰ TEMPORAL PERFORMANCE ANALYSIS")

    segment_size = 512
    num_segments = 4096 // segment_size

    segment_performances = []
    for seg in range(num_segments):
        start_idx = seg * segment_size
        end_idx = (seg + 1) * segment_size

        seg_clean = y_val[:, start_idx:end_idx, :]
        seg_pred = y_pred[:, start_idx:end_idx, :]

        seg_r2 = 1 - np.sum((seg_clean - seg_pred)**2) / \
            np.sum((seg_clean - np.mean(seg_clean))**2)
        segment_performances.append(seg_r2)

        print(
            f"Time segment {seg+1} ({start_idx}-{end_idx}): R² = {seg_r2:.3f}")

    print("\n🌊 FREQUENCY DOMAIN ANALYSIS")

    sample_idx = len(y_val) // 2
    freq_metrics = calculate_frequency_domain_metrics(
        y_val[sample_idx], y_pred[sample_idx]
    )

    print(
        f"Frequency domain correlation: {freq_metrics['freq_correlation']:.3f}")
    print(f"Spectral distortion: {freq_metrics['spectral_distortion']:.4f}")

    print("\n📈 ERROR DISTRIBUTION ANALYSIS")

    errors = (y_pred - y_val).flatten()

    print(f"Error statistics:")
    print(f"  Mean error: {np.mean(errors):.6f}")
    print(f"  Error std: {np.std(errors):.6f}")
    print(f"  Error skewness: {(np.mean(errors**3) / np.std(errors)**3):.3f}")
    print(
        f"  Error kurtosis: {(np.mean(errors**4) / np.std(errors)**4 - 3):.3f}")

    error_percentiles = np.percentile(np.abs(errors), [50, 90, 95, 99, 99.9])
    print(f"Absolute error percentiles:")
    for i, pct in enumerate([50, 90, 95, 99, 99.9]):
        print(f"  {pct}th percentile: {error_percentiles[i]:.6f}")

    print("\n🚨 OUTLIER ANALYSIS")

    sample_r2_scores = []
    for i in range(len(y_val)):
        r2 = 1 - np.sum((y_val[i] - y_pred[i])**2) / \
            np.sum((y_val[i] - np.mean(y_val[i]))**2)
        sample_r2_scores.append(r2)

    poor_threshold = np.percentile(sample_r2_scores, 5)
    poor_samples = np.where(np.array(sample_r2_scores) < poor_threshold)[0]

    print(f"Samples with R² < {poor_threshold:.3f}:")
    for idx in poor_samples[:5]:
        print(f"  Sample {idx}: R² = {sample_r2_scores[idx]:.3f}")

    print("\n📊 GENERATING ANALYSIS PLOTS...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0, 0].scatter(signal_amplitudes, sample_r2_scores, alpha=0.6)
    axes[0, 0].set_xlabel('Signal Amplitude')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].set_title('Performance vs Signal Amplitude')
    axes[0, 0].grid(True)

    axes[0, 1].hist(errors, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Prediction Error')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Error Distribution')
    axes[0, 1].grid(True)

    axes[0, 2].plot(range(1, num_segments + 1), segment_performances, 'o-')
    axes[0, 2].set_xlabel('Time Segment')
    axes[0, 2].set_ylabel('R² Score')
    axes[0, 2].set_title('Performance Across Time')
    axes[0, 2].grid(True)

    axes[1, 0].loglog(freq_metrics['frequencies'], freq_metrics['psd_clean'],
                      label='Clean Signal', alpha=0.8)
    axes[1, 0].loglog(freq_metrics['frequencies'], freq_metrics['psd_predicted'],
                      label='Predicted', alpha=0.8)
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('Power Spectral Density')
    axes[1, 0].set_title('Frequency Domain Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].hist(sample_r2_scores, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('R² Score')
    axes[1, 1].set_ylabel('Number of Samples')
    axes[1, 1].set_title('R² Score Distribution')
    axes[1, 1].grid(True)

    sample_subset = np.random.choice(len(y_val), 1000)
    true_vals = y_val[sample_subset].flatten()
    pred_vals = y_pred[sample_subset].flatten()

    axes[1, 2].scatter(true_vals, pred_vals, alpha=0.3)
    axes[1, 2].plot([true_vals.min(), true_vals.max()],
                    [true_vals.min(), true_vals.max()], 'r--', lw=2)
    axes[1, 2].set_xlabel('True Values')
    axes[1, 2].set_ylabel('Predicted Values')
    axes[1, 2].set_title('Predicted vs True Values')
    axes[1, 2].grid(True)

    plt.tight_layout()
    plt.savefig('../../results/comprehensive_analysis.png',
                dpi=150, bbox_inches='tight')
    print("Comprehensive analysis plot saved to: results/comprehensive_analysis.png")
    plt.close()

    with open('../../results/detailed_analysis_report.txt', 'w') as f:
        f.write("=== DETAILED GRAVITATIONAL WAVE DENOISER ANALYSIS ===\n\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Validation samples: {len(X_val)}\n\n")

        f.write("PERFORMANCE BY AMPLITUDE:\n")
        for i in range(len(amplitude_bins) - 1):
            mask = (np.array(signal_amplitudes) >= amplitude_bins[i]) & \
                   (np.array(signal_amplitudes) < amplitude_bins[i+1])
            if np.sum(mask) > 0:
                bin_r_squared = 1 - np.sum((y_val[mask] - y_pred[mask])**2) / \
                    np.sum((y_val[mask] - np.mean(y_val[mask]))**2)
                f.write(
                    f"  {amplitude_bins[i]:.3f}-{amplitude_bins[i+1]:.3f}: R²={bin_r_squared:.3f}\n")

        f.write(f"\nFREQUENCY DOMAIN:\n")
        f.write(
            f"  Frequency correlation: {freq_metrics['freq_correlation']:.3f}\n")
        f.write(
            f"  Spectral distortion: {freq_metrics['spectral_distortion']:.4f}\n")

        f.write(f"\nERROR STATISTICS:\n")
        f.write(f"  Mean error: {np.mean(errors):.6f}\n")
        f.write(f"  Error std: {np.std(errors):.6f}\n")
        f.write(f"  50th percentile |error|: {error_percentiles[0]:.6f}\n")
        f.write(f"  95th percentile |error|: {error_percentiles[2]:.6f}\n")
        f.write(f"  99th percentile |error|: {error_percentiles[3]:.6f}\n")

    print("Detailed analysis report saved to: results/detailed_analysis_report.txt")
    print("\n✅ Advanced analysis complete!")

if __name__ == "__main__":
    analyze_model_performance()
