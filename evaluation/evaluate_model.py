import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import os
import sys
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

def calculate_snr(clean, noise):
    eps = 1e-12
    clean_power = np.mean(clean**2)
    noise_power = np.mean(noise**2) + eps
    return 10 * np.log10(clean_power / noise_power + eps)

def calculate_snr_improvement(noisy, clean, denoised):

    original_noise = noisy - clean
    residual_noise = denoised - clean
    snr_before = calculate_snr(clean, original_noise)
    snr_after = calculate_snr(clean, residual_noise)
    return snr_after - snr_before

def calculate_cross_correlation(s1, s2):

    s1 = (s1 - np.mean(s1)) / (np.std(s1) + 1e-12)
    s2 = (s2 - np.mean(s2)) / (np.std(s2) + 1e-12)
    return float(np.mean(s1 * s2))

def waveform_similarity(clean, predicted):
    clean_norm = clean / (np.linalg.norm(clean) + 1e-12)
    pred_norm = predicted / (np.linalg.norm(predicted) + 1e-12)
    return np.sum(clean_norm * pred_norm)

def calculate_metrics(noisy, clean, predicted):

    noisy = np.squeeze(noisy)
    clean = np.squeeze(clean)
    predicted = np.squeeze(predicted)

    metrics = {}

    metrics['mse'] = mean_squared_error(clean, predicted)
    metrics['mae'] = mean_absolute_error(clean, predicted)
    metrics['rmse'] = np.sqrt(metrics['mse'])

    metrics['pearson_r'], _ = pearsonr(clean.flatten(), predicted.flatten())
    metrics['cross_correlation'] = calculate_cross_correlation(clean, predicted)

    ss_res = np.sum((clean - predicted) ** 2)
    ss_tot = np.sum((clean - np.mean(clean)) ** 2)
    metrics['r_squared'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    metrics['snr_improvement'] = calculate_snr_improvement(noisy, clean, predicted)
    metrics['waveform_similarity'] = waveform_similarity(clean, predicted)

    orig_noise = noisy - clean
    resid_noise = predicted - clean
    orig_power = np.mean(orig_noise ** 2)
    resid_power = np.mean(resid_noise ** 2)
    metrics['noise_reduction_pct'] = (
        max(0.0, (1 - resid_power / (orig_power + 1e-12))) * 100
    )

    return metrics

print("Step 8: Evaluating and Visualizing Model Performance...")

try:
    MODEL_PATH = '../../data/models/denoiser_model.keras'
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at '{MODEL_PATH}'")
        print("Please run Step 7 to train and save your model first.")
        sys.exit(1)

    print(f"Loading trained model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully.")

    print("Loading validation data...")
    X_val = np.load('../../data/processed/X_val.npy')
    y_val = np.load('../../data/processed/y_val.npy')
    print(f"Loaded {len(X_val)} validation samples.")

    print("\n=== QUANTITATIVE EVALUATION ===")
    print("Generating predictions on validation set...")
    y_pred = model.predict(X_val, verbose=0)

    overall_metrics = calculate_metrics(X_val, y_val, y_pred)
    print(f"\n📊 OVERALL PERFORMANCE")
    print("┌──────────────────────────────────────────┐")
    print(f"│ MSE:                {overall_metrics['mse']:.6f}")
    print(f"│ RMSE:               {overall_metrics['rmse']:.6f}")
    print(f"│ MAE:                {overall_metrics['mae']:.6f}")
    print(f"├──────────────────────────────────────────┤")
    print(f"│ Pearson r:          {overall_metrics['pearson_r']:.4f}")
    print(f"│ Cross-Correlation:  {overall_metrics['cross_correlation']:.4f}")
    print(f"│ R²:                 {overall_metrics['r_squared']:.4f}")
    print(f"├──────────────────────────────────────────┤")
    print(f"│ SNR Improvement:    {overall_metrics['snr_improvement']:.2f} dB")
    print(f"│ Noise Reduction:    {overall_metrics['noise_reduction_pct']:.2f}%")
    print(f"│ Waveform Similarity:    {overall_metrics['waveform_similarity']:.2f}%")
    print("└──────────────────────────────────────────┘")

    print("\n📈 SAMPLE STATISTICS (1000 validation samples)")
    sample_metrics = [calculate_metrics(X_val[i:i+1], y_val[i:i+1], y_pred[i:i+1])
                      for i in range(len(X_val))]

    sample_r2 = np.array([m['r_squared'] for m in sample_metrics])
    sample_snr = np.array([m['snr_improvement'] for m in sample_metrics])
    sample_nr = np.array([m['noise_reduction_pct'] for m in sample_metrics])

    print(f"  • R² mean: {sample_r2.mean():.3f} ± {sample_r2.std():.3f}")
    print(f"  • SNR gain mean: {sample_snr.mean():.2f} ± {sample_snr.std():.2f} dB")
    print(f"  • Noise reduction: {sample_nr.mean():.2f}% ± {sample_nr.std():.2f}%")

    print("\n=== VISUAL EVALUATION ===")
    num_samples_to_plot = 3
    for _ in range(num_samples_to_plot):
        idx = random.randint(0, len(X_val) - 1)
        noisy = X_val[idx]
        clean = y_val[idx]
        denoised = y_pred[idx]

        m = calculate_metrics(noisy, clean, denoised)

        print(f"\n📊 Sample {idx}: R²={m['r_squared']:.3f}, "
              f"SNR Δ={m['snr_improvement']:.2f} dB, "
              f"Noise↓={m['noise_reduction_pct']:.2f}%")

        time = np.linspace(0, 1, noisy.shape[0])

        plt.figure(figsize=(15, 12))

        plt.subplot(4, 1, 1)
        plt.plot(time, noisy, label="Noisy Input", alpha=0.7)
        plt.title(f"Original Noisy Input (Sample {idx})")
        plt.ylabel("Strain")
        plt.legend()
        plt.grid(True)

        plt.subplot(4, 1, 2)
        plt.plot(time, denoised, color='r', label="AI Denoised Output")
        plt.title(f"AI Output | R²={m['r_squared']:.3f} | SNR Gain=+{m['snr_improvement']:.2f} dB")
        plt.ylabel("Strain")
        plt.legend()
        plt.grid(True)

        plt.subplot(4, 1, 3)
        plt.plot(time, clean, color='g', label="Ground Truth (Clean Signal)")
        plt.title(f"Ground Truth | Noise Reduction={m['noise_reduction_pct']:.2f}%")
        plt.ylabel("Strain")
        plt.legend()
        plt.grid(True)

        plt.subplot(4, 1, 4)
        residual = denoised - clean
        plt.plot(time, residual, color='orange', alpha=0.8, label="Residual Error")
        plt.title(f"Prediction Error | RMSE={m['rmse']:.6f}")
        plt.xlabel("Time (s)")
        plt.ylabel("Error")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        save_path = f"../../results/evaluation_sample_{idx}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   → Plot saved: {save_path}")

    print("\n💾 Writing evaluation report...")
    report_path = "../../results/evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write("=== GRAVITATIONAL WAVE DENOISER EVALUATION REPORT ===\n\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Validation Samples: {len(X_val)}\n\n")
        f.write("OVERALL PERFORMANCE:\n")
        for k, v in overall_metrics.items():
            if k == "noise_reduction_pct":
                f.write(f"  {k:25s}: {v:.2f}%\n")
            elif "snr" in k:
                f.write(f"  {k:25s}: {v:.2f} dB\n")
            else:
                f.write(f"  {k:25s}: {v:.6f}\n")
        f.write("\nSAMPLE STATISTICS:\n")
        f.write(f"  R² Mean ± Std: {sample_r2.mean():.3f} ± {sample_r2.std():.3f}\n")
        f.write(f"  SNR Gain Mean ± Std: {sample_snr.mean():.2f} ± {sample_snr.std():.2f} dB\n")
        f.write(f"  Noise Reduction: {sample_nr.mean():.2f}% ± {sample_nr.std():.2f}%\n")

    print(f"✅ Report saved to: {report_path}")
    print("\n🎉 Step 8 Complete — Evaluation finished successfully.")

    print("\n📊 SUMMARY:")
    print(f"   • Overall R²: {overall_metrics['r_squared']:.3f}")
    print(f"   • Mean SNR Improvement: {overall_metrics['snr_improvement']:.2f} dB")
    print(f"   • Mean Noise Reduction: {overall_metrics['noise_reduction_pct']:.2f}%")
    print(f"   • Pearson Correlation: {overall_metrics['pearson_r']:.3f}")
    print("\nCheck:")
    print("   → PNG plots in '../../results/'")
    print("   → Detailed metrics in 'evaluation_report.txt'")

except Exception as e:
    print(f"❌ Error during evaluation: {e}")
    sys.exit(1)
