import numpy as np
import matplotlib.pyplot as plt

# Function to simulate SNR vs Precision comparison
def simulate_snr_precision(snr_range, bit_length):
    results = {
        "SNR": [],
        "Precision_Pre_IFFT": [],
        "Precision_Post_IFFT": [],
        "TP_Pre_IFFT": [],
        "FP_Pre_IFFT": [],
        "TN_Pre_IFFT": [],
        "FN_Pre_IFFT": [],
        "TP_Post_IFFT": [],
        "FP_Post_IFFT": [],
        "TN_Post_IFFT": [],
        "FN_Post_IFFT": []
    }

    for snr in snr_range:
        # Generate random bits for transmission
        bits = np.random.randint(0, 2, bit_length)

        # Simulate transmission over noisy channel (Pre-IFFT)
        noisy_bits_pre_ifft = add_noise(bits, snr)
        
        # Simulate transmission over noisy channel (Post-IFFT)
        noisy_bits_post_ifft = add_noise(bits, snr)  # You can change this based on your actual logic

        # Calculate metrics for Pre-IFFT
        tp_pre_ifft, fp_pre_ifft, tn_pre_ifft, fn_pre_ifft = evaluate_detection(bits, noisy_bits_pre_ifft)

        # Calculate metrics for Post-IFFT
        tp_post_ifft, fp_post_ifft, tn_post_ifft, fn_post_ifft = evaluate_detection(bits, noisy_bits_post_ifft)

        # Calculate Precision for both Pre and Post IFFT
        precision_pre_ifft = tp_pre_ifft / (tp_pre_ifft + fp_pre_ifft) if (tp_pre_ifft + fp_pre_ifft) > 0 else 0
        precision_post_ifft = tp_post_ifft / (tp_post_ifft + fp_post_ifft) if (tp_post_ifft + fp_post_ifft) > 0 else 0

        # Update results for reporting
        results["SNR"].append(snr)
        results["Precision_Pre_IFFT"].append(precision_pre_ifft)
        results["Precision_Post_IFFT"].append(precision_post_ifft)
        results["TP_Pre_IFFT"].append(tp_pre_ifft)
        results["FP_Pre_IFFT"].append(fp_pre_ifft)
        results["TN_Pre_IFFT"].append(tn_pre_ifft)
        results["FN_Pre_IFFT"].append(fn_pre_ifft)
        results["TP_Post_IFFT"].append(tp_post_ifft)
        results["FP_Post_IFFT"].append(fp_post_ifft)
        results["TN_Post_IFFT"].append(tn_post_ifft)
        results["FN_Post_IFFT"].append(fn_post_ifft)

    return results

def add_noise(bits, snr):
    """Add Gaussian noise to bits based on SNR."""
    noise_std = 1 / np.sqrt(2 * (10 ** (snr / 10)))
    noise = noise_std * np.random.randn(len(bits))
    return np.clip(bits + noise, 0, 1).round().astype(int)

def evaluate_detection(original_bits, received_bits):
    """Evaluate TP, FP, TN, FN counts."""
    tp = np.sum((original_bits == 1) & (received_bits == 1))
    fp = np.sum((original_bits == 0) & (received_bits == 1))
    tn = np.sum((original_bits == 0) & (received_bits == 0))
    fn = np.sum((original_bits == 1) & (received_bits == 0))
    return tp, fp, tn, fn

# Define SNR range and bit length for simulation
snr_range = np.arange(0, 20, 2)  # Example SNR range in dB
bit_length = 1000  # Number of bits in transmission

# Run the simulation
results = simulate_snr_precision(snr_range, bit_length)

# Plotting Precision for Pre-IFFT and Post-IFFT vs SNR
plt.figure(figsize=(12, 6))

# Plot Precision Pre-IFFT
color = 'tab:blue'
plt.plot(results["SNR"], results["Precision_Pre_IFFT"], marker='o', color=color, label='Precision Pre-IFFT', linewidth=2)

# Plot Precision Post-IFFT
color = 'tab:orange'
plt.plot(results["SNR"], results["Precision_Post_IFFT"], marker='o', color=color, label='Precision Post-IFFT', linewidth=2)

# Set labels and title
plt.xlabel('Bit Error Rate (BER)', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.title('Precision Pre-IFFT vs Post-IFFT', fontsize=16)
plt.grid(True)

# Add legend
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()
