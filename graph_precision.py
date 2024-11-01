import numpy as np
import matplotlib.pyplot as plt

# Function to simulate SNR vs Precision comparison
def simulate_snr_precision(snr_range, bit_length):
    results = {"SNR": [], "Precision": [], "Desired Signal": [], "TP": [], "FP": [], "TN": [], "FN": []}

    for snr in snr_range:
        # Generate random bits for transmission
        bits = np.random.randint(0, 2, bit_length)
        # Simulate transmission over noisy channel
        noisy_bits = add_noise(bits, snr)

        # Calculate True Positives, False Positives, etc.
        tp, fp, tn, fn = evaluate_detection(bits, noisy_bits)
        
        # Calculate Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        # Update results for reporting
        results["SNR"].append(snr)
        results["Precision"].append(precision)
        results["Desired Signal"].append(np.sum(bits))  # Total number of '1's in original bits
        results["TP"].append(tp)
        results["FP"].append(fp)
        results["TN"].append(tn)
        results["FN"].append(fn)

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

def calculate_ber(original_bits, received_bits):
    """Calculate Bit Error Rate."""
    errors = np.sum(original_bits != received_bits)
    return errors / len(original_bits)

# Define SNR range and bit length for simulation
snr_range = np.arange(0, 20, 2)  # Example SNR range in dB
bit_length = 1000  # Number of bits in transmission

# Run the simulation
results = simulate_snr_precision(snr_range, bit_length)

# Plotting Precision, TP, FP, TN, and FN vs SNR on the same graph
plt.figure(figsize=(10, 5))

# Create the first y-axis for Precision
color = 'tab:blue'
plt.plot(results["SNR"], results["Precision"], marker='o', color=color, label='Precision', linewidth=2)
plt.xlabel('BER')  # Assuming you want BER here
plt.ylabel('Precision', color=color)
plt.tick_params(axis='y', labelcolor=color)

# Create the second y-axis for TP
ax2 = plt.gca().twinx()  
color = 'tab:green'
ax2.plot(results["SNR"], results["TP"], marker='o', color=color,  linewidth=2)

ax2.tick_params(axis='y', labelcolor=color)

# Create the third y-axis for FP
ax3 = plt.gca().twinx()  
color = 'tab:red'
ax3.spines['right'].set_position(('outward', 60))  # Offset the third y-axis
ax3.plot(results["SNR"], results["FP"], marker='o', color=color, linewidth=2)

ax3.tick_params(axis='y', labelcolor=color)

# Create the fourth y-axis for TN
ax4 = plt.gca().twinx()
color = 'tab:orange'
ax4.spines['right'].set_position(('outward', 120))  # Offset the fourth y-axis
ax4.plot(results["SNR"], results["TN"], marker='o', color=color, linewidth=2)

ax4.tick_params(axis='y', labelcolor=color)

# Create the fifth y-axis for FN
ax5 = plt.gca().twinx()
color = 'tab:purple'
ax5.spines['right'].set_position(('outward', 180))  # Offset the fifth y-axis
ax5.plot(results["SNR"], results["FN"], marker='o', color=color, linewidth=2)

ax5.tick_params(axis='y', labelcolor=color)

# Adding grid lines to the plot
plt.grid(True)

# Create a single legend for all metrics
lines, labels = plt.gca().get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines3, labels3 = ax3.get_legend_handles_labels()
lines4, labels4 = ax4.get_legend_handles_labels()
lines5, labels5 = ax5.get_legend_handles_labels()

# Combine all labels and lines
lines += lines2 + lines3 + lines4 + lines5
labels += labels2 + labels3 + labels4 + labels5
plt.legend(lines, labels, loc='upper left')

# Show the plot
plt.show()
