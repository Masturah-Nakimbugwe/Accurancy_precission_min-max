import numpy as np
import matplotlib.pyplot as plt

# Define the time array
t = np.linspace(0, 1, 1000)  # 1 second of data at 1000 Hz

# Define a desired signal (e.g., a sinusoidal wave)
frequency_desired = 5  # 5 Hz
desired_signal = np.sin(2 * np.pi * frequency_desired * t)  # Desired signal

# Define a jamming signal (e.g., noise)
jamming_signal = 0.5 * np.sin(2 * np.pi * 30 * t)  # Jamming at 30 Hz

# Combine desired signal with jamming
combined_signal = desired_signal + jamming_signal

# Calculate powers
P_target = np.mean(desired_signal ** 2)  # Power of target signal
P_combined = np.mean(combined_signal ** 2)  # Power of combined signal

# Define the values for True Negatives, False Positives, and False Negatives
tn_values = [0.97, 0.96, 0.94, 0.93, 0.91, 0.89, 0.88, 0.86, 0.85, 0.83]  # True Negatives
fp_values = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11]   # False Positives
fn_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]   # False Negatives

# Assume True Positives (TP) for the calculation of precision
tp_values = [0.98, 0.97, 0.95, 0.93, 0.90, 0.88, 0.85, 0.83, 0.82, 0.80]  # Example True Positives

# Calculate Precision for metrics
precision_values = np.array(tp_values) / (np.array(tp_values) + np.array(fp_values))

# Define BER values based on twice the frequency error
frequency_errors = np.linspace(0, 0.05, len(tn_values))  # Frequency errors from 0 to 0.05
ber_values = 2 * frequency_errors  # Twice the frequency error gives us BER values

# Estimate probability of jamming signal presence
# Assuming we have a simple metric based on the power of combined signal
# If the power of combined signal is above a threshold, we assume jamming is present
threshold = 0.1  # Define a threshold for detecting jamming
prob_jamming = np.clip((P_combined - threshold) / (P_target - threshold), 0, 1)

# Set up the figure and axes for plotting
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# Plot the desired signal in the first subplot
axs[0].plot(t, desired_signal, label='Desired Signal', color='blue', linewidth=2)
axs[0].set_title('Amplitude vs Time', fontsize=16)
axs[0].set_xlabel('Time (seconds)', fontsize=14)
axs[0].set_ylabel('Amplitude', fontsize=14)
axs[0].legend()
axs[0].grid(True)

# Plot each metric for precision in the second subplot
axs[1].plot(ber_values, tn_values, marker='o', color='g', linestyle='-', linewidth=2, markersize=6, label="True Negatives (TN)")
axs[1].plot(ber_values, fp_values, marker='o', color='r', linestyle='-', linewidth=2, markersize=6, label="False Positives (FP)")
axs[1].plot(ber_values, fn_values, marker='o', color='purple', linestyle='-', linewidth=2, markersize=6, label="False Negatives (FN)")

# Add labels and title for the second plot
axs[1].set_title('\nPrecision vs Bit Error Rate', fontsize=16)
axs[1].set_xlabel('Bit Error Rate (BER)', fontsize=14)
axs[1].set_ylabel('Precision', fontsize=14)
axs[1].set_ylim(0, 1)
axs[1].set_xlim(0, 0.1)
axs[1].grid(True)
axs[1].legend()

# Show the plots
plt.tight_layout()
plt.show()
