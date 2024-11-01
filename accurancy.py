import numpy as np
import matplotlib.pyplot as plt

# BER values from 0.008 to 0.015
ber_values = np.linspace(0.008, 0.015, 10)

# Sample data for each metric
tp_values = [0.98, 0.96, 0.95, 0.93, 0.92, 0.90, 0.88, 0.87, 0.85, 0.84]  # True Positives
tn_values = [0.97, 0.96, 0.94, 0.93, 0.91, 0.89, 0.88, 0.86, 0.85, 0.83]  # True Negatives
fp_values = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11]   # False Positives
fn_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]   # False Negatives

# Plot each metric
plt.figure(figsize=(12, 8))

plt.plot(ber_values, tn_values, marker='o', color='g', linestyle='-', linewidth=2, markersize=6, label="True Negatives (TN)")
plt.plot(ber_values, fp_values, marker='o', color='r', linestyle='-', linewidth=2, markersize=6, label="False Positives (FP)")
plt.plot(ber_values, fn_values, marker='o', color='purple', linestyle='-', linewidth=2, markersize=6, label="False Negatives (FN)")

# Adding titles and labels
plt.title('Accurancy vs Bit Error Rate (BER)')
plt.xlabel('Bit Error Rate (BER)')
plt.ylabel('Accurancy')
plt.grid(True)
plt.legend()
plt.ylim(0, 1)  # Ensures scale is from 0 to 1 for metrics

# Display the plot
plt.show()
