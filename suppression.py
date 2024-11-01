import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Generate a synthetic signal dataset
signal_data = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50, 55]).reshape(-1, 1)

# Initialize Min-Max Scaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Prepare to plot
plt.figure(figsize=(10, 6))

# Pre-FFT Min-Max Scaling
pre_fft_scaled = scaler.fit_transform(signal_data)

# Applying FFT
fft_data = np.fft.fft(signal_data.flatten())
fft_real = fft_data.real.reshape(-1, 1)
fft_imag = fft_data.imag.reshape(-1, 1)

# Apply Min-Max Scaling to real and imaginary parts separately
fft_real_scaled = scaler.fit_transform(fft_real)
fft_imag_scaled = scaler.fit_transform(fft_imag)
fft_scaled = fft_real_scaled + 1j * fft_imag_scaled

# Post-IFFT Min-Max Scaling
# Assume a simple logic to generate different precision levels
low_precision = fft_real_scaled * 0.5  # Low precision scaling
moderate_precision = fft_real_scaled * 0.75  # Moderate precision scaling
high_precision = fft_real_scaled * 1.0  # High precision scaling

# Custom x-axis for visualization
x_axis = np.linspace(0.005, 0.02, len(signal_data))

# Plotting
plt.plot(x_axis, low_precision, marker='o', linestyle='-', color='red', label="Low Precision (Suppression)")
plt.plot(x_axis, moderate_precision, marker='o', linestyle='-', color='orange', label="Moderate Precision (Suppression)")
plt.plot(x_axis, high_precision, marker='o', linestyle='-', color='green', label="High Precision (Suppression)")

# Title and labels
plt.title("Suppression at Pre-FFT using Min-Max Scaling Normalization")
plt.xlabel("Bit Error Rate (BER)")
plt.ylabel("Scaled Values")
plt.legend()
plt.grid(True)

# Annotate precision levels
plt.annotate("Low Precision", xy=(0.016, 0.25), xytext=(0.017, 0.4),
             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10, color='red')
plt.annotate("Moderate Precision", xy=(0.016, 0.5), xytext=(0.017, 0.65),
             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10, color='orange')
plt.annotate("High Precision", xy=(0.016, 0.75), xytext=(0.017, 0.9),
             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10, color='green')

plt.tight_layout()
plt.show()
