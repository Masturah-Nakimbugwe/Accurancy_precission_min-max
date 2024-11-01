import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Generate a synthetic signal dataset
signal_data = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50, 55]).reshape(-1, 1)

# Initialize Min-Max Scaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Prepare to plot
plt.figure(figsize=(14, 10))

# Pre-IFFT Min-Max Scaling
pre_ifft_scaled = scaler.fit_transform(signal_data)

# Applying FFT
fft_data = np.fft.fft(signal_data.flatten())
fft_real = fft_data.real.reshape(-1, 1)
fft_imag = fft_data.imag.reshape(-1, 1)

# Apply Min-Max Scaling to real and imaginary parts separately
fft_real_scaled = scaler.fit_transform(fft_real)
fft_imag_scaled = scaler.fit_transform(fft_imag)
fft_scaled = fft_real_scaled + 1j * fft_imag_scaled

# Post-IFFT: Apply IFFT and then scale
ifft_data = np.fft.ifft(fft_scaled.flatten())
ifft_real = ifft_data.real.reshape(-1, 1)
ifft_imag = ifft_data.imag.reshape(-1, 1)
post_ifft_scaled = scaler.fit_transform(ifft_real)  # Scale only the real part for simplicity

# Generate different precision levels for Pre-IFFT
low_precision_pre_ifft = pre_ifft_scaled * 0.5
moderate_precision_pre_ifft = pre_ifft_scaled * 0.75
high_precision_pre_ifft = pre_ifft_scaled * 1.0

# Generate different precision levels for Post-IFFT
low_precision_post_ifft = post_ifft_scaled * 0.5
moderate_precision_post_ifft = post_ifft_scaled * 0.75
high_precision_post_ifft = post_ifft_scaled * 1.0

# Generate different precision levels for Pre-FFT
low_precision_fft = fft_real_scaled * 0.5
moderate_precision_fft = fft_real_scaled * 0.75
high_precision_fft = fft_real_scaled * 1.0

# Custom x-axis for visualization
x_axis = np.linspace(0.005, 0.02, len(signal_data))

# Plotting Pre-IFFT Suppression
plt.subplot(3, 1, 1)
plt.plot(x_axis, low_precision_pre_ifft, marker='o', linestyle='-', color='red', label="Low Persuasion (Pre-IFFT)")
plt.plot(x_axis, moderate_precision_pre_ifft, marker='o', linestyle='-', color='orange', label="Moderate Persuasion (Pre-IFFT)")
plt.plot(x_axis, high_precision_pre_ifft, marker='o', linestyle='-', color='green', label="High Persuasion (Pre-IFFT)")
plt.title("Suppression at Pre-IFFT using Min-Max Scaling Normalization")
plt.xlabel("\nBit Error Rate (BER)")
plt.ylabel("Normalization")
plt.legend()
plt.grid(True)

# Plotting Post-IFFT Suppression
plt.subplot(3, 1, 2)
plt.plot(x_axis, low_precision_post_ifft, marker='o', linestyle='-', color='red', label="Low Persuasion (Post-IFFT)")
plt.plot(x_axis, moderate_precision_post_ifft, marker='o', linestyle='-', color='orange', label="Moderate Persuasion (Post-IFFT)")
plt.plot(x_axis, high_precision_post_ifft, marker='o', linestyle='-', color='green', label="High Persuasion (Post-IFFT)")
plt.title("\nSuppression at Post-IFFT using Min-Max Scaling Normalization")
plt.xlabel("\nBit Error Rate (BER)")
plt.ylabel("Normalization")
plt.legend()
plt.grid(True)

# Plotting Pre-FFT Suppression
plt.subplot(3, 1, 3)
plt.plot(x_axis, low_precision_fft, marker='o', linestyle='-', color='red', label="Low Persuasion (Pre-FFT)")
plt.plot(x_axis, moderate_precision_fft, marker='o', linestyle='-', color='orange', label="Moderate Persuasion (Pre-FFT)")
plt.plot(x_axis, high_precision_fft, marker='o', linestyle='-', color='green', label="High Persuasion (Pre-FFT)")
plt.title("Suppression at Pre-FFT using Min-Max Scaling Normalization")
plt.xlabel("Bit Error Rate (BER)")
plt.ylabel("Normalization")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
