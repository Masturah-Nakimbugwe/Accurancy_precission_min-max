import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Synthetic signal data (e.g., SNR values for demonstration)
signal_data = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50, 55]).reshape(-1, 1)

# Initialize Min-Max Scaler to scale between [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))

# 1. Pre-FFT Min-Max Scaling
pre_fft_scaled = scaler.fit_transform(signal_data)

# Applying FFT (For demonstration, we apply FFT on original data to compare)
fft_data = np.fft.fft(signal_data.flatten())

# Separate real and imaginary parts
fft_real = fft_data.real.reshape(-1, 1)
fft_imag = fft_data.imag.reshape(-1, 1)

# Apply Min-Max Scaling to real and imaginary parts separately
fft_real_scaled = scaler.fit_transform(fft_real)
fft_imag_scaled = scaler.fit_transform(fft_imag)

# Recombine scaled real and imaginary parts if necessary
fft_scaled = fft_real_scaled + 1j * fft_imag_scaled

# 2. Pre-IFFT Min-Max Scaling
# Apply scaling on the FFT data, then perform the inverse FFT (IFFT)
ifft_data = np.fft.ifft(fft_scaled.flatten())

# 3. Post-IFFT Min-Max Scaling
# Scale after applying IFFT
ifft_real = ifft_data.real.reshape(-1, 1)
ifft_imag = ifft_data.imag.reshape(-1, 1)
post_ifft_real_scaled = scaler.fit_transform(ifft_real)
post_ifft_imag_scaled = scaler.fit_transform(ifft_imag)

# Custom x-axis from 0.005 to 0.02 with the same length as the data
x_axis = np.linspace(0.005, 0.02, len(signal_data))

# Plotting all scaled data
plt.figure(figsize=(14, 10))

# Pre-FFT Scaling Plot
plt.subplot(3, 1, 1)
plt.plot(x_axis, signal_data, marker='o', color='blue', label="Original Signal Data")
plt.plot(x_axis, pre_fft_scaled, marker='o', color='red', label="Pre-FFT Scaled")
plt.title("\nNormalization Scaling under Pre-FFT Suppression")
plt.xlabel("Bit Error Rate (BER)")
plt.ylabel("MIN-MAX SCALING")
plt.legend()
plt.grid(True)

# Pre-IFFT Scaling Plot
plt.subplot(3, 1, 2)
plt.plot(x_axis, np.abs(fft_data), marker='o', color='purple', label="FFT Data (Magnitude)")
plt.plot(x_axis, np.abs(ifft_data), marker='o', color='green', label="Pre-IFFT Scaled Data")
plt.title("\nNormalization Scaling under Pre-IFFT Suppression")
plt.xlabel("Bit Error Rate (BER)")
plt.ylabel("MIN-MAX SCALING")
plt.legend()
plt.grid(True)

# Post-IFFT Scaling Plot
plt.subplot(3, 1, 3)
plt.plot(x_axis, np.abs(ifft_data), marker='o', color='orange', label="Post-IFFT Original Data (Magnitude)")
plt.plot(x_axis, np.hstack((post_ifft_real_scaled, post_ifft_imag_scaled)), marker='o', color='brown', label="Post-IFFT Scaled Data")
plt.title("Normalization Scaling under Post-IFFT Suppression")
plt.xlabel("Bit Error Rate (BER)")
plt.ylabel("MIN-MAX SCALING")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
