import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Generate three different synthetic signal datasets
signal_data1 = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50, 55]).reshape(-1, 1)
signal_data2 = np.array([5, 12, 18, 22, 27, 33, 39, 46, 52, 58]).reshape(-1, 1)
signal_data3 = np.array([8, 14, 19, 24, 29, 34, 41, 49, 53, 60]).reshape(-1, 1)

# Initialize Min-Max Scaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Prepare to plot
plt.figure(figsize=(14, 10))

# Function to process and plot each signal dataset
def process_and_plot(signal_data, idx):
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

    # Apply IFFT
    ifft_data = np.fft.ifft(fft_scaled.flatten())

    # Post-IFFT Min-Max Scaling
    ifft_real = ifft_data.real.reshape(-1, 1)
    ifft_imag = ifft_data.imag.reshape(-1, 1)
    post_ifft_real_scaled = scaler.fit_transform(ifft_real)
    post_ifft_imag_scaled = scaler.fit_transform(ifft_imag)

    # Custom x-axis for the current dataset
    x_axis = np.linspace(0.005 + idx * 0.001, 0.02 + idx * 0.001, len(signal_data))

    # Plot for detection (Pre-FFT) and suppression (Post-IFFT)
    plt.plot(x_axis, pre_fft_scaled, marker='o', label=f"Pre-FFT (Dataset {idx + 1}) = Low Persuasion", linestyle='-', color='red')
    plt.plot(x_axis, np.hstack((post_ifft_real_scaled, post_ifft_imag_scaled)), marker='o', label=f"Post-IFFT (Dataset {idx + 1}) = High Persuasion", linestyle='--', color='green')

    # Adding Pre-IFFT line for moderate persuasion
    plt.plot(x_axis, fft_real_scaled, marker='x', label=f"Pre-IFFT (Dataset {idx + 1}) = Moderate Persuasion", linestyle=':', color='blue')

# Process and plot each signal dataset
process_and_plot(signal_data1, 0)
process_and_plot(signal_data2, 1)
process_and_plot(signal_data3, 2)

# Title and labels
plt.title("Performance Under Detection and Suppression Across Datasets")
plt.xlabel("Bit Error Rate (BER)")
plt.ylabel("Normalization")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
