import numpy as np
import matplotlib.pyplot as plt

# Create a time vector and a sample signal with noise
sampling_rate = 1000  # samples per second
t = np.linspace(0, 1, sampling_rate, endpoint=False)  # 1 second of data
freq_signal = 5  # Frequency of the original signal in Hz
signal = np.sin(2 * np.pi * freq_signal * t)  # Original clean signal

# Add high-frequency noise
noise_freq = 50  # Frequency of the noise in Hz
noise = 0.5 * np.sin(2 * np.pi * noise_freq * t)  # Noise component
noisy_signal = signal + noise  # Combined signal with noise

# Apply FFT to the noisy signal
fft_noisy_signal = np.fft.fft(noisy_signal)
fft_freqs = np.fft.fftfreq(sampling_rate, 1 / sampling_rate)

# Post-FFT filtering (suppress high frequencies)
fft_filtered = np.copy(fft_noisy_signal)
cutoff_freq = 10  # Frequency threshold for filtering in Hz
fft_filtered[np.abs(fft_freqs) > cutoff_freq] = 0  # Suppress frequencies above cutoff

# Apply IFFT to get the filtered signal back in the time domain
filtered_signal = np.fft.ifft(fft_filtered)

# Plotting the results
plt.figure(figsize=(15, 10))

# Original Signal
plt.subplot(3, 1, 1)
plt.plot(t, signal, color='blue', label='Original Signal')
plt.title("Original Signal (5 Hz)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

# Noisy Signal
plt.subplot(3, 1, 2)
plt.plot(t, noisy_signal, color='orange', label='Noisy Signal (5 Hz + 50 Hz noise)')
plt.title("Noisy Signal (with 50 Hz Noise)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

# Filtered Signal after Post-FFT Technique
plt.subplot(3, 1, 3)
plt.plot(t, filtered_signal.real, color='green', label='Filtered Signal (Post-FFT)')
plt.title("Filtered Signal using Post-FFT Suppression (Low-pass Filter with 10 Hz Cutoff)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
