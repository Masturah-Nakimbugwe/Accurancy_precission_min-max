import numpy as np

# Function to simulate SNR vs BER comparison
def simulate_snr_ber(snr_range, bit_length):
    # Use 'TN' or 'FN' explicitly for NP as needed
    results = {"SNR": [], "BER": [], "Desired Signal": [], "TP": [], "FP": [], "NP": []}

    for snr in snr_range:
        # Generate random bits for transmission
        bits = np.random.randint(0, 2, bit_length)
        # Simulate transmission over noisy channel
        noisy_bits = add_noise(bits, snr)

        # Calculate True Positives, False Positives, etc.
        tp, fp, tn, fn = evaluate_detection(bits, noisy_bits)
        
        # Calculate BER
        ber = calculate_ber(bits, noisy_bits)

        # Update results for reporting
        results["SNR"].append(snr)
        results["BER"].append(ber)
        results["Desired Signal"].append(np.sum(bits))  # Total number of '1's in original bits
        results["TP"].append(tp)
        results["FP"].append(fp)
        
        # Choose either tn or fn for NP as needed
        results["NP"].append(fn)  # If NP is intended as False Negative (FN)
        # results["NP"].append(tn)  # If NP is intended as True Negative (TN)

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
results = simulate_snr_ber(snr_range, bit_length)

# Report results
for i, snr in enumerate(results["SNR"]):
    print(f"SNR: {snr} dB")
    print(f"  BER: {results['BER'][i]:.5f}")
    print(f"  Desired Signal: {results['Desired Signal'][i]}")
    print(f"  True Positive (TP): {results['TP'][i]}")
    print(f"  False Positive (FP): {results['FP'][i]}")
    print(f"  Negative Positive (NP): {results['NP'][i]}")
    print()
