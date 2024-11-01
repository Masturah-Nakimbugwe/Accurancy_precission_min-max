import numpy as np
import matplotlib.pyplot as plt

# Original feature values
original_values = np.array([10, 15, 20, 25, 30])

# Calculate min and max
X_min = original_values.min()
X_max = original_values.max()

# Apply Min-Max Scaling formula
scaled_values = (original_values - X_min) / (X_max - X_min)

# Plotting the original and scaled values
plt.figure(figsize=(10, 6))

# Plot original values
plt.plot(range(len(original_values)), original_values, marker='o', color='b', linestyle='-', linewidth=2, markersize=6, label="Original Values")

# Plot scaled values
plt.plot(range(len(scaled_values)), scaled_values, marker='o', color='orange', linestyle='-', linewidth=2, markersize=6, label="Scaled Values (Min-Max)")

# Adding titles and labels
plt.title("Min-Max Scaling Normalization Demonstration")
plt.xlabel("Data Index")
plt.ylabel("Feature Value")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
