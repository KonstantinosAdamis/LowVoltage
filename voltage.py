import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Redefine the constants for the new functions in the convolution
A1 = 125000               # Amplitude for the first function
alpha1 = 64000            # Exponential decay rate for the first function
omega1 = 86130            # Frequency of the sine wave for the first function

A2 = 177187               # Amplitude for the second function
alpha2 = 8.85936e7        # Exponential decay rate for the second function

# Define the two functions for convolution
def g1(t):
    return A1 * np.exp(-alpha1 * t) * np.sin(omega1 * t)

def g2(t):
    return A2 * np.exp(-alpha2 * t)

# Convolution using numerical integration
def convolution_g(t):
    # Generate integration points for the convolution
    integrand = lambda tau: g1(tau) * g2(t - tau)
    # Perform the convolution integral
    result, _ = quad(integrand, 0, t)
    return result

# Define the time range for convolution
t_values = np.linspace(0, 0.00015, 10000)  # Time vector from 0 to 1 ms

# Compute convolution at each time point
convolution_g_values = [convolution_g(t) for t in t_values]

# Compute the Discrete Fourier Transform (DFT) of the convolution output
dft_result = np.fft.fft(convolution_g_values)

# Compute the corresponding frequency bins
freq_bins = np.fft.fftfreq(len(convolution_g_values), d=(t_values[1] - t_values[0]))

# Filter out negative frequencies
positive_freqs = freq_bins[freq_bins >= 0]
positive_dft = np.abs(dft_result[freq_bins >= 0])

# Plot the convolution result
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t_values, convolution_g_values, label="Convolution $g(t)$", color='darkgreen')
plt.title('Convolution of $g_1(t) = 125000 e^{-64000 t} \\sin(86130 t)$ and $g_2(t) = 177187 e^{-8.85936 \\times 10^7 t}$')
plt.xlabel('Time (s)')
plt.ylabel('Convolution $g(t)$')
plt.legend()
plt.grid(True)

# Plot the DFT result with only positive frequencies
plt.subplot(2, 1, 2)
plt.plot(positive_freqs, positive_dft, label='DFT of Convolution', color='blue')
plt.title('Discrete Fourier Transform of the Convolution Output (Positive Frequencies)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.xlim(0, np.max(positive_freqs))  # Set x-limits to positive frequencies
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
