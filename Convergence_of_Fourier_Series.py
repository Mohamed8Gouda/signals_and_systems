import numpy as np 
import matplotlib.pyplot as plt
import scipy.linalg
from numpy.fft import fft, fftshift, ifft

# Number of data points
n = 200 

# Create a linearly spaced array of n points from 0 to 2*pi
x = np.linspace(0, 2 * np.pi, n)

# Four generic sinusoidal signals with different frequencies and offsets
s1 = np.sin(1 * x)        # Sinusoid with frequency 1x
s2 = np.sin(2 * x) + 2.5  # Sinusoid with frequency 2x and offset 2.5
s3 = np.sin(3 * x) + 5    # Sinusoid with frequency 3x and offset 5
s4 = np.sin(4 * x) + 7.4  # Sinusoid with frequency 4x and offset 7.4

# Gaussian function centered around pi+1
f = np.exp(-2 * np.square(x - (np.pi + 1)))

# Dot product of the Gaussian function with each sinusoidal signal
# and scaling the sinusoidal signals by these dot products
a1 = np.dot(f, s1); f1 = a1 * s1
a2 = np.dot(f, s2); f2 = a1 * s1 + a2 * s2
a3 = np.dot(f, s3); f3 = a1 * s1 + a2 * s2 + a3 * s3
a4 = np.dot(f, s4); f4 = a1 * s1 + a2 * s2 + a3 * s3 + a4 * s4

# Initialize a figure for plotting
fig = plt.figure()

# Initialize the approximation with zeros
approx = 0 * x

# Perform the dot product with more sine functions and plot
for harmonic_index  in range(9):
    ax = fig.add_subplot(3, 3, harmonic_index  + 1)  # Create a subplot in a 4x3 grid
    s = np.sin((harmonic_index  + 1) * x /2 )        # Generate sine wave with increasing frequency
    s = s / np.sqrt(np.dot(s, s))      # Normalize the sine wave
    a = np.dot(f, s)                   # Dot product with the Gaussian function
    approx = approx + a * s            # Update the approximation
    plt.plot(x, f, x, approx)          # Plot the Gaussian function and the approximation

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the plot
plt.show()
