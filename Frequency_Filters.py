import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as img
from skimage.color import rgb2gray
from skimage import io ,transform,color
from numpy.fft import fft, fftshift, ifft

# Read the image
A= io.imread('b&w.jpeg');

# Resize the image to the desired resolution (600, 800)
A_resized = transform.resize(A, (600, 800), anti_aliasing=True)

# Convert the resized image to grayscal
Abw = color.rgb2gray(A_resized)

# Add noise to the grayscale image
B=Abw+0.5*np.random.randn(600,800)



# Perform FFT and shift for the original image
At = np.fft.fft2(Abw)
Ats = np.fft.fftshift(At)
A_spectrum = np.abs(Ats)

# Perform FFT and shift for the noisy image
Bt = np.fft.fft2(B)
Bts = np.fft.fftshift(Bt)
B_spectrum = np.abs(Bts)

# Original Image and its FFT
plt.subplot(2, 2, 1)
plt.title("Original Image")
plt.imshow(Abw, cmap='hot')

plt.subplot(2, 2, 2)
plt.title("FFT of the Original Image")
plt.imshow(np.log(1 + A_spectrum), cmap='hot')

# Noisy Image and its FFT
plt.subplot(2, 2, 3)
plt.title("Noisy Image")
plt.imshow(B, cmap='hot')

plt.subplot(2, 2, 4)
plt.title("FFT of the noisy Image")
plt.imshow(np.log(1 + B_spectrum), cmap='hot')

plt.tight_layout()
plt.show()