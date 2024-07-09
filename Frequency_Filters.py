import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as img
from skimage.color import rgb2gray
from skimage import io ,transform,color
from numpy.fft import fft, fftshift, ifft

# Read the image
A= io.imread('photo.jpeg');

# Resize the image to the desired resolution (600, 800)
A_resized = transform.resize(A, (600, 800), anti_aliasing=True)

# Convert the resized image to grayscal
Abw = color.rgb2gray(A_resized)

# Add noise to the grayscale image
B=Abw+0.5*np.random.randn(600,800)

# Perform FFT and shift
Bt=np.fft.fft2(B)
Bts=fftshift(Bt)
# Calculate magnitude spectrum
B_spectrum = np.abs(Bts)

# Display the original and noisy images with hot colormap
plt.subplot(1, 2, 1)
plt.title("The orginal Image")
plt.imshow(Abw, cmap='hot')

plt.subplot(1, 2, 2)
plt.title("The noisy Image")
plt.imshow(B, cmap='hot')

plt.show()
