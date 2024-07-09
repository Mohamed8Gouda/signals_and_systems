import numpy as np 
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import io, transform, color

# Read the image
A = io.imread('b&w.jpeg')

# Resize the image to the desired resolution (600, 800)
A_resized = transform.resize(A, (600, 800), anti_aliasing=True)

# Convert the resized image to grayscale
Abw = color.rgb2gray(A_resized)

# Add noise to the grayscale image
B = Abw + 0.5 * np.random.randn(600, 800)

# Perform FFT and shift for the original image
At = np.fft.fft2(Abw)
Ats = np.fft.fftshift(At)
A_spectrum = np.abs(Ats)

# Perform FFT and shift for the noisy image
Bt = np.fft.fft2(B)
Bts = np.fft.fftshift(Bt)
B_spectrum = np.abs(Bts)

# Create a low-pass filter
Fs = np.zeros((600, 800))
width = 50
mask = np.ones((2*width+1, 2*width+1))
Fs[300-width:300+width+1, 400-width:400+width+1] = mask

# Apply the filter to the noisy image in the frequency domain
btsf = np.multiply(Bts, Fs)
btf = np.fft.ifftshift(btsf)
bf = np.fft.ifft2(btf)
bf = np.abs(bf)  # Take the absolute value to get the real part of the image

# Perform FFT and shift for the filtered image
Bf_fft = np.fft.fft2(bf)
Bf_fft_shifted = np.fft.fftshift(Bf_fft)
Bf_spectrum = np.abs(Bf_fft_shifted)

# Plotting
plt.figure(figsize=(15, 10))

# Original Image 
plt.subplot(2, 3, 1)
plt.title("Original Image")
plt.imshow(Abw, cmap='gray')

# Filtered Image
plt.subplot(2, 3, 3)
plt.title("Filtered Image")
plt.imshow(bf, cmap='gray')

# Noisy Image and its FFT
plt.subplot(2, 3, 2)
plt.title("Noisy Image")
plt.imshow(B, cmap='gray')

plt.subplot(2, 3, 4)
plt.title("FFT of Noisy Image")
plt.imshow(np.log(1 + B_spectrum), cmap='hot')

# Original Image FFT
plt.subplot(2, 3, 5)
plt.title("FFT of Original Image")
plt.imshow(np.log(1 + A_spectrum), cmap='hot')

# FFT of Filtered Image
plt.subplot(2, 3, 6)
plt.title("FFT of Filtered Image")
plt.imshow(np.log(1 + Bf_spectrum), cmap='hot')

# Show all plots
plt.tight_layout()
plt.show()