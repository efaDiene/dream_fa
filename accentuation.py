import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from scipy.signal import butter,sosfiltfilt
from numpy.fft import fft2, ifft2
from skimage import img_as_float, io, img_as_ubyte


def sharpen_image(image, alpha=1):
    # Appliquer un flou gaussien avec skimage
    blurred_image_skimage = gaussian(image, sigma=5)

    
    # Calculer l'image renforcée
    sharpened_image_skimage = (image - blurred_image_skimage)

    # Calculer l'image accentuée
    accent_image= image + (alpha * sharpened_image_skimage)
    
    return sharpened_image_skimage, accent_image

def gaussian_kernel(kernel_size, sigma=1.):
    if isinstance(kernel_size, (int, float)):
        kernel_size = [kernel_size, kernel_size]
    x, y = np.mgrid[-kernel_size[0]//2 + 1:kernel_size[0]//2 + 1,
                    -kernel_size[1]//2 + 1:kernel_size[1]//2 + 1]
    g = (1 / (2 * np.pi * sigma**2)) * np.exp(-((x**2 + y**2)/(2.0 * sigma**2)))
    g_normalise = g / np.sum(g)
    return g_normalise

def lowpass(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 5):
    sos = butter(poles, cutoff, 'lowpass', fs=sample_rate, output='sos')
    filtered_data = sosfiltfilt(sos, data)
    return filtered_data

def apply_frequency_filter(image,image2,cutoff_low1=12,cutoff_high1=12):
    r, c = image.shape
    r2, c2 = image2.shape 
    hs_gauss = 30
    cutoff_low = 1 / (2 * np.pi * cutoff_low1)
    cutoff_high = 1 / (2 * np.pi * cutoff_high1)

    #definir la gaussienne
    gaussienne = gaussian_kernel([cutoff_low1*2+1, cutoff_low1*2+1], 1 / ( np.pi * cutoff_low))
    gaussienne2 = gaussian_kernel([cutoff_high1*2+1, cutoff_high1*2+1], 1 / (2 * np.pi * cutoff_high))


    # Appliquer la FFT2 à l'image et à la gaussienne
    image_fft = fft2(image)
    image2_fft = fft2(image2)
    gaussienne_fft = fft2(gaussienne, [r, c])
    gaussienne2_fft = fft2(gaussienne2, [r2, c2])

    # Centrer la FFT et appliquer np.abs 
    # Juste pour visualiser
    image_fft_shifted = np.fft.fftshift(np.log(np.abs(image_fft)))
    image2_fft_shifted = np.fft.fftshift(np.log(np.abs(image2_fft)))
    gaussienne_fft_shifted = np.fft.fftshift(np.abs(gaussienne_fft))


    # Appliquer le filtre fréquentiel 1
    filtered_image_fft_shifted = image_fft * gaussienne_fft
    filtered_image2_fft_shifted = image2_fft * gaussienne2_fft
    filtered_image2_fft_shifted = image2_fft - filtered_image2_fft_shifted


    # Calcul les basses fréquences
    freq = np.fft.fftfreq(len(image_fft), 5)
    freq2 = np.fft.fftfreq(len(image2_fft), 5)

    #couper les hautes frequences
    filtered_image_fft_shifted[freq >=0] = 0
    filtered_image2_fft_shifted[freq2 <0] = 0
    
    # Inverser le centrage
    filtered_image_fft = np.fft.ifftshift(np.exp(image_fft_shifted))
    filtered_image2_fft = np.fft.ifftshift(np.exp(image2_fft_shifted))

    # Appliquer l'inverse de la FFT2 pour obtenir l'image filtrée
    filtered_image = np.real(ifft2(filtered_image_fft_shifted))
    filtered_image2 = np.real(ifft2(filtered_image2_fft_shifted))
    filtered_image2=sharpen_image(filtered_image2)

    
    # enlever le "padding"
    filtered_image = filtered_image[hs_gauss:image.shape[0] + hs_gauss, hs_gauss:image.shape[1] + hs_gauss] 
    filtered_image2 = filtered_image2[hs_gauss:image2.shape[0] + hs_gauss, hs_gauss:image2.shape[1] + hs_gauss] 


    return filtered_image, filtered_image2, freq2

def apply_frequency_filter2(image, gausienne):
    # Appliquer la FFT2 à l'image
    image_fft = fft2(image)
    gausienne_fft = fft2(gausienne)

    # Centrer la FFT
    image_fft_shifted = np.log(np.abs(np.fft.fftshift(image_fft)))
    gausienne_fft_shifted = np.log(np.abs(np.fft.fftshift(gausienne_fft)))


    # Appliquer le filtre fréquentiel
    filtered_image_fft_shifted = image_fft_shifted * gausienne_fft_shifted

    # Inverser le centrage
    filtered_image_fft = np.log(np.abs(np.fft.ifftshift(filtered_image_fft_shifted)))

    # Appliquer l'inverse de la FFT2 pour obtenir l'image filtrée
    filtered_image = np.abs(ifft2(filtered_image_fft))

    return filtered_image


# Charger les images
image = io.imread('./data/image.png')
image= img_as_float(image)
#image2 = io.imread('./images/Marilyn_Monroe.png', pilmode='L')
#image2= img_as_float(image2)



# Paramètre d'accentuation (ajustez selon vos besoins)
alpha = 0.6

# Appliquer le renforcement des contours
sharpened_result,im_accent = sharpen_image(image, alpha)
#imageResult, img2, gaussienne = apply_frequency_filter(image,image2)
# conversion en ubyte
#image = img_as_ubyte(image)
im_accent = img_as_ubyte(np.clip(im_accent, 0, 1))


# sauvegarder l'image
fname = './data/image2.png'
io.imsave(fname, im_accent)


# afficher l'image
io.imshow(im_accent)
io.show()


# Afficher les images
#plt.subplot(1, 3, 1), plt.imshow(image), plt.title('Image originale')
#plt.subplot(1, 3, 2), plt.imshow(im_accent), plt.title('Image accentuée')
#plt.subplot(1, 3, 3), plt.imshow(gaussienne), plt.title('haute fréquences')
#plt.show()