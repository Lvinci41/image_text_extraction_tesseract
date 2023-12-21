import cv2 as cv
import numpy as np
from PIL import Image
import tensorflow as tf 
#from libsvm import svmutil
import math
from scipy.stats import beta
from numpy.linalg import norm

import os
import sys 
from libsvm import svmutil
import svmutil
import matplotlib.pyplot as plt
from numpy.linalg import norm

#DECLARATIONS
filename = 'IMG_4303 - Copy_val.png'
img = cv.imread(filename)
img1 = cv.imread(filename, 0)
im = Image.open(filename).convert('L') # to grayscale
im1 = Image.open(filename)
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)

d = dict()

RED_SENSITIVITY = 0.299
GREEN_SENSITIVITY = 0.587
BLUE_SENSITIVITY = 0.114
#/END DECLARATIONS

"""
TONE MAPPING
"""

def convert_to_brightness_image(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        raise ValueError("uint8 is not a good dtype for the image")

    return np.sqrt(
        image[..., 0] ** 2 * RED_SENSITIVITY
        + image[..., 1] ** 2 * GREEN_SENSITIVITY
        + image[..., 2] ** 2 * BLUE_SENSITIVITY
    )

def get_resolution(image: np.ndarray):
    height, width = image.shape[:2]
    return height * width

def brightness_histogram(image: np.ndarray) -> np.ndarray:
    nr_of_pixels = get_resolution(image)
    brightness_image = convert_to_brightness_image(image)
    hist, _ = np.histogram(brightness_image, bins=256, range=(0, 255))
    return hist / nr_of_pixels

def distribution_pmf(dist: any, start: float, stop: float, nr_of_steps: int):
    xs = np.linspace(start, stop, nr_of_steps)
    ys = dist.pdf(xs)
    return ys / np.sum(ys)

def correlation_distance(distribution_a: np.ndarray, distribution_b: np.ndarray) -> float:
    dot_product = np.dot(distribution_a, distribution_b)
    squared_dist_a = np.sum(distribution_a ** 2)
    squared_dist_b = np.sum(distribution_b ** 2)
    return dot_product / math.sqrt(squared_dist_a * squared_dist_b)

def compute_hdr(cv_image: np.ndarray):
    img_brightness_pmf = brightness_histogram(np.float32(cv_image))
    ref_pmf = distribution_pmf(beta(2, 2), 0, 1, 256)
    return correlation_distance(ref_pmf, img_brightness_pmf)

d['toneMapping'] = round( compute_hdr(img), 4)

"""
/END TONE MAPPING
FFT
"""

def detect_blur_fft(image, size=60, thresh=10, vis=True):
    (h,w) = image.shape
    (cX, cY) = (int(w/2.0), int(h/2.0))
    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)
    if vis:
        magnitude = 20 * np.log(np.abs(fftShift))
        (fig, ax) = plt.subplots(1, 2, )
        ax[0].imshow(image, cmap="gray")
        ax[0].set_title("Input")
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[1].imshow(magnitude, cmap="gray")
        ax[1].set_title("Magnitude Spectrum")
        ax[1].set_xticks([])
        ax[1].set_yticks([])

    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)

    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)
    return (mean, mean <= thresh)

fft_score, fft_thresh = detect_blur_fft(gray)
d['FFT'] = round( fft_score, 4 ) #these two can be one line

"""
/END FFT
SHARPNESS-LAPLACIAN
"""

laplacian = cv.Laplacian(img,cv.CV_64F)
gnorm_lp = np.sqrt(laplacian**2)
d['sharpness_lp'] = round( np.average(gnorm_lp), 4 )


"""
/END SHARPNESS-LAPLACIAN
SHARPNESS-AVGGRADIENTMAGNITUDE
"""

array = np.asarray(im, dtype=np.int32)
gy, gx = np.gradient(array)
gnorm_agm = np.sqrt(gx**2 + gy**2)
d['sharpness_agm'] = round( np.average(gnorm_agm), 4 )

"""
/END SHARPNESS-AVGGRADIENTMAGNITUDE
SHARPNESS-AVGGRADIENTMAGNITUDEDX
"""
dx = np.diff(im)[1:,:] # remove the first row
dy = np.diff(im, axis=0)[:,1:] # remove the first column
dnorm = np.sqrt(dx**2 + dy**2)
d['sharpness_dx'] = np.average(dnorm)

"""
/END SHARPNESS-AVGGRADIENTMAGNITUDEDX
BRISQUE
"""

d['brisque'] = int( cv.quality.QualityBRISQUE_compute( img, "brisque_model_live.yml", "brisque_range_live.yml")[0])

"""
/END BRISQUE
BRIGHTNESS1
"""

def brightness(img):
    if len(img.shape) == 3:
        # Colored RGB or BGR (*Do Not* use HSV images with this function)
        # create brightness with euclidean norm
        return np.average(norm(img, axis=2)) / np.sqrt(3)
    else:
        # Grayscale
        return np.average(img)
    
d['brightness1'] = int( brightness(img) )

"""
/END BRIGHTNESS1
BRIGHTNESS2
"""

def isbright(image, dim=10, thresh=0.5):
    # Resize image to 10x10
    image = cv.resize(image, (dim, dim))
    # Convert color space to LAB format and extract L channel
    L, A, B = cv.split(cv.cvtColor(image, cv.COLOR_BGR2LAB))
    # Normalize L channel by dividing all pixel values with maximum pixel value
    L = L/np.max(L)
    # Return True if mean is greater than thresh else False
    return np.mean(L)

d['brightness2'] = round(isbright(img), 4)

"""
/END BRIGHTNESS3
VARIANCE
"""

d['variance'] = round(np.var(img) , 2)

"""
/END VARIANCE
LAPLACIAN VARIANCE
"""

d['lp_variance'] = round(cv.Laplacian(img, cv.CV_64F).var(), 4)

"""
/END LAPLACIAN VARIANCE
GRAY VARIANCE
"""

d['variance_gray'] = round(np.var(gray), 2)

"""
/END GRAY VARIANCE
GRAY LAPLACIAN VARIANCE
"""

d['lp_variance_gray'] = round(cv.Laplacian(gray, cv.CV_64F).var(), 2)

"""
/END GRAY LAPLACIAN VARIANCE
BACKGROUND/MODE COLOR (only works for JPG)
"""
im_size = im1.size[0]*im1.size[1]

#DOWNSAMPLE
allColors = im1.getcolors(im_size)
downsampleColors = []
dsColorsWithCount = []
colorCounts = [i[0] for i in allColors]
for color in allColors:
    colorVec = color[1]
    newVec = []
    for coordinate in colorVec:
        newVec.append( int(coordinate/17) )
        newEntry = [color[0], tuple(newVec) ]
    downsampleColors.append( tuple(newVec) )
    dsColorsWithCount.append( tuple( newEntry ) )

dsColorsAndCounts = []
dsColors_arr = set( downsampleColors )
for color in dsColors_arr:
    count = 0
    for entry in dsColorsWithCount:
        if entry[1] == color:
            count += entry[0]
    newEntry = [count, color ] 
    dsColorsAndCounts.append( tuple( newEntry ) )
#/END DOWNSAMPLE


all_colors = sorted(dsColorsAndCounts, reverse=True, key=lambda tup: tup[0])
color1 = all_colors[0]
color2 = all_colors[1]
color3 = all_colors[2]

d['color1_vector'] = color1[1]
d['color2_vector'] = color2[1]
d['color3_vector'] = color3[1]

d['color1_maxCoordinate'] = color1[1].index(max( color1[1] ))
d['color2_maxCoordinate'] = color2[1].index(max( color2[1] ))
d['color3_maxCoordinate'] = color3[1].index(max( color3[1] ))

d['color1_strength'] = max( color1[1] )
d['color2_strength'] = max( color2[1] )
d['color3_strength'] = max( color3[1] )

d['color1_dominance'] = round( color1[0]/im_size , 4)
d['color2_dominance'] = round( color2[0]/im_size , 4)
d['color3_dominance'] = round( color3[0]/im_size , 4)

d['color1_dupCoordinate'] = color1[1].count( max( color1[1]))
d['color2_dupCoordinate'] = color2[1].count( max( color2[1]))
d['color3_dupCoordinate'] = color3[1].count( max( color2[1]))

d['color1_isGray'] = color1[1][0]==color1[1][1]==color1[1][2]
d['color2_isGray'] = color2[1][0]==color2[1][1]==color2[1][2]
d['color3_isGray'] = color3[1][0]==color3[1][1]==color3[1][2]

"""
/END BACKGROUND/MODE COLOR

"""