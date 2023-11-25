import cv2 as cv
import numpy as np
from PIL import Image
import tensorflow as tf 
from libsvm import svmutil
import math
from scipy.stats import beta

img = cv.imread('dave.jpg',0)
im = Image.open(filename).convert('L') # to grayscale
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#BRIGHTNESS
#https://towardsdatascience.com/measuring-enhancing-image-quality-attributes-234b0f250e10
def pixel_brightness(pixel):
    assert 3 == len(pixel)
    r, g, b = pixel
    return math.sqrt(0.299 * r ** 2 + 0.587 * g ** 2 + 0.114 * b ** 2)
    
def image_brightness(img):
    nr_of_pixels = len(img) * len(img[0])
    return sum(pixel_brightness(pixel) for pixel in row for row in img) / nr_of_pixels


#TONE MAPPING
#https://towardsdatascience.com/measuring-enhancing-image-quality-attributes-234b0f250e10
RED_SENSITIVITY = 0.299
GREEN_SENSITIVITY = 0.587
BLUE_SENSITIVITY = 0.114

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

def distribution_pmf(dist: Any, start: float, stop: float, nr_of_steps: int):
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

#FFT
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

fft_score, fft_thresh = detect_blur_fft(img)

#LAPLACIAN SHARPNESS
laplacian = cv.Laplacian(img,cv.CV_64F)
gnorm = np.sqrt(laplacian**2)
sharpness = np.average(gnorm)

#AVERAGE GRADIENT MAGNITUDE
array = np.asarray(im, dtype=np.int32)
gy, gx = np.gradient(array)
gnorm = np.sqrt(gx**2 + gy**2)
sharpness = np.average(gnorm)

#AVERAGE GRADIENT MAGNITUDE DERIVATIVE
dx = np.diff(array)[1:,:] # remove the first row
dy = np.diff(array, axis=0)[:,1:] # remove the first column
dnorm = np.sqrt(dx**2 + dy**2)
sharpness = np.average(dnorm)

#TOTAL VARIATION
tf.image.total_variation(img, name=None) 

#BRISQUE 
blurred = cv.GaussianBlur(img, (7, 7), 1.166) 
blurred_sq = blurred * blurred
sigma = cv.GaussianBlur(img * img, (7, 7), 1.166)
sigma = (sigma - blurred_sq) ** 0.5
sigma = sigma + 1.0/255 
structdis = (img - blurred)/sigma

M = np.float32([[1, 0, reqshift[1]], [0, 1, reqshift[0]]])
ShiftArr = cv.warpAffine(OrigArr, M, (structdis.shape[1], structdis.shape[0]))

model = svmutil.svm_load_model("allmodel")
x, idx = gen_svm_nodearray(x[1:], isKernel=(model.param.kernel_type == PRECOMPUTED))
nr_classifier = 1 # fixed for svm type as EPSILON_SVR (regression)
prob_estimates = (c_double * nr_classifier)()
qualityscore = svmutil.libsvm.svm_predict_probability(model, x, dec_values)

