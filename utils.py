import time

import numpy as np


def rgb_to_gray(img):
    copy_image = img.copy()

    R = np.array(copy_image[:, :, 0])
    G = np.array(copy_image[:, :, 1])
    B = np.array(copy_image[:, :, 2])

    R = (R * .299)
    G = (G * .587)
    B = (B * .114)

    Avg = (R + G + B)
    return Avg


def histogram(img):
    hist = np.zeros(256)
    for i in range(len(hist)):
        hist[i] = np.sum(img == i)
    hist = hist.astype(int)
    return hist


def gaussian_noise(img_array, sigma):
    copy_image = img_array.copy()

    mean = 0.0
    noise = np.random.normal(mean, sigma, copy_image.size)
    shaped_noise = noise.reshape(copy_image.shape)
    gauss = copy_image + shaped_noise
    return gauss


def salt_pepper_noise(img, strength):
    copy_image = img.copy()

    out = np.copy(copy_image)
    m, n = copy_image.shape

    num_salt = np.ceil(strength * copy_image.size * 0.5)
    for i in range(int(num_salt)):
        x = np.random.randint(0, m - 1)
        y = np.random.randint(0, n - 1)
        out[x][y] = 0

    num_pepper = np.ceil(strength * copy_image.size * 0.5)
    for i in range(int(num_pepper)):
        x = np.random.randint(0, m - 1)
        y = np.random.randint(0, n - 1)
        out[x][y] = 0

    return out


def equalized_histogram(img):
    copy_image = img.copy()

    img_hist = histogram(copy_image)
    cum_sum = np.cumsum(img_hist)
    cum_sum = (cum_sum - cum_sum.min()) * 255 / (cum_sum.max() - cum_sum.min())
    cum_sum = cum_sum.astype(np.uint8)
    equalized = cum_sum[copy_image.flatten().astype(np.uint8)]

    img_new = np.reshape(equalized, copy_image.shape)
    return img_hist, histogram(img_new), img_new


def mean_square_error(original_img, quantized_img):
    mse = (np.square(original_img - quantized_img)).mean()
    return mse
