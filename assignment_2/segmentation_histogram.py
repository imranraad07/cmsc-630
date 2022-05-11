import numpy as np
from PIL import Image

from utils import histogram, rgb_to_gray


# help: https://en.wikipedia.org/wiki/Balanced_histogram_thresholding

def bht(hist, min_count: int = 5):
    n_bins = len(hist)
    h_s = 0
    while hist[h_s] < min_count:
        h_s += 1
    h_e = n_bins - 1
    while hist[h_e] < min_count:
        h_e -= 1
    h_c = int(round(np.average(np.linspace(0, 2 ** 8 - 1, n_bins), weights=hist)))
    w_l = np.sum(hist[h_s:h_c])
    w_r = np.sum(hist[h_c: h_e + 1])

    while h_s < h_e:
        if w_l > w_r:
            w_l -= hist[h_s]
            h_s += 1
        else:
            w_r -= hist[h_e]
            h_e -= 1
        new_c = int(round((h_e + h_s) / 2))

        if new_c < h_c:
            w_l -= hist[h_c]
            w_r += hist[h_c]
        elif new_c > h_c:
            w_l += hist[h_c]
            w_r -= hist[h_c]

        h_c = new_c

    return h_c


def histogram_thresholding(gray_image):
    copy_image = gray_image.copy()
    hist = histogram(copy_image)
    pivot = bht(hist)
    copy_image[copy_image > pivot] = 255
    copy_image[copy_image < pivot] = 0
    out = copy_image.astype(np.uint8)
    return out.reshape(gray_image.shape)

# image_path = 'data/Cancerous cell smears/cyl01.BMP'
# img = Image.open(image_path)
# gray_image = rgb_to_gray(np.array(img), 'R')
# gray_image = gray_image.astype(np.uint8)
# segmented_thresholding = histogram_thresholding(gray_image)
# segmented_thresholding1 = Image.fromarray(segmented_thresholding.astype(np.uint8))
# segmented_thresholding1.show()
