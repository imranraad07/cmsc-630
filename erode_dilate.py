import numpy as np
from PIL import Image

from utils import rgb_to_gray


def binarize(gray_image, pivot=127):
    copy_image = gray_image.copy()
    copy_image[copy_image > pivot] = 255
    copy_image[copy_image < pivot] = 0
    return copy_image


def erode_image(gray_image, erosion_level=3):
    copy_image = gray_image.copy()

    kernel = np.full(shape=(erosion_level, erosion_level), fill_value=255)
    image_src = binarize(copy_image)

    orig_shape = image_src.shape
    pad_width = erosion_level - 2

    image_pad = np.pad(array=image_src, pad_width=pad_width, mode='constant')
    pimg_shape = image_pad.shape
    h, w = (pimg_shape[0] - orig_shape[0]), (pimg_shape[1] - orig_shape[1])

    flat = np.array([image_pad[i:(i + erosion_level), j:(j + erosion_level)] for i in range(pimg_shape[0] - h) for j in
                     range(pimg_shape[1] - w)])

    out = np.array([255 if (i == kernel).all() else 0 for i in flat])
    out = out.reshape(orig_shape)
    out = out.astype(np.uint8)
    return out


def dilate_image(gray_image):
    copy_image = gray_image.copy()
    inverted_img = np.invert(copy_image)
    eroded_inverse = erode_image(inverted_img, 3).astype(np.uint8)
    out = np.invert(eroded_inverse)
    return out


# image_path = 'data/Cancerous cell smears/para11.BMP'
#
# img = Image.open(image_path)
# gray_image = rgb_to_gray(np.array(img), 'R')
# gray_image = gray_image.astype(np.uint8)
#
# eroded_image = erode_image(gray_image)
# eroded_image1 = Image.fromarray(eroded_image.astype(np.uint8))
# eroded_image1.show()
#
# inverted_img = np.invert(gray_image)
# eroded_inverse = erode_image(inverted_img, 3).astype(np.uint8)
# dilated_image = np.invert(eroded_inverse)
# dilated_image1 = Image.fromarray(dilated_image.astype(np.uint8))
# dilated_image1.show()
