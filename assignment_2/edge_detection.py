import math

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from utils import rgb_to_gray

neighbours_x = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
neighbours_y = [-1, 0, 1, -1, 0, 1, -1, 0, 1]
PI = 180


def convolution(image, kernel, average=False):
    copy_image = image.copy()
    r, c = copy_image.shape
    kernel_row, kernel_col = kernel.shape

    out = np.zeros(copy_image.shape)

    pad_h = int((kernel_row - 1) / 2)
    pad_w = int((kernel_col - 1) / 2)

    padded_image = np.zeros((r + (2 * pad_h), c + (2 * pad_w)))

    padded_image[pad_h:padded_image.shape[0] - pad_h, pad_w:padded_image.shape[1] - pad_w] = copy_image

    for row in range(r):
        for col in range(c):
            out[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            if average:
                out[row, col] /= kernel.shape[0] * kernel.shape[1]

    return out


def non_max_suppression(gradient_magnitude, gradient_direction):
    r, c = gradient_magnitude.shape
    output = np.zeros(gradient_magnitude.shape)

    for row in range(1, r - 1):
        for col in range(1, c - 1):
            direction = gradient_direction[row, col]
            if (0 <= direction < PI / 8) or (15 * PI / 8 <= direction <= 2 * PI):
                before_pixel = gradient_magnitude[row, col - 1]
                after_pixel = gradient_magnitude[row, col + 1]

            elif (PI / 8 <= direction < 3 * PI / 8) or (9 * PI / 8 <= direction < 11 * PI / 8):
                before_pixel = gradient_magnitude[row + 1, col - 1]
                after_pixel = gradient_magnitude[row - 1, col + 1]

            elif (3 * PI / 8 <= direction < 5 * PI / 8) or (11 * PI / 8 <= direction < 13 * PI / 8):
                before_pixel = gradient_magnitude[row - 1, col]
                after_pixel = gradient_magnitude[row + 1, col]

            else:
                before_pixel = gradient_magnitude[row - 1, col - 1]
                after_pixel = gradient_magnitude[row + 1, col + 1]

            if gradient_magnitude[row, col] >= before_pixel and gradient_magnitude[row, col] >= after_pixel:
                output[row, col] = gradient_magnitude[row, col]

    return output


def threshold(image, low, high, weak):
    output = np.zeros(image.shape)
    strong = 255
    strong_row, strong_col = np.where(image >= high)
    weak_row, weak_col = np.where((image <= high) & (image >= low))
    output[strong_row, strong_col] = strong
    output[weak_row, weak_col] = weak
    return output


def check_neighbours(arr, r, c):
    for i in range(3):
        for j in range(3):
            if neighbours_x[i] == 0 and neighbours_y[j] == 0:
                continue
            if arr[r + neighbours_x[i]][c + neighbours_y[j]] == 255:
                return 255
    return 0


def hysteresis(image, weak):
    copy_image = image.copy()
    r, c = copy_image.shape

    top_to_bottom = copy_image.copy()
    for row in range(1, r):
        for col in range(1, c):
            if top_to_bottom[row, col] == weak:
                top_to_bottom[row, col] = check_neighbours(top_to_bottom, row, col)

    bottom_to_top = copy_image.copy()

    for row in range(r - 1, 0, -1):
        for col in range(c - 1, 0, -1):
            if bottom_to_top[row, col] == weak:
                bottom_to_top[row, col] = check_neighbours(bottom_to_top, row, col)

    right_to_left = copy_image.copy()

    for row in range(1, r):
        for col in range(c - 1, 0, -1):
            if right_to_left[row, col] == weak:
                right_to_left[row, col] = check_neighbours(right_to_left, row, col)

    left_to_right = copy_image.copy()

    for row in range(r - 1, 0, -1):
        for col in range(1, c):
            if left_to_right[row, col] == weak:
                left_to_right[row, col] = check_neighbours(left_to_right, row, col)
    out = top_to_bottom + bottom_to_top + right_to_left + left_to_right
    out[out > 255] = 255
    return out


def sobel_edge_detection(image, edge_filter):
    copy_image = image.copy()
    new_image_x = convolution(copy_image, edge_filter)
    new_image_y = convolution(copy_image, np.flip(edge_filter.T, axis=0))
    gradient_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))
    gradient_magnitude *= 255.0 / gradient_magnitude.max()
    gradient_direction = np.arctan2(new_image_y, new_image_x)
    gradient_direction = np.rad2deg(gradient_direction)
    gradient_direction += PI
    return gradient_magnitude, gradient_direction


def normalize(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)


def gaussian_kernel(size, sigma=1):
    kernel_1D = np.linspace(- int(size / 2), int(size / 2), size)
    for i in range(size):
        kernel_1D[i] = normalize(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)

    kernel_2D *= 1.0 / kernel_2D.max()

    return kernel_2D


def gaussian_blur(image, kernel_size):
    kernel = gaussian_kernel(kernel_size, sigma=math.sqrt(kernel_size))
    return convolution(image, kernel, average=True)


def edge_detect(gray_image, edge_filter):
    copy_image = gray_image.copy()
    gaussian_blurred_image = gaussian_blur(copy_image, len(edge_filter) * len(edge_filter[0]))
    gradient_magnitude, gradient_direction = sobel_edge_detection(gaussian_blurred_image, edge_filter)
    out = non_max_suppression(gradient_magnitude, gradient_direction)
    out = threshold(out, 5, 20, 100)
    out = hysteresis(out, 100)
    out = out.astype(np.uint8)
    return out

# image_path = 'data/Cancerous cell smears/para11.BMP'
#
# img = Image.open(image_path)
# gray_image = rgb_to_gray(np.array(img), 'R')
# gray_image = gray_image.astype(np.uint8)
# edge_image = edge_detect(gray_image, np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]))
# edge_image1 = Image.fromarray(edge_image.astype(np.uint8))
# edge_image1.show()
