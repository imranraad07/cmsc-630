# importing PIL
import random

from PIL import Image, ImageFilter
import numpy as np
from utils import rgb_to_gray, salt_pepper_noise, histogram, gaussian_noise, equalized_histogram

if __name__ == "__main__":
    img = Image.open('data/Cancerous cell smears/cyl01.BMP')
    # Output Images
    # img.show()
    # print(img.format)
    # print(img.mode)

    # convert to gray image
    print("Gray Image...")
    gray_image = rgb_to_gray(np.array(img))
    # gray_image1 = Image.fromarray(gray_image)
    # gray_image1.show()
    print("Done.")

    # salt n pepper noise
    print("Salt and Pepper Noise...")
    noised_image = salt_pepper_noise(gray_image, 0.1)
    # noised_image1 = Image.fromarray(noised_image)
    # noised_image1.show()
    print("Done.")

    # gaussian noise
    print("Gaussian Noise...")
    noised_image_2 = gaussian_noise(gray_image, 0.1)
    # noised_image_21 = Image.fromarray(noised_image_2)
    # noised_image_21.show()
    print("Done.")

    # # histogram
    # print("Histogram...")
    # histogram = histogram(gray_image)
    # print(histogram)
    # print("Done.")

    print("Equalization Histogram...")
    histogram_org, histogram_eq, img_eq = equalized_histogram(gray_image)
    # print(img_eq.shape)
    # img_eq1 = Image.fromarray(img_eq)
    # img_eq1.show()
    # print(histogram_org)
    # print(histogram_eq)
    print("Done.")
