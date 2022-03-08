# importing PIL
import random

from PIL import Image, ImageFilter
import numpy as np
from utils import rgb_to_gray, salt_pepper_noise


# function to obtain histogram of an image
def hist_plot(img):
    m, n = img.shape
    count = []
    r = []
    for k in range(0, 256):
        r.append(k)
        count1 = 0
        for i in range(m):
            for j in range(n):
                if img[i, j] == k:
                    count1 += 1
        count.append(count1)
    return (r, count)




def gaussian_noise(img_array, sigma):
    mean = 0.0
    noise = np.random.normal(mean, sigma, img_array.size)
    shaped_noise = noise.reshape(img_array.shape)
    gauss = img_array + shaped_noise
    return gauss


if __name__ == "__main__":
    img = Image.open('data/Cancerous cell smears/cyl01.BMP')
    # Output Images
    img.show()
    print(img.format)
    print(img.mode)

    # convert to gray image
    gray_image = rgb_to_gray(np.array(img))
    gray_image1 = Image.fromarray(gray_image)
    gray_image1.show()

    # salt n pepper noise
    noised_image = salt_pepper_noise(gray_image, 0.1)
    noised_image1 = Image.fromarray(noised_image)
    noised_image1.show()

    # gaussian noise
    noised_image_2 = gaussian_noise(gray_image, 0.1)
    noised_image_21 = Image.fromarray(noised_image_2)
    noised_image_21.show()

    # histogram
    r1, count = hist_plot(gray_image)
    print(r1)
    print(count)
