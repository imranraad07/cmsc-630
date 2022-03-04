# importing PIL
from PIL import Image
import numpy as np


def rgb_to_gray(img):
    grayImage = np.zeros(img.shape)
    R = np.array(img[:, :, 0])
    G = np.array(img[:, :, 1])
    B = np.array(img[:, :, 2])

    R = (R * .299)
    G = (G * .587)
    B = (B * .114)

    Avg = (R + G + B)
    grayImage = img.copy()

    for i in range(3):
        grayImage[:, :, i] = Avg

    return grayImage


if __name__ == "__main__":
    img = Image.open('data/Cancerous cell smears/cyl01.BMP')
    # Output Images
    img.show()
    # prints format of image
    print(img.format)
    # prints mode of image
    print(img.mode)

    img_arr = np.array(img)
    gray_image = rgb_to_gray(img_arr)
    print(gray_image)

    gray_image = Image.fromarray(gray_image)
    gray_image.show()
