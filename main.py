# importing PIL
import time
from multiprocessing import Process
from pathlib import Path

from PIL import Image
import numpy as np

from utils import rgb_to_gray, salt_pepper_noise, gaussian_noise, equalized_histogram, mean_square_error


def test_utils(image_path):
    img = Image.open(image_path)
    # Output Images
    img.show()
    # print(img.format)
    # print(img.mode)

    # convert to gray image
    print("Gray Image...")
    gray_image = rgb_to_gray(np.array(img))
    gray_image1 = Image.fromarray(gray_image)
    gray_image1.show()
    print("Done.")

    # salt n pepper noise
    print("Salt and Pepper Noise...")
    noised_image = salt_pepper_noise(gray_image, 0.1)
    noised_image1 = Image.fromarray(noised_image)
    noised_image1.show()
    print("Done.")

    # gaussian noise
    print("Gaussian Noise...")
    noised_image_2 = gaussian_noise(gray_image, 10)
    noised_image_21 = Image.fromarray(noised_image_2)
    noised_image_21.show()
    print("Done.")

    # # histogram
    # print("Histogram...")
    # histogram = histogram(gray_image)
    # print(histogram)
    # print("Done.")

    print("Equalization Histogram...")
    histogram_org, histogram_eq, img_eq = equalized_histogram(gray_image)
    # print(img_eq.shape)
    img_eq1 = Image.fromarray(img_eq)
    img_eq1.show()
    # print(histogram_org)
    # print(histogram_eq)
    print("Done.")

    print("MSQE...")
    msqe = mean_square_error(gray_image, img_eq)
    print(msqe)
    print("Done.")


def operations(image_path):
    t0 = time.time()
    print("Started..", image_path)
    img = Image.open(image_path)
    gray_image = rgb_to_gray(np.array(img))
    noised_image = salt_pepper_noise(gray_image, 0.1)
    noised_image_2 = gaussian_noise(gray_image, 10)
    histogram_org, histogram_eq, img_eq = equalized_histogram(gray_image)
    msqe = mean_square_error(gray_image, img_eq)
    print("Ended..")
    t1 = time.time()
    print("Time Taken", t1 - t0, image_path)


if __name__ == "__main__":
    # test_utils('data/Cancerous cell smears/cyl01.BMP')
    base_path = Path('data/Cancerous cell smears')
    files = list(base_path.glob("*.BMP"))
    print(len(files))
    proc = []
    counter = 0
    for file in files:
        counter = counter + 1
        p = Process(target=operations, args=[file])
        p.start()
        proc.append(p)
    for p in proc:
        p.join()
