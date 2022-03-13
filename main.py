# importing PIL
import time
from pathlib import Path

import numpy as np
from PIL import Image
from PIL.ImageFile import ImageFile

import tqdm_batch

ImageFile.LOAD_TRUNCATED_IMAGES = True

from utils import rgb_to_gray, salt_pepper_noise, gaussian_noise, equalized_histogram, mean_square_error, linear_filter, \
  median_filter


def test_utils(image_path):
  img = Image.open(image_path)
  # Output Images
  img.show()
  # print(img.format)
  # print(img.mode)

  # convert to gray image
  print("Gray Image...")
  gray_image = rgb_to_gray(np.array(img), 'R')
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

  print("Linear Filter...")
  linear_filtered_img = linear_filter(noised_image, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))
  linear_filtered_img1 = Image.fromarray(linear_filtered_img)
  linear_filtered_img1.show()
  print("Done.")

  print("Median Filter...")
  median_filtered_img = median_filter(noised_image, 3)
  median_filtered_img1 = Image.fromarray(median_filtered_img)
  median_filtered_img1.show()
  print("Done.")


def operations(image_path):
  t0 = time.time()
  img = Image.open(image_path)
  gray_image = rgb_to_gray(np.array(img), 'G')
  noised_image = salt_pepper_noise(gray_image, 0.1)
  noised_image_2 = gaussian_noise(gray_image, 10)
  histogram_org, histogram_eq, img_eq = equalized_histogram(gray_image)
  msqe = mean_square_error(gray_image, img_eq)
  linear_filtered_img = linear_filter(noised_image, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))
  median_filtered_img = median_filter(noised_image, 3)
  t1 = time.time()


def batch_process_function(image_path):
  return operations(image_path)


if __name__ == "__main__":
  # test_utils('data/Cancerous cell smears/cyl01.BMP')
  base_path = Path('data/Cancerous cell smears')
  files = list(base_path.glob("*.BMP"))
  tqdm_batch.batch_process(files, batch_process_function, n_workers=5, sep_progress=True)
