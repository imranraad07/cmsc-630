import argparse
import time
from pathlib import Path
from matplotlib import pyplot as plt

import numpy as np
from PIL import Image
from PIL.ImageFile import ImageFile
from batch_process import batch_process

ImageFile.LOAD_TRUNCATED_IMAGES = True

from utils import rgb_to_gray, calc_avg_histogram, salt_pepper_noise, gaussian_noise, image_quantization, \
  equalized_histogram, mean_square_error, linear_filter, median_filter


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

  print("Quantization...")
  img_quant = image_quantization(gray_image, [0, 5, 10, 30, 50, 90, 200])
  img_quant1 = Image.fromarray(img_quant)
  img_quant1.show()
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
  median_filtered_img = median_filter(noised_image, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))
  median_filtered_img1 = Image.fromarray(median_filtered_img)
  median_filtered_img1.show()
  print("Done.")


def save_image(name, data):
  img = Image.fromarray(data)
  if img.mode != 'RGB':
    img = img.convert('RGB')
  img.save(name)


def save_histogram(name, data):
  fig = plt.hist(data, bins=256, range=(0, 256))
  plt.savefig(name)
  plt.close()


def operations(image_path, args):
  t0 = time.time()
  file_name = str(image_path.stem)
  img = Image.open(image_path)
  gray_image = rgb_to_gray(np.array(img), args.color_channel)
  save_image(args.output_path + '/' + file_name + '_gray_image.' + args.image_type, gray_image)
  noised_image = salt_pepper_noise(gray_image, args.salt_pepper_noise_strength)
  save_image(args.output_path + '/' + file_name + '_salt_n_pepper.' + args.image_type, noised_image)
  noised_image_2 = gaussian_noise(gray_image, args.gaussian_strength)
  save_image(args.output_path + '/' + file_name + '_gaussian.' + args.image_type, noised_image_2)
  histogram_org, histogram_eq, img_eq = equalized_histogram(gray_image)
  save_histogram(args.output_path + '/' + file_name + '_histogram.png', histogram_org)
  save_histogram(args.output_path + '/' + file_name + '_equalized_histogram.png', histogram_eq)
  save_image(args.output_path + '/' + file_name + '_equalized.' + args.image_type, img_eq)

  thresholds = args.quantization_thresholds.split()
  thresholds = list(map(int, thresholds))
  img_quant = image_quantization(gray_image, thresholds)
  save_image(args.output_path + '/' + file_name + '_quantized.' + args.image_type, img_eq)

  msqe = mean_square_error(gray_image, img_quant)

  weight = np.array(args.linear_filter_weights.split())
  weight = weight.astype(int)
  weight = weight.reshape(args.linear_filter_mask, args.linear_filter_mask)
  linear_filtered_img = linear_filter(noised_image, weight)
  save_image(args.output_path + '/' + file_name + '_linear_filtered.' + args.image_type, linear_filtered_img)

  weight = np.array(args.median_filter_weights.split())
  weight = weight.astype(int)
  weight = weight.reshape(args.median_filter_mask, args.median_filter_mask)
  median_filtered_img = median_filter(noised_image, weight)
  save_image(args.output_path + '/' + file_name + '_median_filtered.' + args.image_type, median_filtered_img)

  image_classes = args.image_class_type.split()
  image_class = 'let'
  for name in image_classes:
    if file_name.startswith(name):
      image_class = name

  t1 = time.time()
  print(file_name, "Finished, MSQE", msqe, ', Operation time', t1 - t0)
  return image_class, histogram_org


def batch_process_function(image_path, args):
  return operations(image_path, args)


def calc_avg_hist(results, output_path, image_classes):
  for img_class in image_classes:
    hist_list = []
    for res in results:
      if res[0] == img_class:
        hist_list.append(res[1])
    if len(hist_list) > 0:
      avg_hist = calc_avg_histogram(hist_list)
      name = output_path + '/avg_' + img_class + '_histogram'
      save_histogram(name, avg_hist)


if __name__ == "__main__":
  # test_utils('data/Cancerous cell smears/cyl01.BMP')
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_path', type=str, required=True)
  parser.add_argument('--output_path', type=str, required=True)
  parser.add_argument('--image_type', type=str, required=True)
  parser.add_argument('--batch_size', type=int, required=True, default=10)
  parser.add_argument('--color_channel', type=str, required=False, choices=['R', 'G', 'B'], default=None)
  parser.add_argument('--linear_filter_weights', type=str, required=False, default='1 1 1 1 1 1 1 1 1')
  parser.add_argument('--linear_filter_mask', type=int, required=False, default=3)
  parser.add_argument('--median_filter_weights', type=str, required=False, default='1 1 1 1 1 1 1 1 1')
  parser.add_argument('--median_filter_mask', type=int, required=False, default=3)
  parser.add_argument('--salt_pepper_noise_strength', type=float, required=False, default=0.1)
  parser.add_argument('--gaussian_strength', type=int, required=False, default=10)
  parser.add_argument('--image_class_type', type=str, required=False, default='cyl inter let mod para super svar')
  parser.add_argument('--quantization_thresholds', type=str, required=False, default='10 30 60 120 180')

  args = parser.parse_args()

  base_path = Path(args.base_path)
  files = list(base_path.glob("*." + args.image_type))
  batch_size = args.batch_size
  results = batch_process(files, batch_process_function, batch_size=batch_size, args=args)
  calc_avg_hist(results, args.output_path, args.image_class_type.split())
