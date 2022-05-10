import csv
from pathlib import Path

import numpy as np
from PIL import Image

from utils import rgb_to_gray, histogram

np.seterr(divide='ignore', invalid='ignore')

img = Image.open('Test/cyl01_segment_kmeans_image_B.BMP')
gray_image = rgb_to_gray(np.array(img), 'B')
print(gray_image.shape)
hist = histogram(gray_image)
# print(np.sum(hist) - hist[255])
print(hist, np.std(hist))
#
# img = Image.open('Test/cyl01_segment_hist_imageG.BMP')
# gray_image = rgb_to_gray(np.array(img), 'G')
# print(gray_image.shape)
# hist = histogram(gray_image)
# print(np.sum(hist) - hist[255])
# print(np.std(hist))
#
# img = Image.open('Test/cyl01_segment_hist_imageB.BMP')
# gray_image = rgb_to_gray(np.array(img), 'B')
# print(gray_image.shape)
# hist = histogram(gray_image)
# print(np.sum(hist) - hist[255])
# print(np.std(hist))

base_path = Path('output')
files = list(base_path.glob("*.BMP"))
print(len(files))
rows = []
for file in files:
    if 'segment_hist' in file.name:
        color_channel = file.name[len(file.name) - 5]
        img = Image.open(file)
        gray_image = rgb_to_gray(np.array(img), color_channel)
        hist = histogram(gray_image)
        name = file.name.split("_")[0]
        type = ''.join(i for i in name if not i.isdigit())
        print(file.name, name, type, color_channel, np.sum(hist) - hist[255], np.std(hist))
        rows.append([name, color_channel, np.sum(hist) - hist[255], np.std(hist), type])

with open('features.tsv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',')
    csvwriter.writerows(rows)
