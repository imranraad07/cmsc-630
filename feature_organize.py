import csv
from pathlib import Path

import numpy as np
from PIL import Image

from utils import rgb_to_gray, histogram

np.seterr(divide='ignore', invalid='ignore')

# img = Image.open('Test/cyl01_segment_hist_imageR.BMP')
# gray_image = rgb_to_gray(np.array(img), 'R')
# print(gray_image.shape)
# hist = histogram(gray_image)
# print(np.sum(hist) - hist[255])
# print(np.std(hist))
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

# base_path = Path('output')
# files = list(base_path.glob("*.BMP"))
# print(len(files))
# rows = []
# for file in files:
#     if 'segment_hist' in file.name:
#         color_channel = file.name[len(file.name) - 5]
#         img = Image.open(file)
#         gray_image = rgb_to_gray(np.array(img), color_channel)
#         hist = histogram(gray_image)
#         name = file.name.split("_")[0]
#         type = ''.join(i for i in name if not i.isdigit())
#         print(file.name, name, type, color_channel, np.sum(hist) - hist[255], np.std(hist))
#         rows.append([name, color_channel, np.sum(hist) - hist[255], np.std(hist), type])

dict_red_pixel = {}
dict_green_pixel = {}
dict_blue_pixel = {}
dict_red_std = {}
dict_green_std = {}
dict_blue_std = {}
dict_red_edge = {}
dict_green_edge = {}
dict_blue_edge = {}
dict_red_erode = {}
dict_green_erode = {}
dict_blue_erode = {}
dict_red_dilate = {}
dict_green_dilate = {}
dict_blue_dilate = {}

ids = []

with open('features.tsv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if row[0] not in ids:
            ids.append(row[0])
        if row[1] == 'R':
            dict_red_pixel[row[0]] = row[2]
            dict_red_std[row[0]] = row[3]
        if row[1] == 'G':
            dict_green_pixel[row[0]] = row[2]
            dict_green_std[row[0]] = row[3]
        if row[1] == 'B':
            dict_blue_pixel[row[0]] = row[2]
            dict_blue_std[row[0]] = row[3]

with open('features2.tsv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if row[0] not in ids:
            ids.append(row[0])
        if row[1] == 'R':
            dict_red_edge[row[0]] = row[2]
        if row[1] == 'G':
            dict_green_edge[row[0]] = row[2]
        if row[1] == 'B':
            dict_blue_edge[row[0]] = row[2]

with open('features3.tsv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if row[0] not in ids:
            ids.append(row[0])
        if row[1] == 'R':
            dict_red_erode[row[0]] = row[2]
        if row[1] == 'G':
            dict_green_erode[row[0]] = row[2]
        if row[1] == 'B':
            dict_blue_erode[row[0]] = row[2]

rows = []

for id in ids:
    type = ''.join(i for i in id if not i.isdigit())
    rows.append(
        [id, dict_red_pixel[id], dict_green_pixel[id], dict_blue_pixel[id], dict_red_std[id], dict_green_std[id],
         dict_blue_std[id], dict_red_edge[id], dict_green_edge[id], dict_blue_edge[id],
         dict_red_erode[id], dict_green_erode[id], dict_blue_erode[id], type])

with open('features_organized.tsv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',')
    csvwriter.writerows(rows)
