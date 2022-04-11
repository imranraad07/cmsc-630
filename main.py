import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image
from PIL.ImageFile import ImageFile

from edge_detection import edge_detect
from batch_process import batch_process
from erode_dilate import erode_image, dilate_image

from segmentation_histogram import histogram_thresholding
from segmentation_kmeans import kmeans_clusters

ImageFile.LOAD_TRUNCATED_IMAGES = True

from utils import rgb_to_gray


def save_image(name, data):
    img = Image.fromarray(data)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.save(name)


def operations(image_path, args):
    t0 = time.time()
    file_name = str(image_path.stem)
    img = Image.open(image_path)
    gray_image = rgb_to_gray(np.array(img), 'G')

    filter = np.array(args.edge_filter_operator.split())
    filter = filter.astype(int)
    filter_size = np.array(args.edge_filter_size.split())
    filter_size = filter_size.astype(int)
    filter = filter.reshape(filter_size[0], filter_size[1])
    edge_image = edge_detect(gray_image, filter)
    save_image(args.output_path + '/' + file_name + '_edge_image.' + args.image_type, edge_image)

    eroded_image = erode_image(gray_image)
    save_image(args.output_path + '/' + file_name + '_eroded_image.' + args.image_type, eroded_image)

    dilated_image = dilate_image(gray_image)
    save_image(args.output_path + '/' + file_name + '_dilated_image.' + args.image_type, dilated_image)

    segment_hist_image = histogram_thresholding(gray_image)
    save_image(args.output_path + '/' + file_name + '_segment_hist_image.' + args.image_type, segment_hist_image)

    kmeans_cluster_image = kmeans_clusters(gray_image, args.k_means_clusters)
    save_image(args.output_path + '/' + file_name + '_segment_kmeans_image.' + args.image_type, kmeans_cluster_image)

    t1 = time.time()
    time_calculation = t1 - t0
    # print(time_calculation)
    return edge_image


def batch_process_function(image_path, args):
    return operations(image_path, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--image_type', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True, default=10)
    parser.add_argument('--k_means_clusters', type=int, required=True, default=2)
    parser.add_argument('--edge_filter_operator', type=str, required=True, default='-3 0 3 -10 0 10 -3 0 3')
    parser.add_argument('--edge_filter_size', type=str, required=True, default='3 3')
    parser.add_argument('--color_channel', type=str, required=False, choices=['R', 'G', 'B'], default=None)

    args = parser.parse_args()

    base_path = Path(args.base_path)
    files = list(base_path.glob("*." + args.image_type))
    batch_size = args.batch_size
    results = batch_process(files, batch_process_function, batch_size=batch_size, args=args)
