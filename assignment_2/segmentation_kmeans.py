from math import sqrt

from PIL import Image

from utils import rgb_to_gray

import numpy as np
import random

np.seterr(divide='ignore', invalid='ignore')


# help: https://medium.com/analytics-vidhya/image-segmentation-using-k-means-clustering-from-scratch-1545c896e38e

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def find_clusters(flatten_image, centers):
    clusters = np.zeros(flatten_image.shape[0])
    for i, pixel in enumerate(flatten_image):
        min_distance = None
        for j, centroid in enumerate(centers):
            distance = euclidean_distance(pixel, centroid)
            if min_distance is None or distance < min_distance:
                min_distance = distance
                clusters[i] = j

    # print(clusters)
    return clusters


def find_cluster_means(flatten_image, k, clusters):
    clusters_sums = np.zeros(k)
    cluster_sizes = np.zeros(k)

    for i in range(k):
        cluster_points = flatten_image[clusters == i]
        clusters_sums[i] = np.sum(cluster_points)
        cluster_sizes[i] = len(cluster_points)

    new_centers = np.divide(clusters_sums, cluster_sizes).astype(int)
    return new_centers


def kmeans_clusters(gray_image, k, max_iteration=100):
    copy_image = gray_image.copy()
    flatten_image = copy_image.flatten().reshape(-1)

    centers = np.random.randint(256, size=k)
    clusters = np.random.randint(low=0, high=k, size=len(flatten_image))

    iteration = 0

    while iteration < max_iteration:
        clusters = find_clusters(flatten_image, centers)
        new_centers = find_cluster_means(flatten_image, k, clusters)
        diffs = np.abs(centers - new_centers)
        centers = new_centers
        iteration += 1
        if np.all(diffs == 0):
            break

    # print(centers)
    out = flatten_image.copy()
    for i, center in enumerate(centers):
        out = np.where(clusters == i, center, out)

    out = out.reshape(gray_image.shape)
    out = out.astype(np.uint8)
    return out

# image_path = 'data/Cancerous cell smears/para11.BMP'
# img = Image.open(image_path)
# gray_image = rgb_to_gray(np.array(img), 'G')
# segmented = kmeans_clusters(gray_image, 2)
# # segmented = kmeans_clusters(gray_image, 3)
# # segmented = kmeans_clusters(gray_image, 4)
# # segmented = kmeans_clusters(gray_image, 5)
# segmented_1 = Image.fromarray(segmented.astype(np.uint8))
# segmented_1.show()
#
# from sklearn.cluster import KMeans
#
# image_path = 'data/Cancerous cell smears/para11.BMP'
# img = Image.open(image_path)
# gray_image = rgb_to_gray(np.array(img), 'R')
# flatten_image = gray_image.flatten()
# flatten_image = flatten_image.reshape(-1, 1)
# print(flatten_image)
# print(flatten_image.shape)
# kmeans = KMeans(n_clusters=2, random_state=0).fit(flatten_image)
# print(kmeans.labels_)
# print(kmeans.labels_.shape)
# img_copy = kmeans.labels_.reshape(gray_image.shape)
# img_copy[img_copy == 1] = 255
# print(img_copy)
# segmented_1 = Image.fromarray(img_copy.astype(np.uint8))
# segmented_1.show()
