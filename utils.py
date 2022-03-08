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
    return Avg
    # grayImage = img.copy()
    #
    # for i in range(3):
    #     grayImage[:, :, i] = Avg
    #
    # return grayImage


def salt_pepper_noise(img, strength):
    out = np.copy(img)
    m, n = img.shape

    num_salt = np.ceil(strength * img.size * 0.5)
    for i in range(int(num_salt)):
        x = np.random.randint(0, m - 1)
        y = np.random.randint(0, n - 1)
        out[x][y] = 0

    num_pepper = np.ceil(strength * img.size * 0.5)
    for i in range(int(num_pepper)):
        x = np.random.randint(0, m - 1)
        y = np.random.randint(0, n - 1)
        out[x][y] = 0

    return out
