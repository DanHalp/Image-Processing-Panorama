from imageio import imread
from scipy.signal import convolve2d
import numpy as np
import os
import scipy.ndimage.filters as filters
from skimage.color import rgb2grey


def gaussian_kernel(kernel_size):
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img


def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)


def is_grey(image):
    """
    Determines whether the image is gray-scale or RGB.
    Notice that some gray-scale images are loaded with 3-dimensions, where in each pixel, the 3 values
    are the same.
    :param image:
    :return:
    """
    image = np.array(image)

    if len(image.shape) > 2:
        return np.array_equal(image[:, :, 0], image[:, :, 1]) and np.array_equal(image[:, :, 0], image[:, :, 2])
    return True

def read_image(filename, representation):
    """
    reads an image file and converts it into a given representation.
    :param filename: Name of the photo, in the working directory.
    :param representation: Result representation form (1 for grayscale and 2 for RGB).
    :return: A np.float64 matrix (image) normalized to the range [0, 1].
    """
    image = imread(filename)
    is_g = is_grey(image)

    # Should not convert a gray-scale image into RBG.
    if is_g and representation == 2:
        raise Exception("Should not convert gray-scale to RGB")
    # Convert an RBG image into gray-scale
    elif not is_g and representation == 1:
        # Returns normalized
        image = rgb2grey(image)
    # Some gray-scale images are loaded as RBG, but all values of the each pixels are the same.
    elif is_g and len(image.shape) == 3:
        image = np.mean(image, axis=2) / 255
    else:
        # Normalize
        image = image / 255

    return image.astype("float64")


def reduce(image, g_filter):
    """
    Reduce pic by cutting each dimension by half, taking only even indices, after blurring
    with a gaussian filter.
    """
    # 1. blur
    image = filters.convolve(image, g_filter)
    image = filters.convolve(image, g_filter.T)

    # 2. reduce:
    return image[::2,0::2]


def build_gaussian_pyramid(im, max_level, filter_size):
    """
    Built the gaussian pyramid, where the dimensions of the smallest image must be higher than 16.
    """
    # Create the gaussian filter, with requested size ([1,1] or [1,2,1] etc).
    g_filter = gaussian_kernel(filter_size)

    # Run at maximum max_level iterations, to create the gaussian pyramid:
    pyramid = [im]
    current_level_im = im.copy()
    for _ in range(max_level - 1):

        # Lower bound of last image of the pyramid.
        temp = reduce(current_level_im, g_filter)
        x, y = temp.shape
        if x < 16 or y < 16:
            break
        current_level_im = temp
        pyramid.append(current_level_im)

    return pyramid, g_filter
