# Initial code for ex4.
# You may change this code, but keep the functions' signatures
# You can also split the code to multiple files as long as this file's API is unchanged 

import numpy as np
import os
import matplotlib.pyplot as plt

from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label, center_of_mass
import shutil
from imageio import imwrite
from scipy.ndimage.filters import convolve

import sol4_utils as ut
from scipy.ndimage import map_coordinates


def harris_corner_detector(im):
    """
    Detects harris corners.
    Make sure the returned coordinates are x major!!!
    :param im: A 2D array representing an image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """

    # 1) let us create the derevatives:
    kernel = np.array([[1, 0, -1]]).astype(np.float64)
    Ix = convolve(im, kernel)
    Iy = convolve(im, kernel.T)

    # 2) Create M
    Ix2 = ut.blur_spatial(Ix * Ix, 3)
    Iy2 = ut.blur_spatial(Iy * Iy, 3)
    Ixy = ut.blur_spatial(Iy * Ix, 3)
    # M = np.stack((Ix2, Ixy, Ixy, Iy2), axis=-1).reshape((im.shape[0], im.shape[1], 2, 2))

    # Det(M) - K * (trace(M)) ^ 2
    R = ((Ix2 * Iy2) - (Ixy * Ixy)) - (0.04 * np.square(Ix2 + Iy2))
    points = np.array(np.where(non_maximum_suppression(R)))

    # Return the relevant indices as (col, row) and NOT (row, col) as usual.
    return np.array([points[1], points[0]]).T


def sample_descriptor(im, pos, desc_rad):
    """
    Samples descriptors at the given corners.
    :param im: A 2D array representing an image.
    :param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.
    :param desc_rad: "Radius" of descriptors to compute.
    :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
    """

    K, r, N = 2 * desc_rad + 1, desc_rad, len(pos)

    # creating mashgrids of descriptors for all points together (np.mashgrid does not allow that):
    y = np.linspace(pos[:, 0] - r, pos[:, 0] + r, K).T
    x = np.linspace(pos[:, 1] - r, pos[:, 1] + r, K).T
    a = np.repeat(x, K, axis=0).reshape((N, K, K))
    b = np.repeat(y, K, axis=0).reshape((N, K, K)).transpose((0,2,1))
    g = np.array([a,b])

    # Translate values from original image coordinates to the new_ones, since the the indices in
    # pos_new are not necessarily integers.
    z1 = np.array(map_coordinates(im, g, order=1, prefilter=False)).reshape(N, K ** 2)

    # Normalize
    mean1 = np.mean(z1, axis=1).reshape(N, 1)
    norm1 = np.linalg.norm(z1 - mean1, axis=1).reshape(N, 1)
    norm1[norm1 == 0] = 1   # Deal with division in zero
    z1 = ((z1 - mean1) / norm1).reshape(N, K, K)

    return z1


def translate_coors_between_pyr_levels(p, l_cur, l_to):
    return (2 ** (l_cur - l_to)) * np.array(p)


def find_features(pyr):
    """
    Detects and extracts feature points from a pyramid.
    :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
    :return: A list containing:
                1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                   These coordinates are provided at the pyramid level pyr[0].
                2) A feature descriptor array with shape (N,K,K)
    """
    r, patch_axis = 20, 7
    points = spread_out_corners(pyr[0], patch_axis, patch_axis, r)
    # Translate the positions from the original image to the 3rd layer of the gaussian pyramid
    pos_new = translate_coors_between_pyr_levels(points, 0, 2)
    descriptors = sample_descriptor(pyr[2], pos_new, r)
    return points, descriptors


def match_features(desc1, desc2, min_score):
    """
    Return indices of matching descriptors.
    :param desc1: A feature descriptor array with shape (N1,K,K).
    :param desc2: A feature descriptor array with shape (N2,K,K).
    :param min_score: Minimal match score.
    :return: A list containing:
                1) An array with shape (M,) and dtype int of matching indices in desc1.
                2) An array with shape (M,) and dtype int of matching indices in desc2.
    """
    # Find the best matches. First dot the two matrices, and then sort the rows and cols separately.
    # NOTICE: Rows represent the descriptors in desc1, and the Cols represent the descriptors in desc2.
    # Then, extract the second best value from each row, and then from each column. Do it by sorting by axis.
    # Now, evaluate each row in "scores" against the sec best value in each row (scores >= sec_best_rows).
    # Do the same against the the columns, and min_score.
    # Return the indices of the fitting matches, where x holds the relevant ones in desc1, and y holds the
    # CORRESPONDING ones in desc2.
    shape1, shape2 = desc1.shape, desc2.shape
    desc1 = desc1.reshape([shape1[0], shape1[1] * shape1[2]])
    desc2 = desc2.reshape([shape2[0], shape2[1] * shape2[2]])
    scores = np.dot(desc1, desc2.T)
    sec_best_rows = np.sort(scores, axis=1)[:, -2:][:, 0].reshape((len(desc1), 1))
    sec_best_cols = np.sort(scores, axis=0)[-2:, :][0, :].reshape((1, len(desc2)))
    x, y = np.where((scores >= sec_best_rows) & (scores >= sec_best_cols) & (scores >= min_score))
    return x, y


def apply_homography(pos1, H12):
    """
    Apply homography to inhomogenous points.
    :param pos1: An array with shape (N,2) of [x,y] point coordinates.
    :param H12: A 3x3 homography matrix.
    :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
    """
    # transform all points in pos1 (x, y) --> (x, y, 1)
    ones = np.ones((pos1.shape[0], pos1.shape[1] + 1))
    ones[:, :2] = pos1
    # [x', y', z'] = np.dot(H12, [x, y, 1])
    res = np.dot(H12, ones.T).T
    # [x, y] = [x', y'] / z'
    return res[:, :2] / res[:, 2].reshape(len(res), -1)


def ransac_homography(points1, points2, num_iter, inlier_tol, translation_only=False):
    """
      Computes homography between two sets of points using RANSAC.
      :param pos1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
      :param pos2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
      :param num_iter: Number of RANSAC iterations to perform.
      :param inlier_tol: inlier tolerance threshold.
      :param translation_only: see estimate rigid transform
      :return: A list containing:
                  1) A 3x3 normalized homography matrix.
                  2) An Array with shape (S,) where S is the number of inliers,
                      containing the indices in pos1/pos2 of the maximal set of inlier matches found.
      """

    # How many points would we need for the H?
    n_points = 1 if translation_only else 2
    inliners = []
    smaller_am_of_indi = min(points1.shape[0], points2.shape[0])

    # RANSAC ALGO
    for _ in range(num_iter):
        indices = np.random.permutation(np.arange(smaller_am_of_indi))[:n_points]
        rand_points_1 = points1[indices]
        rand_points_2 = points2[indices]

        # Find the error of the estimated rigid_transform by comaring the euclidean
        # distance between transformation results and the wanted ones.
        curr_H = estimate_rigid_transform(rand_points_1, rand_points_2, translation_only)
        trans_points1 = apply_homography(points1, curr_H)
        euclidean_dis = np.linalg.norm(trans_points1 - points2, axis=1)
        curr_inliners = np.array(np.where(euclidean_dis < inlier_tol))[0]

        # Pick the best choice:
        if len(curr_inliners) > len(inliners):
            inliners = curr_inliners

    H = estimate_rigid_transform(points1[inliners], points2[inliners], translation_only)

    return [H, inliners]


def display_matches(im1, im2, points1, points2, inliers):
    """
  Dispalay matching points.
  :param im1: A grayscale image.
  :param im2: A grayscale image.
  :parma pos1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
  :param pos2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
  :param inliers: An array with shape (S,) of inlier matches.
  """

    # Create image
    matched_image = np.hstack((im1, im2))

    # All points
    points2[:, 0] += im1.shape[1]  # Move the im2 dots to their fitting place in the new image.

    # Outliers.
    outliers1, outliers2 = np.ones(len(points1)), np.ones(len(points2))
    outliers1[inliers], outliers2[inliers] = 0, 0
    outliers_p = np.array([points1[outliers1.astype(np.bool)], points2[outliers2.astype(np.bool)]])

    # Inliers
    inliers_p = np.array([points1[inliers], points2[inliers]])

    # Show points, inliers and outliers:
    plt.imshow(matched_image, cmap="gray")
    plt.plot(outliers_p[:, :, 0], outliers_p[:, :, 1], mfc='r', c='b', lw=.1, ms=10, marker=",")
    plt.plot(inliers_p[:, :, 0], inliers_p[:, :, 1], mfc='r', c='y', lw=.4, ms=10, marker=",")
    plt.show()


def accumulate_homographies(H_succesive, m):
    """
    Convert a list of succesive homographies to a
    list of homographies to a common reference frame.
    :param H_successive: A list of M-1 3x3 homography
    matrices where H_successive[i] is a homography which transforms points
    from coordinate system i to coordinate system i+1.
    :param m: Index of the coordinate system towards which we would like to
    accumulate the given homographies.
    :return: A list of M 3x3 homography matrices,
    where H2m[i] transforms points from coordinate system i to coordinate system m
    """
    H_succesive = np.array(H_succesive)
    homo = np.zeros((H_succesive.shape[0] + 1, 3, 3))
    homo[m] = np.eye(3)

    # i < m
    for i in range(m-1, -1, -1):
        homo[i] = np.dot(homo[i + 1], H_succesive[i])
        homo[i] /= homo[i, 2, 2]

    # i > m
    for i in range(m, len(H_succesive)):
        homo[i + 1] = np.dot(homo[i], np.linalg.inv(H_succesive[i]))
        homo[i + 1] /= homo[i + 1, 2, 2]

    return homo


def compute_bounding_box(homography, w, h):
    """
    computes bounding box of warped image under homography, without actually warping the image
    :param homography: homography
    :param w: width of the image
    :param h: height of the image
    :return: 2x2 array, where the first row is [x,y] of the top left corner,
    and the second row is the [x,y] of the bottom right corner
    """
    coor = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    coor = apply_homography(coor, homography)
    xs, ys = coor[:, 0], coor[:, 1]
    Xmax, Ymax = np.ceil(np.max(xs)), np.ceil(np.max(ys))
    Xmin, Ymin = np.floor(np.min(xs)), np.floor(np.min(ys))
    return np.array([[Xmin, Ymin], [Xmax, Ymax]]).astype(np.int)


def warp_channel(image, homography):
    """
    Warps a 2D image with a given homography.
    :param image: a 2D image.
    :param homography: homograhpy.
    :return: A 2d warped image.
    """
    # Boundaries of the image
    B = compute_bounding_box(homography, image.shape[1], image.shape[0])
    xs, ys = B[:, 0], B[:, 1]
    shape = (xs[1] - xs[0] + 1, ys[1] - ys[0] + 1)
    xrange, yrange = np.arange(xs[0], xs[1] + 1), np.arange(ys[0], ys[1] + 1)
    x, y = np.meshgrid(xrange, yrange)
    points = np.array([x.flatten(), y.flatten()]).T
    wraped_p = apply_homography(points, np.linalg.inv(homography))
    new_im = map_coordinates(image, (wraped_p[:, 1], wraped_p[:, 0]), order=1, prefilter=False)\
        .reshape(shape[1], shape[0])
    return new_im


def warp_image(image, homography):
    """
  Warps an RGB image with a given homography.
  :param image: an RGB image.
  :param homography: homograhpy.
  :return: A warped image.
  """
    return np.dstack([warp_channel(image[..., channel], homography) for channel in range(3)])


def filter_homographies_with_translation(homographies, minimum_right_translation):
    """
  Filters rigid transformations encoded as homographies by the amount of translation from left to right.
  :param homographies: homograhpies to filter.
  :param minimum_right_translation: amount of translation below which the transformation is discarded.
  :return: filtered homographies..
  """
    translation_over_thresh = [0]
    last = homographies[0][0, -1]
    for i in range(1, len(homographies)):
        if homographies[i][0, -1] - last > minimum_right_translation:
            translation_over_thresh.append(i)
            last = homographies[i][0, -1]
    return np.array(translation_over_thresh).astype(np.int)


def estimate_rigid_transform(points1, points2, translation_only=False):
    """
  Computes rigid transforming points1 towards points2, using least squares method.
  points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
  :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
  :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
  :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
  :return: A 3x3 array with the computed homography.
  """
    centroid1 = points1.mean(axis=0)
    centroid2 = points2.mean(axis=0)

    if translation_only:
        rotation = np.eye(2)
        translation = centroid2 - centroid1

    else:
        centered_points1 = points1 - centroid1
        centered_points2 = points2 - centroid2

        sigma = centered_points2.T @ centered_points1
        U, _, Vt = np.linalg.svd(sigma)

        rotation = U @ Vt
        translation = -rotation @ centroid1 + centroid2

    H = np.eye(3)
    H[:2, :2] = rotation
    H[:2, 2] = translation
    return H


def non_maximum_suppression(image):
    """
  Finds local maximas of an image.
  :param image: A 2D array representing an image.
  :return: A boolean array with the same shape as the input image, where True indicates local maximum.
  """
    # Find local maximas.
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    local_max[image < (image.max() * 0.1)] = False

    # Erode areas to single points.
    lbs, num = label(local_max)
    centers = center_of_mass(local_max, lbs, np.arange(num) + 1)
    centers = np.stack(centers).round().astype(np.int)
    ret = np.zeros_like(image, dtype=np.bool)
    ret[centers[:, 0], centers[:, 1]] = True

    return ret


def spread_out_corners(im, m, n, radius):
    """
  Splits the image im to m by n rectangles and uses harris_corner_detector on each.
  :param im: A 2D array representing an image.
  :param m: Vertical number of rectangles.
  :param n: Horizontal number of rectangles.
  :param radius: Minimal distance of corner points from the boundary of the image.
  :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
  """
    corners = [np.empty((0, 2), dtype=np.int)]
    x_bound = np.linspace(0, im.shape[1], n + 1, dtype=np.int)
    y_bound = np.linspace(0, im.shape[0], m + 1, dtype=np.int)
    for i in range(n):
        for j in range(m):
            # Use Harris detector on every sub image.
            sub_im = im[y_bound[j]:y_bound[j + 1], x_bound[i]:x_bound[i + 1]]
            sub_corners = harris_corner_detector(sub_im)
            sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis, :]
            corners.append(sub_corners)
    corners = np.vstack(corners)
    legit = ((corners[:, 0] > radius) & (corners[:, 0] < im.shape[1] - radius) &
             (corners[:, 1] > radius) & (corners[:, 1] < im.shape[0] - radius))
    ret = corners[legit, :]
    return ret


class PanoramicVideoGenerator:
    """
  Generates panorama from a set of images.
  """

    def __init__(self, data_dir, file_prefix, num_images):
        """
        The naming convention for a sequence of images is file_prefixN.jpg,
        where N is a running number 001, 002, 003...
        :param data_dir: path to input images.
        :param file_prefix: see above.
        :param num_images: number of images to produce the panoramas with.
        """
        self.file_prefix = file_prefix
        self.files = [os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i in
                      range(num_images)]
        self.files = list(filter(os.path.exists, self.files))
        self.panoramas = None
        self.homographies = None
        print('found %d images' % len(self.files))

    def align_images(self, translation_only=False):
        """
        compute homographies between all images to a common coordinate system
        :param translation_only: see estimte_rigid_transform
        """
        # Extract feature point locations and descriptors.
        points_and_descriptors = []
        self.images = []
        for file in self.files:
            image = ut.read_image(file, 1)
            self.images.append(image)
            self.h, self.w = image.shape
            pyramid, _ = ut.build_gaussian_pyramid(image, 3, 7)
            points_and_descriptors.append(find_features(pyramid))

        # Compute homographies between successive pairs of images.
        Hs = []
        for i in range(len(points_and_descriptors) - 1):
            points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
            desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]

            # Find matching feature points.
            ind1, ind2 = match_features(desc1, desc2, .7)
            points1, points2 = points1[ind1, :], points2[ind2, :]

            # Compute homography using RANSAC.
            H12, inliers = ransac_homography(points1, points2, 100, 10, translation_only)

            # Uncomment for debugging: display inliers and outliers among matching points.
            # In the submitted code this function should be commented out!
            display_matches(self.images[i], self.images[i+1], points1, points2, inliers)

            Hs.append(H12)

        # Compute composite homographies from the central coordinate system.
        accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
        self.homographies = np.stack(accumulated_homographies)
        self.frames_for_panoramas = filter_homographies_with_translation(self.homographies,
                                                                         minimum_right_translation=5)
        self.homographies = self.homographies[self.frames_for_panoramas]

    def generate_panoramic_images(self, number_of_panoramas, should_crop=True):
        """
        combine slices from input images to panoramas.
        :param number_of_panoramas: how many different slices to take from each input image
        """
        assert self.homographies is not None

        # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
        self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
        for i in range(self.frames_for_panoramas.size):
            self.bounding_boxes[i] = compute_bounding_box(self.homographies[i], self.w, self.h)

        # change our reference coordinate system to the panoramas
        # all panoramas share the same coordinate system
        global_offset = np.min(self.bounding_boxes, axis=(0, 1))
        self.bounding_boxes -= global_offset

        slice_centers = np.linspace(0, self.w, number_of_panoramas + 2, endpoint=True,
                                    dtype=np.int)[1:-1]
        warped_slice_centers = np.zeros((number_of_panoramas, self.frames_for_panoramas.size))
        # every slice is a different panorama, it indicates the slices of the input images from which the panorama
        # will be concatenated
        for i in range(slice_centers.size):
            slice_center_2d = np.array([slice_centers[i], self.h // 2])[None, :]
            # homography warps the slice center to the coordinate system of the middle image
            warped_centers = [apply_homography(slice_center_2d, h) for h in self.homographies]
            # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
            warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]

        panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(np.int) + 1

        # boundary between input images in the panorama
        x_strip_boundary = ((warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
        x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                      x_strip_boundary,
                                      np.ones((number_of_panoramas, 1)) * panorama_size[0]])
        x_strip_boundary = x_strip_boundary.round().astype(np.int)

        self.panoramas = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3),
                                  dtype=np.float64)
        for i, frame_index in enumerate(self.frames_for_panoramas):
            # warp every input image once, and populate all panoramas
            image = ut.read_image(self.files[frame_index], 2)
            warped_image = warp_image(image, self.homographies[i])
            x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
            y_bottom = y_offset + warped_image.shape[0]

            for panorama_index in range(number_of_panoramas):
                # take strip of warped image and paste to current panorama
                boundaries = x_strip_boundary[panorama_index, i:i + 2]
                image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
                x_end = boundaries[0] + image_strip.shape[1]
                self.panoramas[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip

        if should_crop:
            self.crop()

    def crop(self):
        # crop out areas not recorded from enough angles
        # assert will fail if there is overlap in field of view between the left most image and the right most image
        crop_left = int(self.bounding_boxes[0][1, 0])
        crop_right = int(self.bounding_boxes[-1][0, 0])
        if crop_left < crop_right:
            print(crop_left, crop_right)
            self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]

    def save_panoramas_to_video(self):
        assert self.panoramas is not None
        out_folder = 'tmp_folder_for_panoramic_frames/%s' % self.file_prefix
        try:
            shutil.rmtree(out_folder)
        except:
            print('could not remove folder')
            pass
        os.makedirs(out_folder)
        # save individual panorama images to 'tmp_folder_for_panoramic_frames'
        for i, panorama in enumerate(self.panoramas):
            imwrite('%s/panorama%02d.png' % (out_folder, len(self.panoramas) - i), panorama)
        if os.path.exists('%s.mp4' % self.file_prefix):
            os.remove('%s.mp4' % self.file_prefix)
        # write output video to current folder
        os.system('ffmpeg -framerate 8 -i %s/panorama%%02d.png %s.mp4' %
                  (out_folder, self.file_prefix))

    def show_panorama(self, panorama_index, figsize=(20, 20)):
        assert self.panoramas is not None
        plt.figure(figsize=figsize)
        plt.imshow(self.panoramas[panorama_index].clip(0, 1))
        plt.show()
