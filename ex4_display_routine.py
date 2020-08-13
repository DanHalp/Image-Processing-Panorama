import sol4
import sol4_utils as ut
import shlom as sh


def display_routine(min_score=0.8, ransaic_iter=75, inlier_tol=10):
    # Read the two images in such order so they will be aligned from left to right.
    # Make sure that you've put "read_image()" and "relpath()" from ex3 in sol4_utils.py.
    image1 = ut.read_image(ut.relpath("dump/boat/boat051.jpg"), 1)
    image2 = ut.read_image(ut.relpath("dump/boat/boat118.jpg"), 1)

    # For each image - find the harris detector points and their corresponding descriptors
    # using its gaussian pyramid.
    # Make sure you've put your build_guassian_pyramid func from ex3 in sol4_utils.py
    # along with accompanying functions.
    points1, desc1 = sol4.find_features(ut.build_gaussian_pyramid(image1, 3, 3)[0])
    points2, desc2 = sol4.find_features(ut.build_gaussian_pyramid(image2, 3, 3)[0])

    # Match the points between the two images.
    # If implemented correctly, m_indices1 and m_indices2 would hold the
    # corresponding indices in each descriptors arrays, such that:
    # m_indices1[i] would be the matching descriptor, and therefore point,
    # of m_indices2[i], for some i.
    # THERE MIGHT BE DUPLICATIONS, but the RANSAC algorithm would solve it.
    m_indices1, m_indices2 = sol4.match_features(desc1, desc2, min_score)
    p1, p2 = points1[m_indices1], points2[m_indices2]

    # Find for each point in p1 the most fitting point in p2, using RANSAC.
    # The inliers array hold the indices in p1, in which the inliers are located
    # according the algorithm.
    H, inliers = sol4.ransac_homography(p1, p2, ransaic_iter, inlier_tol)

    # Display the two images aligned, and the matching points from both images.
    # TIP: the x values of p2 need to be shifted to the right by image.shape[1]
    # in order to be displayed in the second image.
    sol4.display_matches(image1, image2, p1, p2, inliers)


if __name__ == '__main__':
    # display_routine()
    ob = sol4.PanoramicVideoGenerator("images", "oxford", 2)
    ob.align_images()
    # ob.generate_panoramic_images(7)
    # ob.show_panorama(2)


