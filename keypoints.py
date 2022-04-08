import numpy as np
import cv2
from matplotlib import pyplot as plt
from file_management import *
import math


def extract_features(image):
    """
    Find keypoints and descriptors for the image

    Arguments:
    image -- a grayscale image

    Returns:
    kp -- list of the extracted keypoints (features) in an image
    des -- list of the keypoint descriptors in an image
    """
    # Initiate ORB detector
    orb = cv2.ORB_create()

    # find the keypoints with ORB
    kp = orb.detect(image, None)

    # compute the descriptors with ORB
    kp, des = orb.compute(image, kp)

    return kp, des


def visualize_features(image, kp):
    """
    Visualize extracted features in the image

    Arguments:
    image -- a grayscale image
    kp -- list of the extracted keypoints

    Returns:
    """
    plt.figure(figsize=(8, 6), dpi=100)
    display = cv2.drawKeypoints(image, kp, None, color=(0,255,0), flags=0)
    plt.imshow(display)


def extract_features_dataset(images, extract_features_function):
    """
    Find keypoints and descriptors for each image in the dataset

    Arguments:
    images -- a list of grayscale images
    extract_features_function -- a function which finds features (keypoints and descriptors) for an image

    Returns:
    kp_list -- a list of keypoints for each image in images
    des_list -- a list of descriptors for each image in images

    """
    kp_list = []
    des_list = []

    for im in images:
        kp, des = extract_features(im)
        kp_list.append(kp)
        des_list.append(des)

    return kp_list, des_list


def match_features(des1, des2):
    """
    Match features from two images

    Arguments:
    des1 -- list of the keypoint descriptors in the first image
    des2 -- list of the keypoint descriptors in the second image

    Returns:
    match -- list of matched features from two images. Each match[i] is k or less matches for the same query descriptor
    """
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,  # 12
                        key_size=12,  # 20
                        multi_probe_level=1)  # 2
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    match = flann.knnMatch(des1, des2, k=2)

    return match


def filter_matches_distance(match, dist_threshold):
    """
    Filter matched features from two images by distance between the best matches

    Arguments:
    match -- list of matched features from two images
    dist_threshold -- maximum allowed relative distance between the best matches, (0.0, 1.0)

    Returns:
    filtered_match -- list of good matches, satisfying the distance threshold
    """
    filtered_match = []

    # ratio test as per Lowe's paper
    try:
        for m, n in match:
            if m.distance < dist_threshold * n.distance:
                filtered_match.append(m)
    except ValueError:
        print("Skip filtering match.")

    return filtered_match


def visualize_matches(image1, kp1, image2, kp2, match):
    """
    Visualize corresponding matches in two images

    Arguments:
    image1 -- the first image in a matched image pair
    kp1 -- list of the keypoints in the first image
    image2 -- the second image in a matched image pair
    kp2 -- list of the keypoints in the second image
    match -- list of matched features from the pair of images

    Returns:
    image_matches -- an image showing the corresponding matches on both image1 and image2 or None if you don't use this function
    """
    image_matches = cv2.drawMatches(image1,kp1,image2,kp2,match,None,flags=2)
    plt.figure(figsize=(16, 6), dpi=100)
    plt.imshow(image_matches)
    return image_matches


def match_features_dataset(des_list, match_features):
    """
    Match features for each subsequent image pair in the dataset

    Arguments:
    des_list -- a list of descriptors for each image in the dataset
    match_features -- a function which maches features between a pair of images

    Returns:
    matches -- list of matches for each subsequent image pair in the dataset.
               Each matches[i] is a list of matched features from images i and i + 1

    """
    matches = []
    for i in range(0, len(des_list) - 1):
        des1 = des_list[i]
        des2 = des_list[i + 1]

        match = match_features(des1, des2)
        matches.append(match)
    return matches


def filter_matches_dataset(filter_matches_distance, matches, dist_threshold):
    """
    Filter matched features by distance for each subsequent image pair in the dataset

    Arguments:
    filter_matches_distance -- a function which filters matched features from two images by distance between the best matches
    matches -- list of matches for each subsequent image pair in the dataset.
               Each matches[i] is a list of matched features from images i and i + 1
    dist_threshold -- maximum allowed relative distance between the best matches, (0.0, 1.0)

    Returns:
    filtered_matches -- list of good matches for each subsequent image pair in the dataset.
                        Each matches[i] is a list of good matches, satisfying the distance threshold

    """
    filtered_matches = []

    for i, match in enumerate(matches):
        match = filter_matches_distance(match, dist_threshold)
        filtered_matches.append(match)

    return filtered_matches


if __name__ == "__main__":
    np.random.seed(1)
    dataset_handler = DatasetHandler()

    i = 0
    image = dataset_handler.images[i]
    image_rgb = dataset_handler.images_rgb[i]
    depth = dataset_handler.depth_maps[i]
    print("Depth map shape: {0}".format(depth.shape))

    v, u = depth.shape
    depth_val = depth[v-1, u-1]
    print("Depth value of the very bottom-right pixel of depth map {0} is {1:0.3f}".format(i, depth_val))
    print(dataset_handler.k)
    print(dataset_handler.num_frames)

    kp, des = extract_features(image)
    print("Number of features detected in frame {0}: {1}\n".format(i, len(kp)))
    print("Coordinates of the first keypoint in frame {0}: {1}".format(i, str(kp[0].pt)))

    visualize_features(image, kp)
    images = dataset_handler.images
    kp_list, des_list = extract_features_dataset(images, extract_features)
    print("Number of features detected in frame {0}: {1}".format(i, len(kp_list[i])))
    print("Coordinates of the first keypoint in frame {0}: {1}\n".format(i, str(kp_list[i][0].pt)))

    des1 = des_list[i]
    des2 = des_list[i+1]
    kp1 = kp_list[i]
    kp2 = kp_list[i+1]
    match = match_features(des1, des2)
    print("Number of features matched in frames {0} and {1}: {2}".format(i, i+1, len(match)))

    # Visualize n first matches, set n to None to view all matches
    # set filtering to True if using match filtering, otherwise set to False
    n = 20
    dist_threshold = 0.6
    match = filter_matches_distance(match, dist_threshold)
    print("Number of features matched in frames {0} and {1} after filtering by distance: {2}".format(i, i + 1, len(
        match)))

    image_matches = visualize_matches(image, kp1, dataset_handler.images[i+1], kp2, match[:n])
    unfiltered_matches = match_features_dataset(des_list, match_features)
    matches = filter_matches_dataset(filter_matches_distance, unfiltered_matches, dist_threshold)
    print("Number of filtered matches in frames {0} and {1}: {2}".format(i, i + 1, len(matches[i])))

    plt.show()