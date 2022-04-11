import numpy as np
import cv2
from matplotlib import pyplot as plt
from file_management import *
import math
import keypoints as kp
import odometry as odo

if __name__ == "__main__":
    #cv2.imread("images/asteroid/Vesta/Vesta_seq000 (1).jpg")
    np.random.seed(1)
    dataset_handler = DawnDatasetHandler()
    
    image = dataset_handler.images[0]
    print(dataset_handler.k)
    print(dataset_handler.num_frames)
    # Use matplotlib to display the images
    _, image_cells = plt.subplots(1, 1, figsize=(20, 20))
    image_cells.imshow(image)

    i = 0
    images = dataset_handler.images
    kp_list, des_list = kp.extract_features_dataset(images, kp.extract_features)
    kp1 = kp_list[i]
    kp2 = kp_list[i + 1]

    unfiltered_matches = kp.match_features_dataset(des_list, kp.match_features)
    print("Number of features matched in frames {0} and {1}: {2}".format(i, i + 1, len(unfiltered_matches[i])))
    dist_threshold = 0.6
    matches = kp.filter_matches_dataset(kp.filter_matches_distance, unfiltered_matches, dist_threshold)
    print("Number of filtered features matched in frames {0} and {1}: {2}".format(i, i + 1, len(matches[i])))
    image_matches = kp.visualize_matches(image, kp1, dataset_handler.images[i+1], kp2, matches[i][:20])
    plt.show()

    k = dataset_handler.k
    trajectory = odo.estimate_trajectory(odo.estimate_motion, unfiltered_matches, kp_list, k)
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot3D(trajectory[0, :], trajectory[2, :], trajectory[1, :], 'gray')
    ax.grid()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    odo.set_axes_equal(ax)

    anim = odo.visualize_camera_movement_animation(images, matches, kp_list, dataset_handler.k)

    plt.show()