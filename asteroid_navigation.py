import numpy as np
import cv2
from matplotlib import pyplot as plt
from file_management import *
import time
import math
import keypoints as kp
import odometry as odo
from matplotlib import animation

if __name__ == "__main__":
    #cv2.imread("images/asteroid/Ceres/Vesta_seq000 (1).jpg")
    np.random.seed(1)
    dataset_handler = DawnDatasetHandler()
    """
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    # Take first frame and find corners in it
    old_frame = dataset_handler.images[0]
    old_gray = old_frame
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    im_i = 1
    while (1):
        frame = dataset_handler.images[im_i]
        if im_i != dataset_handler.num_frames - 1:
            im_i = im_i + 1
        else:
            cv2.destroyAllWindows()
            old_frame = dataset_handler.images[0]
            old_gray = old_frame
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

            # Create a mask image for drawing purposes
            mask = np.zeros_like(old_frame)
            im_i = 1
        frame_gray = frame
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        img = cv2.add(frame, mask)
        cv2.imshow('frame', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
        time.sleep(0.1)
    cv2.destroyAllWindows()
    """
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
    f = r"images/Ceres_keypoints.gif"
    writergif = animation.PillowWriter(fps=5)
    anim.save(f, writer=writergif)

    plt.show()
