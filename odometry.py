import numpy as np
import cv2
from matplotlib import pyplot as plt
from file_management import *
from keypoints import *
import math


def visualize_trajectory(trajectory):
    # Unpack X Y Z each trajectory point
    locX = []
    locY = []
    locZ = []
    # This values are required for keeping equal scale on each plot.
    # matplotlib equal axis may be somewhat confusing in some situations because of its various scale on
    # different axis on multiple plots
    max = -math.inf
    min = math.inf

    # Needed for better visualisation
    maxY = -math.inf
    minY = math.inf

    for i in range(0, trajectory.shape[1]):
        current_pos = trajectory[:, i]

        locX.append(current_pos.item(0))
        locY.append(current_pos.item(1))
        locZ.append(current_pos.item(2))
        if np.amax(current_pos) > max:
            max = np.amax(current_pos)
        if np.amin(current_pos) < min:
            min = np.amin(current_pos)

        if current_pos.item(1) > maxY:
            maxY = current_pos.item(1)
        if current_pos.item(1) < minY:
            minY = current_pos.item(1)

    auxY_line = locY[0] + locY[-1]
    if max > 0 and min > 0:
        minY = auxY_line - (max - min) / 2
        maxY = auxY_line + (max - min) / 2
    elif max < 0 and min < 0:
        minY = auxY_line + (min - max) / 2
        maxY = auxY_line - (min - max) / 2
    else:
        minY = auxY_line - (max - min) / 2
        maxY = auxY_line + (max - min) / 2

    # Set styles
    mpl.rc("figure", facecolor="white")
    plt.style.use("seaborn-whitegrid")

    # Plot the figure
    fig = plt.figure(figsize=(8, 6), dpi=100)
    gspec = gridspec.GridSpec(3, 3)
    ZY_plt = plt.subplot(gspec[0, 1:])
    YX_plt = plt.subplot(gspec[1:, 0])
    traj_main_plt = plt.subplot(gspec[1:, 1:])
    D3_plt = plt.subplot(gspec[0, 0], projection='3d')

    # Actual trajectory plotting ZX
    toffset = 1.06
    traj_main_plt.set_title("Autonomous vehicle trajectory (Z, X)", y=toffset)
    traj_main_plt.set_title("Trajectory (Z, X)", y=1)
    traj_main_plt.plot(locZ, locX, ".-", label="Trajectory", zorder=1, linewidth=1, markersize=4)
    traj_main_plt.set_xlabel("Z")
    # traj_main_plt.axes.yaxis.set_ticklabels([])
    # Plot reference lines
    traj_main_plt.plot([locZ[0], locZ[-1]], [locX[0], locX[-1]], "--", label="Auxiliary line", zorder=0, linewidth=1)
    # Plot camera initial location
    traj_main_plt.scatter([0], [0], s=8, c="red", label="Start location", zorder=2)
    traj_main_plt.set_xlim([min, max])
    traj_main_plt.set_ylim([min, max])
    traj_main_plt.legend(loc=1, title="Legend", borderaxespad=0., fontsize="medium", frameon=True)

    # Plot ZY
    # ZY_plt.set_title("Z Y", y=toffset)
    ZY_plt.set_ylabel("Y", labelpad=-4)
    ZY_plt.axes.xaxis.set_ticklabels([])
    ZY_plt.plot(locZ, locY, ".-", linewidth=1, markersize=4, zorder=0)
    ZY_plt.plot([locZ[0], locZ[-1]], [(locY[0] + locY[-1]) / 2, (locY[0] + locY[-1]) / 2], "--", linewidth=1, zorder=1)
    ZY_plt.scatter([0], [0], s=8, c="red", label="Start location", zorder=2)
    ZY_plt.set_xlim([min, max])
    ZY_plt.set_ylim([minY, maxY])

    # Plot YX
    # YX_plt.set_title("Y X", y=toffset)
    YX_plt.set_ylabel("X")
    YX_plt.set_xlabel("Y")
    YX_plt.plot(locY, locX, ".-", linewidth=1, markersize=4, zorder=0)
    YX_plt.plot([(locY[0] + locY[-1]) / 2, (locY[0] + locY[-1]) / 2], [locX[0], locX[-1]], "--", linewidth=1, zorder=1)
    YX_plt.scatter([0], [0], s=8, c="red", label="Start location", zorder=2)
    YX_plt.set_xlim([minY, maxY])
    YX_plt.set_ylim([min, max])

    # Plot 3D
    D3_plt.set_title("3D trajectory", y=toffset)
    D3_plt.plot3D(locX, locZ, locY, zorder=0)
    D3_plt.scatter(0, 0, 0, s=8, c="red", zorder=1)
    D3_plt.set_xlim3d(min, max)
    D3_plt.set_ylim3d(min, max)
    D3_plt.set_zlim3d(min, max)
    D3_plt.tick_params(direction='out', pad=-2)
    D3_plt.set_xlabel("X", labelpad=0)
    D3_plt.set_ylabel("Z", labelpad=0)
    D3_plt.set_zlabel("Y", labelpad=-2)

    # plt.axis('equal')
    D3_plt.view_init(45, azim=30)
    plt.tight_layout()
    plt.show()


def estimate_motion(match_feat, kp_1, kp_2, k):
    """
    Estimate camera motion from a pair of subsequent image frames

    Arguments:
    match_feat -- list of matched features from the pair of images
    kp_1 -- list of the keypoints in the first image
    kp_2 -- list of the keypoints in the second image
    k -- camera calibration matrix

    Returns:
    rmat -- recovered 3x3 rotation numpy matrix
    tvec -- recovered 3x1 translation numpy vector
    image1_points -- a list of selected match coordinates in the first image. image1_points[i] = [u, v], where u and v are
                     coordinates of the i-th match in the image coordinate system
    image2_points -- a list of selected match coordinates in the second image. image1_points[i] = [u, v], where u and v are
                     coordinates of the i-th match in the image coordinate system

    """
    rmat = np.eye(3)
    tvec = np.zeros((3, 1))
    image1_points = []
    image2_points = []
    if len(match_feat) > 3:
        for points in match_feat:
            try:
                image1_points.append(np.array(kp_1[points.queryIdx].pt))
                image2_points.append(np.array(kp_2[points.trainIdx].pt))
            except AttributeError:
                try:
                    image1_points.append(np.array(kp_1[points[0].queryIdx].pt))
                    image2_points.append(np.array(kp_2[points[0].trainIdx].pt))
                except IndexError:
                    print("Empty match.")

        image1_points, image2_points = np.array(image1_points, np.float32), np.array(image2_points, np.float32)
        E, mask = cv2.findEssentialMat(image1_points, image2_points, k)
        points, rmat, tvec, mask_pose = cv2.recoverPose(E, image1_points, image2_points)
    else:
        print("Not enough points in match list for estimating motion.")

    return rmat, tvec, image1_points, image2_points


def visualize_camera_movement(image1, image1_points, image2, image2_points, is_show_img_after_move=False):
    image1 = image1.copy()
    image2 = image2.copy()

    for i in range(0, len(image1_points)):
        # Coordinates of a point on t frame
        p1 = (int(image1_points[i][0]), int(image1_points[i][1]))
        # Coordinates of the same point on t+1 frame
        p2 = (int(image2_points[i][0]), int(image2_points[i][1]))

        cv2.circle(image1, p1, 5, (0, 255, 0), 1)
        cv2.arrowedLine(image1, p1, p2, (0, 255, 0), 1)
        cv2.circle(image1, p2, 5, (255, 0, 0), 1)

        if is_show_img_after_move:
            cv2.circle(image2, p2, 5, (255, 0, 0), 1)

    if is_show_img_after_move:
        return image2
    else:
        return image1


def visualize_camera_movement_animation(images_f, matches_f, kp_list_f, k):
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ims = []
    for i in range(len(matches_f)):
        match = matches_f[i]
        kp1 = kp_list_f[i]
        kp2 = kp_list_f[i+1]
        rmat, tvec, image_1_points, image_2_points = estimate_motion(match, kp1, kp2, k)
        image1 = images_f[i]
        image2 = images_f[i+1]
        image_move = visualize_camera_movement(image1, image_1_points, image2, image_2_points)
        im = plt.imshow(cv2.cvtColor(image_move, cv2.COLOR_RGB2BGR), animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=1000)
    return ani


def estimate_trajectory(estimate_motion, matches_f, kp_list, k, depth_maps=[]):
    """
    Estimate complete camera trajectory from subsequent image pairs

    Arguments:
    estimate_motion -- a function which estimates camera motion from a pair of subsequent image frames
    matches -- list of matches for each subsequent image pair in the dataset.
               Each matches[i] is a list of matched features from images i and i + 1
    kp_list -- a list of keypoints for each image in the dataset
    k -- camera calibration matrix

    Optional arguments:
    depth_maps -- a list of depth maps for each frame. This argument is not needed if you use Essential Matrix Decomposition

    Returns:
    trajectory -- a 3xlen numpy array of the camera locations, where len is the lenght of the list of images and
                  trajectory[:, i] is a 3x1 numpy vector, such as:

                  trajectory[:, i][0] - is X coordinate of the i-th location
                  trajectory[:, i][1] - is Y coordinate of the i-th location
                  trajectory[:, i][2] - is Z coordinate of the i-th location

                  * Consider that the origin of your trajectory cordinate system is located at the camera position
                  when the first image (the one with index 0) was taken. The first camera location (index = 0) is geven
                  at the initialization of this function

    """

    # Create variables for computation
    trajectory = np.zeros((3, len(matches_f) + 1))
    robot_pose = np.zeros((len(matches_f) + 1, 4, 4))

    # Initialize camera pose
    robot_pose[0] = np.eye(4)

    # Iterate through the matched features
    for i in range(len(matches_f)):
        # Estimate camera motion between a pair of images
        rmat, tvec, image1_points, image2_points = estimate_motion(matches_f[i], kp_list[i], kp_list[i + 1], k)

        # Determine current pose from rotation and translation matrices
        current_pose = np.eye(4)
        current_pose[0:3, 0:3] = rmat
        current_pose[0:3, 3] = tvec.T

        # Build the robot's pose from the initial position by multiplying previous and current poses
        robot_pose[i + 1] = robot_pose[i] @ np.linalg.inv(current_pose)

        # Calculate current camera position from origin
        position = robot_pose[i + 1] @ np.array([0., 0., 0., 1.])

        # Build trajectory
        trajectory[:, i + 1] = position[0:3]

    return trajectory


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


if __name__ == "__main__":
    np.random.seed(1)
    dataset_handler = DatasetHandler()

    i = 0
    images = dataset_handler.images
    kp_list, des_list = extract_features_dataset(images, extract_features)
    kp1 = kp_list[i]
    kp2 = kp_list[i+1]

    unfiltered_matches = match_features_dataset(des_list, match_features)
    dist_threshold = 0.6
    matches = filter_matches_dataset(filter_matches_distance, unfiltered_matches, dist_threshold)

    match = matches[i]
    k = dataset_handler.k
    rmat, tvec, image_1_points, image_2_points = estimate_motion(match, kp1, kp2, k)

    print("Estimated rotation:\n {0}".format(rmat))
    print("Estimated translation:\n {0}".format(tvec))

    image1 = dataset_handler.images_rgb[i]
    image2 = dataset_handler.images_rgb[i + 1]
    image_move = visualize_camera_movement(image1, image_1_points, image2, image_2_points)
    plt.figure(figsize=(16, 12), dpi=100)
    plt.imshow(image_move)

    trajectory = estimate_trajectory(estimate_motion, matches, kp_list, k)
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot3D(trajectory[0,:], trajectory[2,:], trajectory[1,:], 'gray')
    ax.grid()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    set_axes_equal(ax)

    anim = visualize_camera_movement_animation(images, matches, kp_list, dataset_handler.k)

    plt.show()