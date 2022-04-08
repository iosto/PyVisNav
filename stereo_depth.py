import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import patches
import file_management as fm


def compute_left_disparity_map(img_left, img_right):
    # Parameters
    num_disparities = 6 * 16
    block_size = 11

    min_disparity = 0
    window_size = 6

    img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # Stereo BM matcher
    left_matcher_BM = cv2.StereoBM_create(
        numDisparities=num_disparities,
        blockSize=block_size
    )

    # Stereo SGBM matcher
    left_matcher_SGBM = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # Compute the left disparity map
    disp_left = left_matcher_SGBM.compute(img_left, img_right).astype(np.float32) / 16

    return disp_left


def decompose_projection_matrix(p):
    k, r, t, _, _, _, _ = cv2.decomposeProjectionMatrix(p)
    t = t / t[3]
    return k, r, t


def calc_depth_map(disp_left, k_left, t_left, t_right):
    # Get the focal length from the K matrix
    f = k_left[0, 0]

    # Get the distance between the cameras from the t matrices (baseline)
    b = t_left[1] - t_right[1]

    # Replace all instances of 0 and -1 disparity with a small minimum value (to avoid div by 0 or negatives)
    disp_left[disp_left == 0] = 0.1
    disp_left[disp_left == -1] = 0.1

    # Initialize the depth map to match the size of the disparity map
    depth_map = np.ones(disp_left.shape, np.single)

    # Calculate the depths
    depth_map[:] = f * b / disp_left[:]

    return depth_map


def locate_obstacle_in_image(image, obstacle_image):
    # Run the template matching from OpenCV
    cross_corr_map = cv2.matchTemplate(image, obstacle_image, method=cv2.TM_CCOEFF)

    # Locate the position of the obstacle using the minMaxLoc function from OpenCV
    _, _, _, obstacle_location = cv2.minMaxLoc(cross_corr_map)

    return cross_corr_map, obstacle_location


def calculate_nearest_point(depth_map, obstacle_location, obstacle_img):
    # Gather the relative parameters of the obstacle box
    obstacle_width = obstacle_img.shape[0]
    obstacle_height = obstacle_img.shape[1]
    obstacle_min_x_pos = obstacle_location[1]
    obstacle_max_x_pos = obstacle_location[1] + obstacle_width
    obstacle_min_y_pos = obstacle_location[0]
    obstacle_max_y_pos = obstacle_location[0] + obstacle_height

    # Get the depth of the pixels within the bounds of the obstacle image, find the closest point in this rectangle
    obstacle_depth = depth_map[obstacle_min_x_pos:obstacle_max_x_pos, obstacle_min_y_pos:obstacle_max_y_pos]
    closest_point_depth = obstacle_depth.min()

    # Create the obstacle bounding box
    obstacle_bbox = patches.Rectangle((obstacle_min_y_pos, obstacle_min_x_pos), obstacle_height, obstacle_width,
                                      linewidth=1, edgecolor='r', facecolor='none')

    return closest_point_depth, obstacle_bbox


if __name__ == "__main__":
    # Read the stereo-pair of images
    img_left = fm.read_left_image()
    img_right = fm.read_right_image()

    # Use matplotlib to display the two images
    _, image_cells = plt.subplots(1, 2, figsize=(20, 20))
    image_cells[0].imshow(img_left)
    image_cells[0].set_title('left image')
    image_cells[1].imshow(img_right)
    image_cells[1].set_title('right image')
    plt.show()

    # Read the calibration
    p_left, p_right = fm.get_projection_matrices()

    # Use regular numpy notation instead of scientific one
    np.set_printoptions(suppress=True)

    print("p_left \n", p_left)
    print("\np_right \n", p_right)

    # Compute the disparity map using the fuction above
    disp_left = compute_left_disparity_map(img_left, img_right)

    # Show the left disparity map
    plt.figure(figsize=(10, 10))
    plt.imshow(disp_left)
    plt.show()

    # Decompose each matrix
    k_left, r_left, t_left = decompose_projection_matrix(p_left)
    k_right, r_right, t_right = decompose_projection_matrix(p_right)

    # Display the matrices
    print("k_left \n", k_left)
    print("\nr_left \n", r_left)
    print("\nt_left \n", t_left)
    print("\nk_right \n", k_right)
    print("\nr_right \n", r_right)
    print("\nt_right \n", t_right)

    # Get the depth map by calling the above function
    depth_map_left = calc_depth_map(disp_left, k_left, t_left, t_right)

    # Display the depth map
    plt.figure(figsize=(8, 8), dpi=100)
    plt.imshow(depth_map_left, cmap='flag')
    plt.show()

    # Get the image of the obstacle
    obstacle_image = fm.get_obstacle_image()

    # Show the obstacle image
    plt.figure(figsize=(4, 4))
    plt.imshow(obstacle_image)
    plt.show()

    # Gather the cross correlation map and the obstacle location in the image
    cross_corr_map, obstacle_location = locate_obstacle_in_image(img_left, obstacle_image)

    # Display the cross correlation heatmap
    plt.figure(figsize=(10, 10))
    plt.imshow(cross_corr_map)
    plt.show()

    # Print the obstacle location
    print("obstacle_location \n", obstacle_location)

    # Use the developed nearest point function to get the closest point depth and obstacle bounding box
    closest_point_depth, obstacle_bbox = calculate_nearest_point(depth_map_left, obstacle_location, obstacle_image)

    # Display the image with the bounding box displayed
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img_left)
    ax.add_patch(obstacle_bbox)
    plt.show()

    # Print the depth of the nearest point
    print("closest_point_depth {0:0.3f}".format(closest_point_depth))