import csv
import numpy as np
import cv2
import os.path
import sys
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D


class DatasetHandler:

    def __init__(self):
        # Define number of frames
        self.num_frames = 52

        # Set up paths
        root_dir_path = os.path.dirname(os.path.realpath(__file__))
        self.image_dir = os.path.join(root_dir_path, 'images/rgb')
        self.depth_dir = os.path.join(root_dir_path, 'images/depth')

        # Set up data holders
        self.images = []
        self.images_rgb = []
        self.depth_maps = []

        self.k = np.array([[640, 0, 640],
                           [0, 480, 480],
                           [0, 0, 1]], dtype=np.float32)

        # Read first frame
        self.read_frame()
        print("\r" + ' ' * 20 + "\r", end='')

    def read_frame(self):
        self._read_depth()
        self._read_image()

    def _read_image(self):
        for i in range(1, self.num_frames + 1):
            zeroes = "0" * (5 - len(str(i)))
            im_name = "{0}/frame_{1}{2}.png".format(self.image_dir, zeroes, str(i))
            self.images.append(cv2.imread(im_name, flags=0))
            self.images_rgb.append(cv2.imread(im_name)[:, :, ::-1])
            print("Data loading: {0}%".format(int((i + self.num_frames) / (self.num_frames * 2 - 1) * 100)), end="\r")

    def _read_depth(self):
        for i in range(1, self.num_frames + 1):
            zeroes = "0" * (5 - len(str(i)))
            depth_name = "{0}/frame_{1}{2}.dat".format(self.depth_dir, zeroes, str(i))
            depth = np.loadtxt(
                depth_name,
                delimiter=',',
                dtype=np.float64) * 1000.0
            self.depth_maps.append(depth)
            print("Data loading: {0}%".format(int(i / (self.num_frames * 2 - 1) * 100)), end="\r")


class DidymosDatasetHandler:

    def __init__(self):
        # Define number of frames
        self.num_frames = 21

        # Set up paths
        root_dir_path = os.path.dirname(os.path.realpath(__file__))
        self.image_dir = os.path.join(root_dir_path, 'images/asteroid/Didymos')

        # Set up data holders
        self.images = []
        pixSize = 1.0035e-5
        f = 0.106/pixSize
        p = 512
        self.k = np.array([[f, 0, p],
                           [0, f, p],
                           [0, 0, 1]], dtype=np.float32)

        # Read first frame
        self.read_frame()
        print("\r" + ' ' * 20 + "\r", end='')

    def read_frame(self):
        self._read_image()

    def _read_image(self):
        for i in range(1, self.num_frames + 1):
            zeroes = "0" * 3
            im_name = "{0}/IMG_PREPRO_{1}{2}.png".format(self.image_dir, zeroes, str(i+408))
            self.images.append(cv2.imread(im_name, flags=0))
            print("Data loading: {0}%".format(int((i + self.num_frames) / (self.num_frames * 2 - 1) * 100)), end="\r")


class DawnDatasetHandler:

    def __init__(self):
        # Define number of frames
        self.num_frames = 30

        # Set up paths
        root_dir_path = os.path.dirname(os.path.realpath(__file__))
        self.image_dir = os.path.join(root_dir_path, 'images/asteroid/Vesta')

        # Set up data holders
        self.images = []
        pixSize = 1.0035e-5
        f = 0.106/pixSize
        p = 512
        self.k = np.array([[f, 0, p],
                           [0, f, p],
                           [0, 0, 1]], dtype=np.float32)

        # Read first frame
        self.read_frame()
        print("\r" + ' ' * 20 + "\r", end='')

    def read_frame(self):
        self._read_image()

    def _read_image(self):
        for i in range(1, self.num_frames + 1):
            zeroes = "0" * 3
            im_name = "{0}/Vesta_seq{1} ({2}).jpg".format(self.image_dir, zeroes, str(i+55))
            self.images.append(cv2.imread(im_name, flags=0))
            print("Data loading: {0}%".format(int((i + self.num_frames) / (self.num_frames * 2 - 1) * 100)), end="\r")


def get_projection_matrices():
    """Frame Calibration Holder
    3x4    p_left, p_right      Camera P matrix. Contains extrinsic and intrinsic parameters.
    """
    p_left = np.array([[640.0,   0.0, 640.0, 2176.0],
                       [  0.0, 480.0, 480.0,  552.0],
                       [  0.0,   0.0,   1.0,    1.4]])
    p_right = np.array([[640.0,   0.0, 640.0, 2176.0],
                       [   0.0, 480.0, 480.0,  792.0],
                       [   0.0,   0.0,   1.0,    1.4]])
    return p_left, p_right


def read_left_image():
    filepath = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(filepath, "images/stereo_set/frame_00077_1547042741L.png")
    img = cv2.imread(image_path)
    if img is None:
        print("Can't load left image, please check the path", file=sys.stderr)
        sys.exit(1)
    return img[...,::-1]


def read_right_image():
    filepath = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(filepath, "images/stereo_set/frame_00077_1547042741R.png")
    img = cv2.imread(image_path)
    if img is None:
        print("Can't load right image, please check the path", file=sys.stderr)
        sys.exit(1)
    return img[..., ::-1]


def get_obstacle_image():
    img_left_colour = read_left_image()
    return img_left_colour[479:509, 547:593, :]