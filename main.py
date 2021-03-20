import cv2
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# Import groundtruth poses from the KITTI dataset. Poses are 3x4 matrices of the kind [R, T] where R is the rotation matrix
# and T is the translation vector. In the dataset, a single pose is represented as a row with 12 entries, so we need to reshape
# the dataset. Moreover, in the first sequence there are 4531 frames (left and right) and poses.
poses = pd.read_csv('dataset/poses/00.txt', delimiter=' ', header = None)
print("Poses dataset shape: ", poses.shape)
print(poses.head())
