import cv2
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time

# Import groundtruth poses from the KITTI dataset. Poses are 3x4 matrices of the kind [R, T] where R is the rotation matrix
# and T is the translation vector. In the dataset, a single pose is represented as a row with 12 entries, so we need to reshape
# the dataset. Moreover, in the first sequence there are 4541 frames (left and right) and poses.

poses = pd.read_csv('dataset/poses/00.txt', delimiter=' ', header = None)
print("Poses dataset shape: ", poses.shape)

# Reshaping the dataset to visualize ground truth trajectory with a numpy array of 4541 rows containing 3x4 matrices

ground_truth = np.zeros((len(poses), 3, 4))
for i in range(len(poses)):
    ground_truth[i] = np.array(poses.iloc[i]).reshape((3,4))
print("Reshaped dataset: ", ground_truth.shape)

# Now it's time to plot the ground_truth poses. The only part used here is the 4th column that is the T vector that encodes the
# tranlsation.

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.plot(ground_truth[:, :, 3][:, 0], ground_truth[:, :, 3][:, 1], ground_truth[:, :, 3][:, 2])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
#plt.show() # uncomment for showing the true trajectory

# Let's see now the calibration file. Calibration file contains the projection matrices from the 4 cameras (P0 to P3). The projection
# matrix contains extrinsic ([R, T]) and intrinsic (K) parameters. P0 and P1: stereo grayscale cameras. Projection matrix P = K*[R, T].
# The projection matrix map the 3D world coordinates to the pixel coordinates of the left camera (that's because of the rectification)

calib = pd.read_csv('dataset/sequences/00/calib.txt', delimiter = ' ', header = None, index_col = 0) # the first col now contains 'P0:' etc..
P0 = np.array(calib.loc['P0:']).reshape((3,4)) # Projection matrix of left gray camera
P1 = np.array(calib.loc['P1:']).reshape((3,4)) #  Projection matrix of right gray camera
#print(P0)
#print(P1)

# We need now to decompose the projection matrix in intrinsic and extrinsic matrices. Opencv has a function to do that with QR decomposition.
# N.B. the rotation is expressed in Euler angles XYZ order
k1, r1, t1, _, _, _, _  = cv2.decomposeProjectionMatrix(P1)
t1 = (t1/t1[3]).round(4) # Divide the vector by it's 4th component to have homogenous coordinates
#print(k1)
#print(r1)
#print(t1)

# Import the sequence of images
filepath = 'dataset/sequences/00/image_0/' # Seqence of left frames
left_images = os.listdir(filepath)
left_images.sort()
n_frames = len(left_images)
#print(n_frames)

#Display sequence of left images
'''''
fig = plt.figure(figsize = (12,4))
for i in range(n_frames):
    img = cv2.imread(filepath + left_images[i], 0)
    cv2.imshow('Left sequence', img)
    cv2.waitKey(300)

cv2.destroyAllWindows()
'''''
