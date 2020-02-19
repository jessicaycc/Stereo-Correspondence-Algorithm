import numpy as np
import matplotlib.pyplot as plt
from mat4py import loadmat
from imageio import imread
from stereo_disparity_fast import stereo_disparity_fast
from stereo_disparity_score import stereo_disparity_score

# Load the stereo images.
Il = imread("cones_image_02.png", as_gray=True)
Ir = imread("cones_image_06.png", as_gray=True)
It = imread("cones_disp_02.png", as_gray=True)

# Load the appropriate bounding box.
bboxes = loadmat("bboxes.mat")
bbox = np.array(bboxes["cones_02"]["bbox"])

Id = stereo_disparity_fast(Il, Ir, bbox, 52)
N, rms, p_bad = stereo_disparity_score(It, Id, bbox)
print (N, rms, p_bad)
plt.imshow(Id, cmap = "gray")
plt.show()

