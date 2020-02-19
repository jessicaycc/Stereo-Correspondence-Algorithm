import numpy as np
import matplotlib.pyplot as plt
from mat4py import loadmat
from imageio import imread
from stereo_disparity_best import stereo_disparity_best
from stereo_disparity_score import stereo_disparity_score
from scipy.ndimage.filters import *


# Load the stereo images.
Il = imread("cones_image_02.png", as_gray=True)
Ir = imread("cones_image_06.png", as_gray=True)
It = imread("cones_disp_02.png", as_gray=True)

# Load the appropriate bounding box.
bboxes = loadmat("bboxes.mat")
bbox = np.array(bboxes["cones_02"]["bbox"])

Id = stereo_disparity_best(Il, Ir, bbox, 52)
# Id = np.load("disparity-map.npy")
np.save("disparity-map.npy", Id)

# Id[bbox[1,0]: bbox[1,1]+1, bbox[0,0]:bbox[0,1]+1] = median_filter(Id[bbox[1,0]: bbox[1,1]+1, bbox[0,0]:bbox[0,1]+1], size=13) # add a median filter to filter out noise 
# Id[bbox[1,0]: bbox[1,1]+1, bbox[0,0]:bbox[0,1]+1] = percentile_filter(Id[bbox[1,0]: bbox[1,1]+1, bbox[0,0]:bbox[0,1]+1],40, size=15) # add a median filter to filter out noise 
N, rms, p_bad = stereo_disparity_score(It, Id, bbox)
val = rms + p_bad
print (val)
plt.imshow(Id, cmap = "gray")
plt.show()
