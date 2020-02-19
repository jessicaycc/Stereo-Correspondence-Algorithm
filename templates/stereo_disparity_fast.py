import numpy as np
from numpy.linalg import inv
from scipy.ndimage.filters import *

def stereo_disparity_fast(Il, Ir, bbox, maxd):
    """
    Fast stereo correspondence algorithm.

    This function computes a stereo disparity image from left stereo 
    image Il and right stereo image Ir. Only disparity values within
    the bounding box region are evaluated.

    Parameters:
    -----------
    Il    - Left stereo image, m x n pixel np.array, greyscale.
    Ir    - Right stereo image, m x n pixel np.array, greyscale.
    bbox  - 2x2 np.array, bounding box, relative to left image, from top left
            corner to bottom right corner (inclusive).
    maxd  - Integer, maximum disparity value; disparities must be within zero
            to maxd inclusive (i.e., don't search beyond rng)

    Returns:
    --------
    Id  - Disparity image (map) as np.array, same size as Il, greyscale.
    """
    # Hints:
    #
    #  - Loop over each image row, computing the local similarity measure, then
    #    aggregate. At the border, you may replicate edge pixels, or just avoid
    #    using values outside of the image.
    #
    #  - You may hard-code any parameters you require in this function.
    #
    #  - Use whatever window size you think might be suitable.
    #
    #  - Don't optimize for runtime (too much), optimize for clarity.

    #--- FILL ME IN ---

    # Your code goes here.
    window = 7
    half_window = int(window/2) 
    sad = np.zeros((1, 2*maxd))
    offset = np.zeros((1, 2*maxd))
    Id = np.zeros(np.shape(Il)) 
    # pad the left and right images 
    Il_padded = np.pad(Il, half_window, mode = 'edge')
    Ir_padded = np.pad(Ir, half_window, mode = 'edge')
    _, padded_width = np.shape(Il_padded)

    # loop through each pixel within the bounding box
    for i in range (bbox[1, 0], bbox[1, 1] + 1):
        for j in range (bbox[0, 0], bbox[0, 1] + 1):
            left = Il_padded[i : i + window, j : j + window]
            num = 0
            for k in range (-maxd, maxd):
                # bound check.. if out of bounds, ignore by assigning -10 to as the sad score
                if (j + k + half_window) < (padded_width - half_window) and (j + k + half_window) > (half_window - 1) :
                    right = Ir_padded[i : i + window, (j + k) : (j + k) + window]
                    sum_diff = np.sum(np.abs(left - right))
                    sad[0, num] = sum_diff #value 
                    offset[0, num] = abs(k) #offset
                else: 
                    sad[0, num] = -10
                    offset[0, num] = abs(k)
                num += 1
            s = sad[0]
            s[s < 0] = np.amax(s) #change 0 to max number 
            best_match = np.argmin(s) #find the min 
            Id[i,j] = offset[0, best_match] # assign disparity to Id
    #------------------
    return Id
