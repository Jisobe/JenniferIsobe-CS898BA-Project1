# TODO : Create print statements for each step for terminal output
# TODO: Save terminal output to file in results

# TODO: Part 2: You know you should at least do basic analysis to get started, so you perform the following on the image:

#   4.  Convert the normalized image back to RGB and save it.
#   5.  You should now have 7 images.
#   6.  Perform random affine transformations on each image (you should perform 14 total transformations - 2 for each image). Affine transformations can be translation, rotation, scaling, or shear as long as each is unique in either transformation type or transformation value (rotate 90 degrees vs rotate 186 degrees). No two images should be transformed in the exact same way. Save each of those images to new files.
#   7.  You should now have 21 images.
#   8.  Apply a Gaussian blur to each image using the levels of sigma: 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5. Discuss how the level of sigma changes the image. Save each of those images to new files.
#   9.  You should now have 168 images.

# TODO: Part 3: You decide that detecting the edges of the unknown figure would be useful, so you do the following:
#   1.  Randomly create 4 equally sized subsets of the images from part 1.
#     Each subset should have 42 images.
#   2.  Choose a subset to use in the remaining steps.
#   3.  You should now have 42 images.
#   4.  Perform these edge detection techniques on that subset:
#     a.  Sobel
#     b.  Laplacian
#     c.  Canny
#     d.  Prewitt
#   5.  Discuss the pros and cons of each edge detection technique and perform an analysis of which of these techniques works best for this image set.
#     Reminder – Canny may be the most used and applied, but it may not be the best in your case. Make sure your analysis fits your results.
#   6.  Save each image before and after adding edges with each technique.
#   7.  You should now have 210 images.
#   8.  Create 42, 5-image plots of the input image (from the start of part 3) next to the edge-detected images and output 6 random plots to add to the readme. Include information on what processing techniques were used on the images.

import numpy as np
import cv2 as cv
from pathlib import Path
import hashlib
from scipy import stats

# Constant variables
IMG_NAME = "original.png"
CURRENT_DIR = Path.cwd()
RESULTS_DIR = CURRENT_DIR / "results" # Directory to store project results
PART2_DIR = RESULTS_DIR / "part2" # Directory to store results from part 2
PART3_DIR = RESULTS_DIR / "part3" # Directory to store results from part 3
CACHE_DIR = CURRENT_DIR / ".cache" # Directory to store cached information about the image to reduce script rerun time

# Create the directories in the current directory if the do not exist
RESULTS_DIR.mkdir(exist_ok=True)
PART2_DIR.mkdir(exist_ok=True)
PART3_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# Helper functions

def file_hash(path):
    with open(path, "rb") as file:
        return hashlib.md5(file.read()).hexdigest()

# Part 2: You know you should at least do basic analysis to get started, so you perform the following on the image:

print("Read in image file")
img = cv.imread(IMG_NAME) # TODO : Move into if else if not needed in the rest of the file
assert img is not None, f'Image file {IMG_NAME} could not be read, check file path.'


#   1.  Find and print basic image statistics of the original image for each individual channel (min, max, average, median, mode, skew, range, standard deviation, variance)

bgr_channels_cache = CACHE_DIR / f'bgr_channel_cache_{file_hash(IMG_NAME)}.npz'

if bgr_channels_cache.is_file():
    print(f'Retrieving channel data from cache')
    channels = np.load(bgr_channels_cache)
    b, g, r = channels["b"], channels["g"], channels["r"]
else:
    print("Extracting channel data from the original image")
    b, g, r = cv.split(img)
    np.savez(bgr_channels_cache, b=b, g=g, r=r)

brg_channels = {
    "Blue": b,
    "Red": r,
    "Green": g
}

for name, channel in brg_channels.items():
    flattened = channel.flatten()
    histogram = np.bincount(flattened, minlength=256)
    mode = np.argmax(histogram)
    print(f'\nStatistics for the {name} channel:')
    print(f'    Min: {channel.min():.4f}')
    print(f'    Max: {channel.max():.4f}')
    print(f'    Average: {channel.mean():.4f}')
    print(f'    Median: {np.median(channel):.4f}')
    print(f'    Mode: {mode} with count {histogram[mode]}')
    print(f'    Skew: {stats.skew(flattened):.4f}')
    print(f'    Range: {(channel.max() - channel.min()):.4f}')
    print(f'    Standard Deviation: {channel.std():.4f}')
    print(f'    Variance: {channel.var():.4f}')

#   2.  Convert and save the image to greyscale, binary, and different color spaces (HSV, CIELAB, and HLS).

def write_file(path, img):
    if cv.imwrite(path, img):
        print(f'Saved ')

grey_file = "grey_img.png"
grey_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
write_file(PART2_DIR / grey_file, grey_img)

bin_file = "bin_img.png"
threshold_val, bin_img = cv.threshold(grey_img, 127, 255, cv.THRESH_BINARY)
write_file(PART2_DIR / bin_file, bin_img)

hsv_file = "hsv_img.png"
hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
write_file(PART2_DIR / hsv_file, hsv_img)

cielab_file = "cielab_img.png"
cielab_img = cv.cvtColor(img, cv.COLOR_BGR2Lab)
write_file(PART2_DIR / cielab_file, cielab_img)

hls_file = "hls_img.png"
hls_img = cv.cvtColor(img, cv.COLOR_BGR2HLS)
write_file(PART2_DIR / hls_file, hls_img)

#   3.  On the HSV converted image, normalize the lighting by performing histogram equalization across the V (value) channel.

channels_cache = CACHE_DIR / f'channel_cache_{file_hash(IMG_NAME)}.npz'

if channels_cache.is_file():
    print(f'Retrieving channel data from cache')
    channels = np.load(channels_cache)
    b, g, r = channels["b"], channels["g"], channels["r"]
else:
    print("Extracting channel data from the original image")
    b, g, r = cv.split(img)
    np.savez(channels_cache, b=b, g=g, r=r)
h, s, v =