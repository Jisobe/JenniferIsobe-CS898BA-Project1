import numpy as np
import cv2 as cv
from pathlib import Path
import hashlib
from scipy import stats
import random
import matplotlib.pyplot as plt

# Constant variables
IMG_NAME = "original.png"
CURRENT_DIR = Path.cwd()
RESULTS_DIR = CURRENT_DIR / "hw2-results" # Directory to store project results
PART2_DIR = RESULTS_DIR / "part2" # Directory to store results from part 2
PART3_DIR = RESULTS_DIR / "part3" # Directory to store results from part 3
# PLOTS_DIR = PART3_DIR / "plots" # Directory to store plots from part 3
# README_PLOTS_DIR = PART3_DIR / "readme_plots" # Directory to store plots from part 3
CACHE_DIR = CURRENT_DIR / ".cache-hw2" # Directory to store cached information about the image to reduce script rerun time

# Create the directories in the current directory if the do not exist
RESULTS_DIR.mkdir(exist_ok=True)
PART2_DIR.mkdir(exist_ok=True)
PART3_DIR.mkdir(exist_ok=True)
# PLOTS_DIR.mkdir(exist_ok=True)
# README_PLOTS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# Helper functions
SEED = 42 # To look at different result sets, change this number
random.seed(SEED)

# Create file hashes for the cache for uniqueness
def file_hash(path):
    with open(path, "rb") as file:
        return hashlib.md5(file.read()).hexdigest()

# Writes files to the dirs
def write_file(path, img):
    cv.imwrite(path, img)

# Used as a helper to check the number of files at various steps
def count_files(path):
    file_count = sum(1 for item in path.iterdir() if item.is_file())
    print(f'Total files in {path.name}: {file_count}')

# Converts images to RGB
def to_rgb_display(img):
    if len(img.shape) == 2:
        return cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)

# Dictionary to store the images and statistics for the plots
all_images = {}

print("==================== Part 2: Image Preprocessing & Multi-Channel Normalization ==================== ")

print("\n Reading in base image file")
img = cv.imread(IMG_NAME)
assert img is not None, f'Image file {IMG_NAME} could not be read, check file path.'

print("\n 1. Split the image into its three color channels, apply Histogram Equalization, and merge the channels back together")

# ==============  BRG Equalization from orig img ==============
# Not used: Equalized image is not as clear as the HSV image.

# norm_brg_file = "norm_brg_img.png"

# bgr_channels_cache = CACHE_DIR / f'bgr_channel_cache_{file_hash(IMG_NAME)}.npz'

# if bgr_channels_cache.is_file():
#     print(f'\n Retrieving channel data from cache')
#     channels = np.load(bgr_channels_cache)
#     b, g, r = channels["b"], channels["g"], channels["r"]
# else:
#     print("\n Extracting channel data from the original image")
#     b, g, r = cv.split(img)
#     np.savez(bgr_channels_cache, b=b, g=g, r=r)

# # Histogram Eq on b,r, and g channels and merge
# norm_brg = cv.merge([cv.equalizeHist(b), cv.equalizeHist(r), cv.equalizeHist(g)])

# write_file(PART2_DIR / norm_brg_file, norm_brg)

# ==============  HSV conversion from orig img ==============
# Provides good contrast with more vibrant colors that will be easier to process later

hsv_file = "hsv_img.png"
norm_hsv_file = "norm_hsv_img.png"
hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
write_file(PART2_DIR / hsv_file, hsv_img)

hsv_channels_cache = CACHE_DIR / f'hsv_channel_cache_{file_hash(PART2_DIR / hsv_file)}.npz'

if hsv_channels_cache.is_file():
    print(f'\n Retrieving channel data from cache')
    channels = np.load(hsv_channels_cache)
    h, s, v = channels["h"], channels["s"], channels["v"]
else:
    print("\n Extracting channel data from the original image")
    h, s, v = cv.split(hsv_img)
    np.savez(hsv_channels_cache, h=h, s=s, v=v)

norm_hsv = cv.merge([h, s, cv.equalizeHist(v)])
norm_hsv_brg = cv.cvtColor(norm_hsv, cv.COLOR_HSV2BGR)

write_file(PART2_DIR / norm_hsv_file, norm_hsv_brg)

# ==============  CIELab conversion from orig img ==============
# Not used: Resulted in very dull colors with little contrast

# cielab_file = "cielab_img.png"
# norm_cielab_file = "norm_cielab_img.png"
# cielab_img = cv.cvtColor(img, cv.COLOR_BGR2Lab)
# write_file(PART2_DIR / cielab_file, cielab_img)

# lab_channels_cache = CACHE_DIR / f'lab_channel_cache_{file_hash(PART2_DIR / cielab_file)}.npz'

# if lab_channels_cache.is_file():
#     print(f'\n Retrieving channel data from cache')
#     channels = np.load(lab_channels_cache)
#     l, a, Bch = channels["l"], channels["a"], channels["Bch"]
# else:
#     print("\n Extracting channel data from the original image")
#     l, a, Bch = cv.split(cielab_img)
#     np.savez(lab_channels_cache, l=l, a=a, Bch=Bch)

# norm_lab = cv.merge([cv.equalizeHist(l), a, Bch])
# norm_lab_brg = cv.cvtColor(norm_lab, cv.COLOR_Lab2BGR)

# write_file(PART2_DIR / norm_cielab_file, norm_lab_brg)

print("\n==================== Part 3: Threshold Based Segmentation ==================== ")

print("\n1. Otsu's Global Thresholding")