import numpy as np
import cv2 as cv
from pathlib import Path
import hashlib
import random
import matplotlib.pyplot as plt

# Constant variables
IMG_NAME = "original.png"
GROUND_TRUTH_NAME = "ground_truth.png"
CURRENT_DIR = Path.cwd()
RESULTS_DIR = CURRENT_DIR / "hw2-results" # Directory to store project results
PART2_DIR = RESULTS_DIR / "part2" # Directory to store results from part 2
PART3_DIR = RESULTS_DIR / "part3" # Directory to store results from part 3
PART4_DIR = RESULTS_DIR / "part4" # Directory to store results from part 3
PLOTS_DIR = RESULTS_DIR / "plots" # Directory to store plots from part 3
CACHE_DIR = CURRENT_DIR / ".cache-hw2" # Directory to store cached information about the image to reduce script rerun time

# Create the directories in the current directory if the do not exist
RESULTS_DIR.mkdir(exist_ok=True)
PART2_DIR.mkdir(exist_ok=True)
PART3_DIR.mkdir(exist_ok=True)
PART4_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# Helper functions

# Create file hashes for the cache for uniqueness
def file_hash(path):
    with open(path, "rb") as file:
        return hashlib.md5(file.read()).hexdigest()

# Writes files to the dirs
def write_file(path, img):
    cv.imwrite(path, img)

# Converts images to RGB
def to_rgb_display(img):
    if len(img.shape) == 2:
        return cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)

print("==================== Part 2: Image Preprocessing & Multi-Channel Normalization ==================== ")

print("\n Reading in base image file")
img = cv.imread(IMG_NAME)
assert img is not None, f'Image file {IMG_NAME} could not be read, check file path.'

print("\n 1. Split the image into its three color channels, apply Histogram Equalization, and merge the channels back together")

# General outline for each of the three equalizations below:
# The image is converted to the proper color space as necessary
# The respective cache is checked to see if the image loaded has been split
# If it has, read the values from the cache
# If it has not, split the image into its channels and save the values to the cache
# This is done to try to save some time of retries and while debugging. I also read that the split method can be inefficient/take a lot of time, so this is an effort to optimize a little
# The proper channel(s) is(are) equalized and then all of the channels are merged back together
# The image is then converted back to BRG as necessary
# The image is saved to results/part2/

# # ==============  BRG Equalization from orig img ==============
# # Not used: Equalized image is not as clear as the HSV image.

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

# # ==============  CIELab conversion from orig img ==============
# # Not used: Resulted in very dull colors with little contrast

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

# Convert to greyscale
grey_img = cv.cvtColor(norm_hsv_brg, cv.COLOR_BGR2GRAY)
# blur = cv.GaussianBlur(grey_img,(5,5),.5) # Applying blur did not improve final IoU or DICE metrics

print("\n1. Otsu's Global Thresholding")

otsu_threshold, otsu_mask = cv.threshold(
    grey_img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU
)

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (35,35))
clean_otsu_mask = cv.morphologyEx(otsu_mask, cv.MORPH_OPEN, kernel, iterations=1)
clean_otsu_mask = cv.morphologyEx(clean_otsu_mask, cv.MORPH_CLOSE, kernel, iterations=2)

otsu_foreground_img = cv.bitwise_and(norm_hsv_brg, norm_hsv_brg, mask=clean_otsu_mask)

otsu_mask_file = "otsu_mask.png"
otsu_foreground = "otsu_foreground.png"
write_file(PART3_DIR / otsu_mask_file, clean_otsu_mask)
write_file(PART3_DIR / otsu_foreground, otsu_foreground_img)

print("\n2. Adaptive Thresholding")

adapt_thresholded_mask = cv.adaptiveThreshold(
    grey_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,1401,50
)

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
adapt_thresholded_mask_clean = cv.morphologyEx(adapt_thresholded_mask, cv.MORPH_OPEN, kernel, iterations=1)
adapt_thresholded_mask_clean = cv.morphologyEx(adapt_thresholded_mask_clean, cv.MORPH_CLOSE, kernel, iterations=30)

adapt_thresh_foreground_img = cv.bitwise_and(norm_hsv_brg, norm_hsv_brg, mask=adapt_thresholded_mask_clean)

adapt_mask = "adapt_mask.png"
adapt_foreground = "adapt_foreground.png"
write_file(PART3_DIR / adapt_mask, adapt_thresholded_mask_clean)
write_file(PART3_DIR / adapt_foreground, adapt_thresh_foreground_img)

print("\n==================== Part 4: Classical and Optimization-Based Segmentation ==================== ")

print("\n Color Space Clustering (K-Means)")

Z = norm_hsv_brg.reshape((-1,3))
Z = np.float32(Z)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 10.0)
K = 5
ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)

img_h, img_w = norm_hsv_brg.shape[:2]
labels_img = label.reshape((img_h, img_w))

# # Run for cluster masking evaluation to determine best cluster to use
# for k in range(K):
#     cluster_mask = (labels_img == k).astype(np.uint8) * 255
#     write_file(PART4_DIR / f"k4_cluster_{k}_mask.png", cluster_mask)

cluster = (labels_img == 0)
cluster_mask = np.uint8(cluster) * 255

kmeans_foreground_img = cv.bitwise_and(norm_hsv_brg, norm_hsv_brg, mask=cluster_mask)

kmeans_mask = "kmeans_mask.png"
kmeans_foreground = "kmeans_foreground.png"
write_file(PART4_DIR / kmeans_mask, cluster_mask)
write_file(PART4_DIR / kmeans_foreground, kmeans_foreground_img)

print("\n==================== Part 5: Evaluation and Analysis ==================== ")

print("\n 1. Qualitative Analysis:\n    See README for discussion")

print("\n 2. Quantitative Comparison")

masks = {
    "Otsu's": clean_otsu_mask,
    "Adaptive Thresholding": adapt_thresholded_mask_clean,
    "K-Means Clustering": cluster_mask}

ground_truth_img = cv.imread(GROUND_TRUTH_NAME)
ground_truth_grey = cv.cvtColor(ground_truth_img, cv.COLOR_BGR2GRAY)
_, ground_truth_binary = cv.threshold(ground_truth_grey, 127, 255, cv.THRESH_BINARY)

metric_results = {}
for name, mask in masks.items():
    print(f"\n    {name}:")
    _, calculated_mask_binary = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)

    intersection = cv.bitwise_and(ground_truth_binary, calculated_mask_binary)
    union = cv.bitwise_or(ground_truth_binary, calculated_mask_binary)

    intersection_area = cv.countNonZero(intersection)
    union_area = cv.countNonZero(union)

    if union_area == 0:
        iou = 1.0 if (intersection_area == 0) else 0.0
    else:
        iou = (intersection_area / union_area)
    metric_results[f'{name[0]}_iou'] = iou

    print(f"\n        IoU: {iou}")

    dice_num = 2 * intersection_area
    dice_denom = cv.countNonZero(ground_truth_binary) + cv.countNonZero(calculated_mask_binary)

    if dice_denom == 0:
        dice = 1.0 if (dice_num == 0) else 0.0
    else:
        dice = dice_num / dice_denom
    metric_results[f'{name[0]}_dice'] = dice

    print(f"\n        Dice Coefficient: {dice}")

# Plotting
print("\nCreating a summary plot...")

fig = plt.figure(figsize=(20, 10))
fig.patch.set_facecolor("#1a1a2e")
gs = fig.add_gridspec(2, 4)

ax_top_left = fig.add_subplot(gs[0, 1])
ax_top_right = fig.add_subplot(gs[0, 2])
ax_bottom_left = fig.add_subplot(gs[1, 0])
ax_bottom_center_left = fig.add_subplot(gs[1, 1])
ax_bottom_center_right = fig.add_subplot(gs[1, 2])
ax_bottom_right = fig.add_subplot(gs[1, 3])

ax_top_left.imshow(to_rgb_display(img))
ax_top_left.set_title("Original", fontsize=12, pad=10, color="white",)
ax_top_left.axis("off")

ax_top_right.imshow(to_rgb_display(norm_hsv_brg))
ax_top_right.set_title("Normalized", fontsize=12, pad=10, color="white",)
ax_top_right.axis("off")

ax_bottom_left.imshow(to_rgb_display(ground_truth_binary))
ax_bottom_left.set_title("Ground Truth Mask", fontsize=12, pad=10, color="white",)
ax_bottom_left.axis("off")

ax_bottom_center_left.imshow(to_rgb_display(clean_otsu_mask))
ax_bottom_center_left.set_title(f'Otsu\'s Mask\nIoU: {metric_results["O_iou"]:.2f}  Dice: {metric_results["O_dice"]:.2f}', fontsize=12, pad=10, color="white",)
ax_bottom_center_left.axis("off")

ax_bottom_center_right.imshow(to_rgb_display(adapt_thresholded_mask_clean))
ax_bottom_center_right.set_title(f'Adaptive Thresholding Mask\nIoU: {metric_results["A_iou"]:.2f}  Dice: {metric_results["A_dice"]:.2f}', fontsize=12, pad=10, color="white",)
ax_bottom_center_right.axis("off")

ax_bottom_right.imshow(to_rgb_display(cluster_mask))
ax_bottom_right.set_title(f'K-Means Clustering Mask\nIoU: {metric_results["K_iou"]:.2f}  Dice: {metric_results["K_dice"]:.2f}', fontsize=12, pad=10, color="white",)
ax_bottom_right.axis("off")

# Multi-line figure title
fig.suptitle(
    "Image Segmentation and Masking Summary\n",
    fontsize=18,
    color="white",
    y=0.95
)

# Adjust spacing
plt.tight_layout(rect=[0, 0, 1, 0.90])

plt.savefig(PLOTS_DIR / f"hw2_plot.png", facecolor=fig.get_facecolor())
plt.close()

print("\n==================== Script Complete ====================\n")