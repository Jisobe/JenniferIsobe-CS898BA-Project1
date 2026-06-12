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
RESULTS_DIR = CURRENT_DIR / "results" # Directory to store project results
PART2_DIR = RESULTS_DIR / "part2" # Directory to store results from part 2
PART3_DIR = RESULTS_DIR / "part3" # Directory to store results from part 3
PLOTS_DIR = PART3_DIR / "plots" # Directory to store plots from part 3
README_PLOTS_DIR = PART3_DIR / "readme_plots" # Directory to store plots from part 3
CACHE_DIR = CURRENT_DIR / ".cache" # Directory to store cached information about the image to reduce script rerun time

# Create the directories in the current directory if the do not exist
RESULTS_DIR.mkdir(exist_ok=True)
PART2_DIR.mkdir(exist_ok=True)
PART3_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)
README_PLOTS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# Helper functions

def file_hash(path):
    with open(path, "rb") as file:
        return hashlib.md5(file.read()).hexdigest()

def write_file(path, img):
    cv.imwrite(path, img)

def count_files(path):
    file_count = sum(1 for item in path.iterdir() if item.is_file())
    print(f'Total files in {path.name}: {file_count}')

all_images = {}

print("==================== Part 2: Basic Analysis ==================== ")

print("Reading in image file")
img = cv.imread(IMG_NAME)
assert img is not None, f'Image file {IMG_NAME} could not be read, check file path.'

print("1.  Find and print min, max, average, median, mode, skew, range, standard deviation, variance of the original image for each individual channel")

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

print("2.  Convert and save the image to greyscale, binary, HSV, CIELAB, and HLS.")

original_file = "original.png"
write_file(PART2_DIR / original_file, img)
all_images["original"] = {
    "name": "Original",
    "image" : img,
    "space": "BRG",
    "angle": 0,
    "scale": 1,
    "tx": None,
    "ty": None,
    "sigma": None}

grey_file = "grey_img.png"
grey_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
write_file(PART2_DIR / grey_file, grey_img)
all_images["grey"] = {
    "name": "Greyscale",
    "image" : grey_img,
    "space": "Greyscale",
    "angle": 0,
    "scale": 1,
    "tx": None,
    "ty": None,
    "sigma": None}

bin_file = "bin_img.png"
threshold_val, bin_img = cv.threshold(grey_img, 127, 255, cv.THRESH_BINARY)
write_file(PART2_DIR / bin_file, bin_img)
all_images["bin"] = {
    "name": "Binary",
    "image" : bin_img,
    "space": "Binary",
    "angle": 0,
    "scale": 1,
    "tx": None,
    "ty": None,
    "sigma": None}

hsv_file = "hsv_img.png"
hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
write_file(PART2_DIR / hsv_file, hsv_img)
all_images["hsv"] = {
    "name": "HSV",
    "image" : hsv_img,
    "space": "HSV",
    "angle": 0,
    "scale": 1,
    "tx": None,
    "ty": None,
    "sigma": None}

cielab_file = "cielab_img.png"
cielab_img = cv.cvtColor(img, cv.COLOR_BGR2Lab)
write_file(PART2_DIR / cielab_file, cielab_img)
all_images["cielab"] = {
    "name": "CIELab",
    "image" : cielab_img,
    "space": "CIELab",
    "angle": 0,
    "scale": 1,
    "tx": None,
    "ty": None,
    "sigma": None}

hls_file = "hls_img.png"
hls_img = cv.cvtColor(img, cv.COLOR_BGR2HLS)
write_file(PART2_DIR / hls_file, hls_img)
all_images["hls"] = {
    "name": "HLS",
    "image" : hls_img,
    "space": "HLS",
    "angle": 0,
    "scale": 1,
    "tx": None,
    "ty": None,
    "sigma": None
}

print("3.  Normalize the lighting on the HSV by performing histogram equalization across the V (value) channel.")

hsv_channels_cache = CACHE_DIR / f'hsv_channel_cache_{file_hash(PART2_DIR / hsv_file)}.npz'

if hsv_channels_cache.is_file():
    print(f'Retrieving channel data from cache')
    channels = np.load(hsv_channels_cache)
    h, s, v = channels["h"], channels["s"], channels["v"]
else:
    print("Extracting channel data from the image")
    h, s, v = cv.split(hsv_img)
    np.savez(hsv_channels_cache, h=h, s=s, v=v)

norm_hsv = cv.merge([h, s, cv.equalizeHist(v)])

print("4.  Convert the normalized image back to RGB and save it.")

norm_rgb_file = "norm_rgb_img.png"
norm_rgb = cv.cvtColor(norm_hsv, cv.COLOR_HSV2RGB)
write_file(PART2_DIR / norm_rgb_file, norm_rgb)
all_images["norm_rgb"] = {
    "name": "Normalized RGB",
    "image" : norm_rgb,
    "space": "RGB",
    "angle": 0,
    "scale": 1,
    "tx": None,
    "ty": None,
    "sigma": None}

print("5. Check for 7 images\n")
count_files(PART2_DIR)

print("6.  Perform 2 random affine transformations on each image.")

# images = {
#     "original": img,
#     "grey": grey_img,
#     "bin": bin_img,
#     "hsv": hsv_img,
#     "cielab": cielab_img,
#     "hls": hls_img,
#     "norm_rgb": norm_rgb
# }

transformed_images = {}

for name, image_dict in all_images.items():

    image = image_dict["image"]

    # Two transforms per image
    for i in range(2):
        # Get original image information
        height, width = img.shape[:2]
        center = (width / 2, height / 2)

        # Determine what transformations will happen
        do_rotate = bool(random.getrandbits(1))
        do_scale = bool(random.getrandbits(1))
        do_translate = bool(random.getrandbits(1))

        # If none of the transformation were activated, force one
        if not (do_rotate or do_scale or do_translate):
            selected = random.randint(0,2)
            print(selected)

            if selected == 0:
                do_rotate = True
            if selected == 1:
                do_scale = True
            else:
                do_translate = True

        angle = random.uniform(-180, 180) if do_rotate else 0
        scale = random.uniform(0.5, 1.2) if do_scale else 1

        M = cv.getRotationMatrix2D(center, angle, scale)

        if do_translate:
            tx = random.uniform(-.1, .1) * width
            ty = random.uniform(-.1, .1) * height
            M[0, 2] += tx
            M[1, 2] += ty

        transformed = cv.warpAffine(image, M, (width, height), borderMode=cv.BORDER_REFLECT)
        transformed_name = f'{name}_transformed_{i+1}'
        transformed_file = f'{transformed_name}.png'

        transformed_images[transformed_name] = {
            "name": f'{image_dict["name"]} Transformed',
            "image": transformed,
            "space": image_dict["space"],
            "angle": angle,
            "scale": scale,
            "sigma": None,
        }
        transformed_images[transformed_name]["tx"] = tx if do_translate else None
        transformed_images[transformed_name]["ty"] = ty if do_translate else None

        print(f'\nTransformation completed on {name}')
        print(f'\n  Image transformation specs:')
        print(f'    Transform angle: {angle:.4f}') if do_rotate else print("    Image not rotated")
        print(f'    Transform scale: {scale:.4f}') if do_scale else print("    Image not scaled")
        if do_translate:
            print(f'    Translate tx: {tx:.4f}')
            print(f'    Translate ty: {ty:.4f}')
        else:
            print(f'    Image not translated')

        write_file(PART2_DIR / transformed_file, transformed)

print("7. Check for 21 files")
count_files(PART2_DIR)

print("8.  Apply a Gaussian blur to each image using the levels of sigma: 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5.")

sigmas = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5)
all_images = all_images | transformed_images
blur_images = {}

for name, image_dict in all_images.items():

    image = image_dict["image"]

    print(f'\nPerforming Gaussian blur on {image_dict["name"]}')

    for sigma in sigmas:
        blur = cv.GaussianBlur(image,(7,7),sigma)
        blur_name = f'{name}_blur_{sigma}'
        blur_file = f'{blur_name}.png'

        blur_images[blur_name] = {
            "name": f'{image_dict["name"]} Blurred',
            "image": blur,
            "space": image_dict["space"],
            "angle": image_dict["angle"],
            "scale": image_dict["scale"],
            "tx": image_dict["tx"],
            "ty": image_dict["ty"],
            "sigma": sigma
        }

        print(f'\tGaussian blur complete using sigma value {sigma}')

        write_file(PART2_DIR / blur_file, blur)

print("9. Check for 168 images")
count_files(PART2_DIR)

print("==================== Part 3: Edge Detection ====================")

print("1. Randomly create 4 subsets of 42 images")

all_images = all_images | blur_images

all_image_names = list(all_images.keys())

random.shuffle(all_image_names)

group1 = all_image_names[:42]
del all_image_names[:42]

group2 = all_image_names[:42]
del all_image_names[:42]

group3 = all_image_names[:42]
del all_image_names[:42]

group4 = all_image_names

print("2.  Choose a subset to use in the remaining steps.")

groups = [group1, group2, group3, group4]

group_num = random.randint(0,3)

selected_group = groups[group_num]

print(f'Selected group {group_num + 1}')

print("3. Check for 42 images")
count_files(PART2_DIR)

print("4-8. Perform Sobel, Laplacian, Canny, and Prewitt edge detection on the chosen subset. Create 42, 5-image plots of the input image next to the edge-detected images.")

prewitt_kernel_x = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]], dtype=np.float32)

prewitt_kernel_y = np.array([[-1,-1,-1],
                               [ 0, 0, 0],
                               [ 1, 1, 1]], dtype=np.float32)

sample_number = 1
readme_plots = random.sample(range(1, 43), 6)

for img_name in selected_group:

    print(f'Configuring plot {sample_number}')
    image_dict = all_images[img_name]
    edge_image = image_dict["image"]
    img_file = f'{img_name}.png'

    write_file(PART3_DIR / img_file, edge_image)

    laplacian_edges = cv.Laplacian(edge_image,cv.CV_64F)
    laplacian_abs = cv.convertScaleAbs(laplacian_edges)
    laplace_name = f'{img_name}_laplacian'
    laplace_file = f'{laplace_name}.png'
    write_file(PART3_DIR / laplace_file, laplacian_abs)

    sobelx_edges = cv.Sobel(edge_image,cv.CV_64F,1,0,ksize=5)
    sobely_edges = cv.Sobel(edge_image,cv.CV_64F,0,1,ksize=5)
    combined_sobel_edges = cv.magnitude(sobelx_edges, sobely_edges)
    sobel_abs = cv.convertScaleAbs(combined_sobel_edges)
    sobel_name = f'{img_name}_sobel'
    sobel_file = f'{sobel_name}.png'
    write_file(PART3_DIR / sobel_file, sobel_abs)

    canny_input = cv.convertScaleAbs(edge_image) if edge_image.dtype != np.uint8 else edge_image
    canny_edges = cv.Canny(edge_image,100,200)
    canny_name = f'{img_name}_canny'
    canny_file = f'{canny_name}.png'
    write_file(PART3_DIR / canny_file, canny_edges)

    prewitt_edges_x = cv.filter2D(edge_image, cv.CV_64F, prewitt_kernel_x)
    prewitt_edges_y = cv.filter2D(edge_image, cv.CV_64F, prewitt_kernel_y)
    prewitt_edges = cv.magnitude(prewitt_edges_x, prewitt_edges_y)
    prewitt_abs = cv.convertScaleAbs(prewitt_edges)
    prewitt_name = f'{img_name}_prewitt'
    prewitt_file = f'{prewitt_name}.png'
    write_file(PART3_DIR / prewitt_file, prewitt_abs)

    # Create plot for this edge detection set
    fig = plt.figure(figsize=(10, 10))
    fig.patch.set_facecolor("#1a1a2e")
    gs = fig.add_gridspec(3, 3)

    edge_image = cv.cvtColor(edge_image, cv.COLOR_BGR2RGB)
    laplacian_abs = cv.cvtColor(laplacian_abs, cv.COLOR_BGR2RGB)
    sobel_abs = cv.cvtColor(sobel_abs, cv.COLOR_BGR2RGB)
    canny_edges = cv.cvtColor(canny_edges, cv.COLOR_BGR2RGB)
    prewitt_abs = cv.cvtColor(prewitt_abs, cv.COLOR_BGR2RGB)

    ax_top = fig.add_subplot(gs[0, 1])
    ax_left = fig.add_subplot(gs[1, 0])
    ax_center = fig.add_subplot(gs[1, 1])
    ax_right = fig.add_subplot(gs[1, 2])
    ax_bottom = fig.add_subplot(gs[2, 1])

    # Plot images and add labels
    ax_top.imshow(sobel_abs)
    ax_top.set_title("Sobel Edge", fontsize=12, pad=10, color="white",)
    ax_top.axis("off")

    ax_left.imshow(laplacian_abs)
    ax_left.set_title("Laplacian Edge", fontsize=12, pad=10, color="white",)
    ax_left.axis("off")

    ax_center.imshow(edge_image)
    ax_center.set_title("Input Image", fontsize=12, pad=20, color="white",)
    ax_center.axis("off")

    ax_right.imshow(canny_edges)
    ax_right.set_title("Canny Edge", fontsize=12, pad=10, color="white",)
    ax_right.axis("off")

    ax_bottom.imshow(prewitt_abs)
    ax_bottom.set_title("Prewitt Edge", fontsize=12, pad=10, color="white",)
    ax_bottom.axis("off")

    # Multi-line figure title
    fig.suptitle(
        f'Sample {sample_number} Pipeline Trajectory:\n'
        f'{image_dict["name"]}\n'
        f'-> {image_dict["space"]}\n'
        f'-> Affine(Rot:{image_dict["angle"]}deg, Scale:{image_dict["scale"]}, Trans:[{image_dict["tx"]}, {image_dict["ty"]}])\n'
        # f'-> Affine(Rot:{image_dict["angle"] if image_dict["angle"] else 0}deg, Scale:{image_dict["scale"] if image_dict["scale"] else None}, Trans:[{image_dict["tx"] if image_dict["tx"] else None}, {image_dict["ty"] if image_dict["ty"] else None}])',
        f'-> Gaussian Blur(sigma: {image_dict["sigma"]})\n',
        fontsize=18,
        color="white",
        y=0.95
    )

    # Adjust spacing
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(PLOTS_DIR / f"{img_name}_comparison.png", facecolor=fig.get_facecolor())
    if sample_number in readme_plots:
        plt.savefig(README_PLOTS_DIR / f"{img_name}_comparison.png", facecolor=fig.get_facecolor())
    plt.close()

    sample_number+=1