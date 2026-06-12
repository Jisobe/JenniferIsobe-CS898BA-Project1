# AI Log

This file provides a log of the AI prompts utilized for the development of this repository. The AI_LOG directory also contains original code and files provided in response to AI prompts if applicable

## Conversation Log

| Date and Time | Prompt | Tool | Response Synopsis | Change |
| --- | --- | --- | --- | --- |
| 2026-06-09 18:24 | Where in these docs can I read about extracting image statistics: https://docs.opencv.org/4.13.0/ | Claude | Directed to Core Array Operations (`group__core__array.html`) for `mean`, `meanStdDev`, `minMaxLoc` etc., and the Python histogram tutorial | No code |
| 2026-06-09 18:31 | Is there any documentation on using this functionality with python | Claude | Pointed to the OpenCV-Python tutorials root, histogram tutorial, and noted that the API reference includes Python signatures for all functions | No code |
| 2026-06-09 18:34 | Should a venv be used when creating a repo that uses python with openCV? if so, would it be better to use uv or pip | Claude | Recommended venv; recommended uv over pip for speed, lockfiles, and built-in venv management. Also noted `opencv-python` vs `opencv-python-headless` decision | No code |
| 2026-06-09 18:38 | When would you need to use GUI windows? What is the functionality of this? What are the limitations if I use opencv-python-headless? | Claude | Explained GUI functions (`imshow`, `waitKey`, trackbars, mouse callbacks). Stated headless removes only display/input functions; all processing functions remain identical | No code |
| 2026-06-09 18:42 | If I am using opencv-python-headless, do I need to install OpenCV using something like brew install opencv? | Claude | No — the wheel is self-contained. `brew install opencv` only needed for C++ projects, custom builds, or system-wide installs. Noted Linux libgl caveat | No code |
| 2026-06-09 18:59 | Where in the documentation does it talk about getting the individual channels out of an image | Claude | Pointed to `tutorial_py_basic_ops.html`. Explained `cv.split()` and NumPy indexing as the two methods. Noted BGR channel order | No code |
| 2026-06-09 19:02 | What about the other channels like H, S, V, Y, a, b, Cr, Cb, U, V, X, Y, and Z | Claude | Explained colour space conversion via `cv.cvtColor()`. Provided table of colour spaces and their conversion codes. Pointed to `tutorial_py_colorspaces.html` | Reviewed cv.cvtColor code against documentation. Updated code for given images in script. |
| 2026-06-09 19:19 | For splitting the image with cv.split(), is there a preferred way to cache this information so it does not have to be rerun every time | Claude | Recommended `np.savez()` / `.npz` format as lossless, fast, and natural for NumPy arrays. Compared against pickle, imwrite, HDF5. Showed file hash cache-busting pattern | Updated caching fucntionality for reading/writing to files |
| 2026-06-09 22:40 | How would mode be calculated for one of the channels | Claude | Explained `scipy.stats.mode()` as the standard approach. Provided NumPy-only alternative with `np.unique()`. Recommended `np.bincount()` as fastest for 8-bit pixel data | Reviewed provided code and imtegrated mode calculation in with the script |
| 2026-06-09 23:09 | How does conversion to a binary image from BGR work | Claude | Explained thresholding pipeline: BGR → grayscale → `cv.threshold()`. Covered threshold types, Otsu's method, and adaptive thresholding. Pointed to `tutorial_py_thresholding.html` | Review code against the documentation to determine the method to use. Updated the color space functionality for binary conversion. |
| 2026-06-09 23:13 | When converting image colors with opencv, what is the difference between the regular and full conversions? e.g. COLOR_BGR2HLS and COLOR_BGR2HLS_FULL | Claude | Regular: H in 0–179; Full: H in 0–255. Explained importance for `cv.inRange()` colour detection. Recommended regular as default | No code |
| 2026-06-09 23:16 | If I am trying to convert from BGR to CIELAB would I use COLOR_BGR2LAB or COLOR_BGR2Lab | Claude | `COLOR_BGR2Lab` is canonical. Explained 8-bit output ranges (L: 0–255, a/b: 0–255 shifted). | No code |
| 2026-06-09 23:19 | When saving an image using my script, what file extension should I use? | Claude | Recommended PNG for lossless intermediate storage. Compared TIFF, BMP, JPEG. Noted OpenCV does not embed colour space metadata on save | No code |
| 2026-06-10 00:28 | What are the 2 arguments the merge function takes in opencv | Claude | Clarified `cv.merge()` takes one argument — a list or tuple of arrays. Noted possible confusion with `cv.addWeighted()` or `cv.mixChannels()` | No code |
| 2026-06-10 00:33 | Does it matter if the arg is a list or tuple | Claude | No difference — both work identically. List is conventional in docs and examples | No code |
| 2026-06-10 00:54 | Explain how the affine transformation works in opencv | Claude | Explained 2x3 matrix math, `cv.warpAffine()`, `getRotationMatrix2D()`, `getAffineTransform()`, border modes. Contrasted with perspective transformation | No code |
| 2026-06-11 08:30 | So if I am just trying to generate random affine transformations would I just use random numbers in the method for creating M | Claude | Recommended randomising parameters of `getRotationMatrix2D()` rather than raw matrix values. Provided safe ranges. Noted data augmentation libraries as an alternative | Reviewed code provided and looked into valued provided for the transformations. |
| 2026-06-11 08:41 | Does the angle value produce the shear effect? What gives a rotation | Claude | Clarified angle = pure rotation, not shear. Explained how to add shear manually via off-diagonal matrix elements | Manual M construction with shear_x/shear_y |
| 2026-06-11 08:44 | For the limits given in the angle, scale, tx, and ty, where do those numbers come from | Claude | Clarified they are arbitrary rules of thumb. Explained reasoning behind each range. Recommended expressing tx/ty as fraction of image dimensions | No code |
| 2026-06-11 09:30 | If I don't want to rotate or scale do I just put 0 for them | Claude | angle=0 correct; scale must be 1 not 0 (0 collapses image to a point) | No code |
| 2026-06-11 09:44 | For GaussianBlur is the (5,5) the kernel and 0 the sigmaX. How do the values for the kernels get decided | Claude | Confirmed kernel must be odd and positive. Explained kernel size effect on blur strength. Explained sigmaX=0 auto-calculation. Showed alternative of passing (0,0) with explicit sigma | Reviewed gaussian blur functions provided and read through documentation to determine the best approach |
| 2026-06-11 10:45 | Where can I read about Sobel, Laplacian, Prewitt, and Canny edge detection in the documentation | Claude | Pointed to `tutorial_py_gradients.html`, `tutorial_py_canny.html`, and filter API reference. Noted Prewitt has no dedicated function and requires `cv.filter2D()` with manual kernel | No code |
| 2026-06-11 10:52 | Explain what each argument is and does for Laplacian, Sobel, Canny, and filter2D | Claude | Detailed explanation of every argument for each function. Covered CV_64F necessity for signed output, hysteresis thresholding in Canny, kernel weight meaning in Prewitt | Review provided calculations for the edge detection and updated for all images that needed to be processed. |
| 2026-06-11 10:55 | For the Sobel method can the x and y both be 1 or 0 in a single call | Claude | Both 1: valid but uncommon mixed derivative. Both 0: returns blurred image, no edges. Showed combining X+Y with `cv.magnitude()` and `cv.phase()` | No code |
| 2026-06-11 11:24 | How would I create a plot with the images | Claude | Provided Matplotlib subplot grid per image showing Original, Laplacian, Sobel, Canny, Prewitt. Explained cmap, plt.close(), figsize, tight_layout, and BGR→RGB conversion for colour images | Reviewed the same plot code and updated to match the format of the example provided. |
| 2026-06-11 21:13 | create an log for this entire conversation using the markdown table format with the following columns date and time, entire prompt, tool, response summary, and changes made. additionally, create a file with all of the code blocks produced in this conversation | Claude | Provided a log with all of the applicable columns formatted as a markdown table and a file with code snippets generated | Updated the log to correct the date and time and to update the changes made col. |

## Code Produced

"""
All code blocks produced during the OpenCV conversation.
Organised by topic.
"""

## COLOUR SPACE CONVERSION

```python
def convert_to_hsv(img):
    return cv.cvtColor(img, cv.COLOR_BGR2HSV)


def convert_to_lab_uint8(img):
    return cv.cvtColor(img, cv.COLOR_BGR2Lab)


def convert_to_lab_float(img):
    # True Lab values: L in 0-100, a/b in ~-127 to 127
    img_float = img.astype(np.float32) / 255.0
    return cv.cvtColor(img_float, cv.COLOR_BGR2Lab)
```

## SPLITTING AND CACHING CHANNELS

```python
def split_and_cache(img_path, cache_path):
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        b, g, r = data["b"], data["g"], data["r"]
    else:
        img = cv.imread(img_path)
        b, g, r = cv.split(img)
        np.savez(cache_path, b=b, g=g, r=r)
    return b, g, r


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()
```

## MODE CALCULATION

```python
def mode_scipy(channel):
    mode_result = stats.mode(channel.flatten())
    mode_value = mode_result.mode
    mode_count = mode_result.count
    return mode_value, mode_count


def mode_numpy(channel):
    values, counts = np.unique(channel.flatten(), return_counts=True)
    mode_value = values[np.argmax(counts)]
    mode_count = counts[np.argmax(counts)]
    return mode_value, mode_count


def mode_bincount(channel):
    # Fastest for 8-bit pixel data
    hist = np.bincount(channel.flatten(), minlength=256)
    mode_value = np.argmax(hist)
    mode_count = hist[mode_value]
    return mode_value, mode_count
```

## BINARY / THRESHOLDING

```python
def to_binary_simple(img, thresh=127):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, thresh, 255, cv.THRESH_BINARY)
    return binary


def to_binary_otsu(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return binary


def to_binary_adaptive(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    binary = cv.adaptiveThreshold(
        gray, 255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY, 11, 2
    )
    return binary
```

## AFFINE TRANSFORMATIONS

```python
def random_affine(img):
    height, width = img.shape[:2]
    center = (width / 2, height / 2)

    angle = random.uniform(-30, 30)
    scale = random.uniform(0.8, 1.2)
    tx = random.uniform(-0.1, 0.1) * width
    ty = random.uniform(-0.1, 0.1) * height

    M = cv.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty

    return cv.warpAffine(img, M, (width, height), borderMode=cv.BORDER_REFLECT)


def affine_translation_only(img):
    height, width = img.shape[:2]
    center = (width / 2, height / 2)

    tx = random.uniform(-0.1, 0.1) * width
    ty = random.uniform(-0.1, 0.1) * height

    # angle=0, scale=1 for no rotation or scaling
    M = cv.getRotationMatrix2D(center, 0, 1)
    M[0, 2] += tx
    M[1, 2] += ty

    return cv.warpAffine(img, M, (width, height))


def affine_with_shear(img):
    height, width = img.shape[:2]
    center = (width / 2, height / 2)

    angle = random.uniform(-30, 30)
    scale = random.uniform(0.8, 1.2)
    shear_x = random.uniform(-0.2, 0.2)
    shear_y = random.uniform(-0.2, 0.2)

    M = cv.getRotationMatrix2D(center, angle, scale)
    M[0, 1] += shear_x
    M[1, 0] += shear_y

    return cv.warpAffine(img, M, (width, height))
```

## GAUSSIAN BLUR

```python
def blur_with_kernel_size(img):
    return cv.GaussianBlur(img, (5, 5), 0)

def blur_with_sigma(img):
    return cv.GaussianBlur(img, (0, 0), sigmaX=2.0)
```

## EDGE DETECTION

```python
prewitt_kernel_x = np.array([[-1, 0, 1],
                               [-1, 0, 1],
                               [-1, 0, 1]], dtype=np.float32)

prewitt_kernel_y = np.array([[-1, -1, -1],
                               [ 0,  0,  0],
                               [ 1,  1,  1]], dtype=np.float32)


def detect_edges(img):
    """
    Returns a dict of edge images for each detection method.
    Assumes img is uint8 grayscale.
    """
    results = {}

    # Laplacian
    laplacian = cv.Laplacian(img, cv.CV_64F)
    results["laplacian"] = cv.convertScaleAbs(laplacian)

    # Sobel
    sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
    sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)
    sobel_magnitude = cv.magnitude(sobelx, sobely)
    results["sobel"] = cv.convertScaleAbs(sobel_magnitude)

    # Sobel direction (optional)
    results["sobel_angle"] = cv.phase(sobelx, sobely, angleInDegrees=True)

    # Canny
    canny_input = cv.convertScaleAbs(img) if img.dtype != np.uint8 else img
    results["canny"] = cv.Canny(canny_input, 100, 200)

    # Prewitt
    prewitt_x = cv.filter2D(img, cv.CV_64F, prewitt_kernel_x)
    prewitt_y = cv.filter2D(img, cv.CV_64F, prewitt_kernel_y)
    prewitt_magnitude = cv.magnitude(prewitt_x, prewitt_y)
    results["prewitt"] = cv.convertScaleAbs(prewitt_magnitude)

    return results
```

### EDGE DETECTION LOOP WITH FILE SAVING AND MATPLOTLIB PLOT

```python
def process_and_plot_edges(selected_group, all_images, output_dir):
    for img_name in selected_group:
        edge_image = all_images[img_name]

        # Save original
        cv.imwrite(str(output_dir / f"{img_name}.png"), edge_image)

        # Laplacian
        laplacian_edges = cv.convertScaleAbs(cv.Laplacian(edge_image, cv.CV_64F))
        cv.imwrite(str(output_dir / f"{img_name}_laplacian.png"), laplacian_edges)

        # Sobel
        sobelx_edges = cv.Sobel(edge_image, cv.CV_64F, 1, 0, ksize=5)
        sobely_edges = cv.Sobel(edge_image, cv.CV_64F, 0, 1, ksize=5)
        sobel_edges = cv.convertScaleAbs(cv.magnitude(sobelx_edges, sobely_edges))
        cv.imwrite(str(output_dir / f"{img_name}_sobel.png"), sobel_edges)

        # Canny
        canny_input = cv.convertScaleAbs(edge_image) if edge_image.dtype != np.uint8 else edge_image
        canny_edges = cv.Canny(canny_input, 100, 200)
        cv.imwrite(str(output_dir / f"{img_name}_canny.png"), canny_edges)

        # Prewitt
        prewitt_x = cv.filter2D(edge_image, cv.CV_64F, prewitt_kernel_x)
        prewitt_y = cv.filter2D(edge_image, cv.CV_64F, prewitt_kernel_y)
        prewitt_edges = cv.convertScaleAbs(cv.magnitude(prewitt_x, prewitt_y))
        cv.imwrite(str(output_dir / f"{img_name}_prewitt.png"), prewitt_edges)

        # Matplotlib comparison plot
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        fig.suptitle(img_name)

        images = [edge_image, laplacian_edges, sobel_edges, canny_edges, prewitt_edges]
        titles = ["Original", "Laplacian", "Sobel", "Canny", "Prewitt"]

        for ax, image, title in zip(axes, images, titles):
            ax.imshow(image, cmap="gray")
            ax.set_title(title)
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(str(output_dir / f"{img_name}_comparison.png"))
        plt.close()
```
