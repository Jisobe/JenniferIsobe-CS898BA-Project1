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
| 2026-06-26 12:01 | In this method from opencv, what are the 11 and 2 values: `cv.adaptiveThreshold(grey_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)` | Claude | Explained that `11` is the `blockSize` (size of the local pixel neighborhood, must be odd) used to compute a per-pixel threshold, and `2` is the constant `C` subtracted from the local weighted mean before binarizing | No code |
| 2026-06-26 12:15 | If I am using Otsu thresholding or adaptive thresholding, what would be the best way to segment the foreground of the image for each? | Claude | Gave best-practice pipelines for both methods: Otsu (grayscale → blur → `cv.threshold` with `THRESH_OTSU` → morphological open/close) for images with uniform lighting and bimodal histograms, and adaptive thresholding (grayscale → median blur → `cv.adaptiveThreshold` → morphology → connected component filtering) for images with uneven illumination; included a decision table for choosing between them | Reviewed the provided segmentation code and updated it to fit existing code. Adjusted parameters to improve outputs. |
| 2026-06-26 18:21 | I am working with k means clustering for image segmentation (foreground from background). Visually there is not really a different between using a K value of 4 or 5. Is there a difference computationally or in other ways in using these values? | Claude | Explained computational cost scaling with K, cluster stability/sensitivity to initialization, and recommended quantitative checks (elbow method/inertia, silhouette score) over visual inspection; discussed implications for downstream foreground/background cluster-merging logic | Review the provided quatitiative measure code. Utilized existing manual method for visual inspection and defaulted to k=4 to reduce cost. |
| 2026-06-27 22:03 | Create a log of this conversation including this prompt formatted as a markdown table using the template below. Additionally, create a file containing all code produced as part of this conversation. | Claude | Generated this log file (`conversation_log.md`) and a consolidated code file (`conversation_code.py`) containing all OpenCV/scikit-learn snippets shared in the conversation | Review AI_LOG provided and updated changes column |
| 2026-07-02 17:49 | explain convolutional neural networks to me like i am 5? | Claude | Explained CNNs using a "magic magnifying glass" analogy — sliding window detects simple patterns (edges), layers combine simple patterns into complex ones (shapes → objects) | No code |
| 2026-07-02 17:53 | What is the best tool to use for creating a cnn? | Claude | Compared PyTorch, TensorFlow/Keras, JAX, and scikit-learn for CNN development; recommended based on whether the goal is understanding vs. speed of prototyping | No code |
| 2026-07-02 18:00 | understanding is definitely more important at this point | Claude | Recommended PyTorch specifically for learning, citing explicit forward pass, visible autograd, and eager execution; outlined a 4-step learning path | No code |
| 2026-07-02 18:11 | what are the forward oass and autograd? | Claude | Explained forward pass as the layer-by-layer flow from input to prediction, and autograd as PyTorch's automatic differentiation engine that computes gradients via the chain rule | Reviewed provided code for understanding |
| 2026-07-02 19:30 | how does the model get created using pytorch | Claude | Explained `nn.Module` subclassing pattern — `__init__` declares layers, `forward` connects them; walked through a `SimpleCNN` class example and channel/shape matching rules | Reviewed provided model code |
| 2026-07-08 17:17 | how does training validation and testing work after the model is created? | Claude | Explained train/validation/test split purposes, overfitting detection, and provided training loop, validation loop, and test evaluation code patterns | Reviewed examples of how data is split |
| 2026-07-08 17:26 | Should the training and validation data be resplit while tuning? if you keep training and validating on the same data is there a risk of the model just learning the specifics of the training or validation data? | Claude | Distinguished model overfitting to training data vs. researcher overfitting to a fixed validation set through repeated tuning decisions; introduced k-fold cross-validation as the standard fix | No code |
| 2026-07-08 17:40 | how is k-fold cross validation done | Claude | Explained the k-fold rotation procedure and why it addresses "lucky split" risk; provided a full k-fold implementation using `sklearn.KFold` combined with PyTorch training/eval | Reviewed k-fold example and implementation |
| 2026-07-08 17:51 | how does the test data get separated/split from the train/validation data | Claude | Explained that test splitting must happen first, before any tuning; covered `train_test_split`, `random_state`, and `stratify` for class balance | Reviewed provided code |
| 2026-07-09 16:50 | how does data get loaded in for the training validation and testing | Claude | Explained `Dataset` and `DataLoader` classes, lazy loading via `__getitem__`, transform pipelines, and shuffle/augmentation differences across train/val/test loaders | Reviewed code to determine applicability |
| 2026-07-09 17:05 | How does the augmentation with random flips, crops, color jitter happen and how do the parameters for each get determined | Claude | Explained that transforms re-roll randomness each `__getitem__` call; explained how augmentation parameter values are chosen based on realistic real-world variation per domain | Reviewed code provided |
| 2026-07-09 17:30 | the documentation says ToTensor is deprecated. what is the difference between ToTensor vs ToImage and ToDtype | Claude (web_search) | Searched torchvision docs; confirmed `v2.ToTensor()` deprecation and explained the split into `ToImage()` (structural conversion) and `ToDtype(..., scale=True)` (dtype conversion + rescaling) | Reviewed code and updated ToTensor to use ToImage and ToDtype |
| 2026-07-09 17:38 | What is the benefit of normalizing to [0,1] vs [-1,1]? | Claude | Compared [0,1] vs [-1,1] normalization schemes and explained zero-centering's effect on gradient descent; introduced ImageNet mean/std normalization as the common pretrained-model standard | No code |
| 2026-07-09 17:50 | compare Binarization vs normalization | Claude | Contrasted binarization (destructive, categorical, used for segmentation/masking) against normalization (non-destructive, continuous, used for model input scaling) with a comparison table | Reviewed code to help understanding |
| 2026-07-09 18:25 | How do i determine how hyperparameters needs to change? | Claude | Outlined diagnostic signals from training/validation loss curves (underfitting, overfitting, noisy loss, LR too high/low) and a practical tuning order (LR → batch size → architecture → regularization) | No code |
| 2026-07-09 18:38 | What is the best number of cnn layers to use? | Claude | Explained why more layers isn't automatically better (shrinking feature maps, overfitting risk); gave rough depth guidelines by dataset/task complexity and an empirical comparison approach | No code |
| 2026-07-09 19:50 | How do i determine the best demensions to resize the input images to? | Claude | Covered tradeoffs between detail preservation, compute cost, and aspect ratio; explained why power-of-2-friendly dimensions (128, 224, 256) avoid pooling/rounding issues | No code |
| 2026-07-18 10:38 | Compare regularization via dropout and wight decay | Claude | Compared dropout (random neuron deactivation during training) vs. weight decay (L2 penalty on weight magnitude in the loss function), including a comparison table and guidance on using both together | Reviewed how to implement regularization properly |
| 2026-07-18 10:45 | compare each of these tuning strategies (grid search, random search, bayesian optimixation via optuna/keras Tuner | Claude | Compared exhaustiveness and efficiency of grid search, random search, and Bayesian optimization (Optuna); provided code examples for each and a practical recommendation for a class project's compute budget | Reviewed implementation benefits to determine best approach |
| 2026-07-19 12:59 | create an ai log for this conversation using markdown format... additionally create a file containing all the code produced in this conversation | Claude | Compiled this AI usage log and a consolidated code file covering all snippets produced during the conversation | Updated changes made |
| 2026-07-19 14:58 | how do the following get calculated: Accuracy, Precision, Recall, and F1-Score | Claude | Explained confusion matrix components (TP/TN/FP/FN) and derived formulas for Accuracy, Precision, Recall, and F1-Score; covered the precision/recall tradeoff and macro/weighted/micro averaging for multi-class problems | Reviewed code to add functionality to script |
| 2026-07-19 16:02 | create update entries for the ai log and code summary | Claude | Appended new log entries and updated the consolidated code file with the sklearn metrics snippet | Updated entry |

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

## IMAGE SEGMENTING

```python
def otsu_segment(img):
    # 1. Convert to grayscale
    grey_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 2. Blur slightly to reduce noise (helps Otsu find a cleaner split)
    blur = cv.GaussianBlur(grey_img, (5, 5), 0)

    # 3. Apply Otsu's threshold
    _, otsu_thresh = cv.threshold(
        blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
    )

    # 4. Clean up with morphological operations
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    clean = cv.morphologyEx(otsu_thresh, cv.MORPH_OPEN, kernel, iterations=1)
    clean = cv.morphologyEx(clean, cv.MORPH_CLOSE, kernel, iterations=1)

    return clean

def adaptive_segment(img, block_size=11, c=2, min_area=30):
    # 1. Grayscale + light blur (median blur preserves edges better here)
    grey_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.medianBlur(grey_img, 5)

    # 2. Adaptive threshold
    adap_thresh = cv.adaptiveThreshold(
        blur, 255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY_INV,   # INV if foreground is darker than local background
        blockSize=block_size,   # tune based on feature size; must be odd
        C=c
    )

    # 3. Morphological cleanup
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    clean = cv.morphologyEx(adap_thresh, cv.MORPH_OPEN, kernel, iterations=1)
    clean = cv.morphologyEx(clean, cv.MORPH_CLOSE, kernel, iterations=2)

    # 4. Remove tiny noise blobs via connected components
    n_labels, labels, stats, _ = cv.connectedComponentsWithStats(clean)
    mask = clean.copy()
    for i in range(1, n_labels):
        if stats[i, cv.CC_STAT_AREA] < min_area:
            mask[labels == i] = 0

    return mask
```

## K-MEANS

```python
def kmeans_inertia_curve(img, k_range=range(2, 8), random_state=42):
    from sklearn.cluster import KMeans

    pixels = img.reshape(-1, 3).astype(np.float32)

    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        km.fit(pixels)
        inertias.append(km.inertia_)

    return list(k_range), inertias

def kmeans_silhouette_scores(img, k_values=(3, 4, 5, 6), sample_size=5000,
                              random_state=42):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    pixels = img.reshape(-1, 3).astype(np.float32)

    rng = np.random.default_rng(random_state)
    sample = pixels[rng.choice(len(pixels), sample_size, replace=False)]

    scores = {}
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(sample)
        scores[k] = silhouette_score(sample, km.labels_)

    return scores

# Usage:
#   scores = kmeans_silhouette_scores(img)
#   for k, score in scores.items():
#       print(k, score)
#   # Unlike inertia, silhouette score can penalize over-segmentation:
#   # if K=4's score is >= K=5's, that's evidence K=4 is the better choice.
```

## Forward pass example (nn.Module.forward)

```python
import torch
import torch.nn as nn


def forward_example(self, x):
    x = self.conv1(x)      # apply first convolution
    x = torch.relu(x)      # apply activation
    x = self.pool(x)       # shrink it down
    x = self.conv2(x)      # apply second convolution
    x = x.flatten()        # squash into a 1D vector
    x = self.fc(x)         # final layer -> prediction
    return x
```

## Autograd usage pattern

```python
def autograd_example(model, x, target, loss_fn, optimizer):
    output = model(x)               # forward pass
    loss = loss_fn(output, target)  # how wrong were we?
    loss.backward()                 # autograd computes ALL gradients
    optimizer.step()                # nudge every weight based on its gradient
```

## Defining a CNN model with nn.Module

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # define the layers you'll use -- just declaring them, not connecting them yet
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, 10)  # final layer -> 10 class scores

    def forward(self, x):
        # this is where the layers actually get connected/used
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = x.flatten(1)     # flatten everything except the batch dimension
        x = self.fc(x)
        return x


# Instantiating the model
model = SimpleCNN()

# Basic train/test usage pattern
# output = model(images)          # this calls forward() automatically
# loss = loss_fn(output, labels)
# loss.backward()                 # autograd computes gradients
# optimizer.step()                # update the weights
```

## Training / validation / testing loop

```python
def training_validation_loop(model, train_loader, val_loader, loss_fn, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()  # tells the model "we're learning, behave accordingly"
        for images, labels in train_loader:
            optimizer.zero_grad()          # clear old gradients
            outputs = model(images)        # forward pass
            loss = loss_fn(outputs, labels)
            loss.backward()                # autograd computes gradients
            optimizer.step()               # update weights

        model.eval()  # tells the model "we're just checking, don't learn"
        with torch.no_grad():  # don't bother tracking gradients, we're not training
            val_loss = 0
            for images, labels in val_loader:
                outputs = model(images)
                val_loss += loss_fn(outputs, labels).item()


def test_evaluation(model, test_loader, evaluate):
    model.eval()
    with torch.no_grad():
        test_accuracy = evaluate(model, test_loader)
    return test_accuracy
```

## K-Fold Cross-Validation

```python
from sklearn.model_selection import KFold
import numpy as np


def kfold_cross_validation(dataset, SimpleCNN, loss_fn, num_epochs=10, k=5):
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"Fold {fold + 1}/{k}")

        # build fresh data loaders using only this fold's indices
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=32)

        # IMPORTANT: create a brand new model each fold -- don't reuse trained weights
        model = SimpleCNN()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # train this fold
        for epoch in range(num_epochs):
            model.train()
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

        # validate this fold
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                predicted = outputs.argmax(dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        fold_accuracy = correct / total
        fold_scores.append(fold_accuracy)
        print(f"  Fold {fold + 1} accuracy: {fold_accuracy:.4f}")

    print(f"\nMean accuracy: {np.mean(fold_scores):.4f}")
    print(f"Std dev: {np.std(fold_scores):.4f}")
    return fold_scores
```

## Train / validation / test splitting

```python
from sklearn.model_selection import train_test_split


def split_train_val_test(all_data, all_labels):
    # first split: carve off the test set
    train_val_data, test_data, train_val_labels, test_labels = train_test_split(
        all_data, all_labels,
        test_size=0.15,        # 15% goes to test
        random_state=42,       # reproducibility -- same split every time you run this
        shuffle=True
    )

    # second split: divide what's left into train/val
    train_data, val_data, train_labels, val_labels = train_test_split(
        train_val_data, train_val_labels,
        test_size=0.176,       # ~15% of the ORIGINAL data (0.176 * 0.85 ~= 0.15)
        random_state=42,
        shuffle=True
    )
    return train_data, val_data, test_data, train_labels, val_labels, test_labels


def split_with_stratify(all_data, all_labels):
    # Stratification preserves class proportions across splits -- useful for imbalanced classes
    train_val_data, test_data, train_val_labels, test_labels = train_test_split(
        all_data, all_labels,
        test_size=0.15,
        random_state=42,
        stratify=all_labels   # keeps class ratios consistent across splits
    )
    return train_val_data, test_data, train_val_labels, test_labels
```

## Custom Dataset, transforms, and DataLoaders

```python
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths  # list of file paths
        self.labels = labels            # list of corresponding labels
        self.transform = transform      # preprocessing to apply

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # loads ONE image at a time -- not all of them upfront
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# Training transform pipeline -- includes augmentation
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),   # augmentation -- ONLY for training
    transforms.ToTensor(),                # converts PIL image -> PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])

# Validation/test transform pipeline -- no random augmentation
val_test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])

# Instantiate datasets and loaders (paths/labels come from split_train_val_test above)
# train_dataset = ImageDataset(train_paths, train_labels, transform=train_transform)
# val_dataset = ImageDataset(val_paths, val_labels, transform=val_test_transform)
# test_dataset = ImageDataset(test_paths, test_labels, transform=val_test_transform)

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

# Example iteration pattern:
# for images, labels in train_loader:
#     # images.shape -> [32, 3, 128, 128]  (batch, channels, height, width)
#     # labels.shape -> [32]
#     outputs = model(images)
```

## Augmentation pipeline (expanded, with more transform types)

```python
augmentation_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[...], std=[...])
])
```

## ToTensor deprecation -> ToImage + ToDtype replacement (torchvision v2)

```python
from torchvision.transforms import v2

updated_transform = transforms.Compose([
    v2.ToImage(),                                   # convert to tensor (still uint8, still 0-255)
    v2.ToDtype(torch.float32, scale=True),          # convert dtype AND rescale to 0.0-1.0
])
```

## Binarization example

```python
import numpy as np
import cv2 as cv


def binarize_example(gray_image, threshold):
    binary = (gray_image > threshold).astype(np.uint8) * 255
    return binary
```

## Dropout and weight decay

```python
class DropoutExampleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=0.5)  # 50% of neurons zeroed out each forward pass

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        return x


# Weight decay is set on the optimizer (L2 regularization)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
```

## Hyperparameter search strategies: grid search, random search, Optuna

```python
# --- Grid search (conceptual definition of the search space) ---
learning_rates = [0.1, 0.01, 0.001]
batch_sizes = [16, 32, 64]
dropout_rates = [0.2, 0.5]
# 3 x 3 x 2 = 18 total combinations, every single one gets trained

# --- Random search using sklearn's ParameterSampler ---
from sklearn.model_selection import ParameterSampler

param_distributions = {
    'lr': [0.1, 0.01, 0.001, 0.0001],
    'batch_size': [16, 32, 64, 128],
    'dropout': [0.1, 0.2, 0.3, 0.4, 0.5]
}
# randomly sample e.g. 20 combinations instead of all 4x4x5=80
# sampled_params = list(ParameterSampler(param_distributions, n_iter=20, random_state=42))

# --- Bayesian optimization using Optuna ---
import optuna


def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    # model = build_model(dropout=dropout)
    # val_accuracy = train_and_evaluate(model, lr=lr, batch_size=batch_size)
    val_accuracy = 0.0  # placeholder -- replace with actual training/eval call
    return val_accuracy


# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=30)
# print(study.best_params)
```

## Classification metrics: Accuracy, Precision, Recall, F1-Score

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


def compute_classification_metrics(y_true, y_pred, average='macro'):
    """
    average: 'macro' (equal weight per class), 'weighted' (weighted by class
    support), or 'micro' (pool TP/FP/FN across classes globally).
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)
    f1 = f1_score(y_true, y_pred, average=average)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


# Full per-class breakdown, all metrics at once:
# print(classification_report(y_true, y_pred))
```
