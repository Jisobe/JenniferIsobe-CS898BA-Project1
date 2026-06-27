# Homework Two: Image Segmentation

Instructions from <https://github.com/codyfarlow1/CS898BA-HW2.git>

**CS 898BA – Image Analysis and Computer Vision**

**Purpose:** To apply and evaluate classical and optimization-based image segmentation techniques.

**Situation:** Your supervisor’s supervisor is back. After reviewing your edge detection plots from Homework One, he is convinced that the "alien" has a distinct torso and head structure. He wants you to isolate the entity from the background completely and mutters something about "extracting the exact pixel area for a bio-mass calculation" before rushing out to a meeting.

Since you already have a repository and a solid pipeline, you decide to waste more time by using image segmentation to isolate the figure.

---

## Part 1: Repository Maintenance & Logistics

1. **Branching:** Do not overwrite your Homework One code. Create a new branch in your existing repository named `Feature-Segmentation`.
2. **AI Log:** Continue tracking all AI usage in your `AI_Log.md` file following the exact same format as Homework One.
3. **Commit Strategy:** Perform incremental commits with clear, professional messages as you complete each segmentation task.

---

## Part 2: Image Preprocessing & Multi-Channel Normalization

Before applying segmentation algorithms, you must standardize the illumination across the entire color space to handle the challenging lighting conditions of the doorbell camera feed:

1. **Multi-Channel Color Normalization:**
   * Load the original image from Homework One.
   * Split the image into its three color channels (e.g., R, G, and B or via the V channel in HSV / L channel in LAB).
   * Apply Histogram Equalization independently to all three channels to normalize illumination and maximize contrast across the entire color spectrum.
   * Merge the channels back together to create a fully normalized color image.
   * Save this normalized color image; it will serve as the primary input for all subsequent segmentation tasks.

---

## Part 3: Threshold-Based Segmentation

Using the normalized color image produced in Part 2, implement the following baseline techniques using OpenCV:

1. **Otsu’s Global Thresholding:**
   * Convert your normalized color image to grayscale.
   * Apply Otsu's automatic thresholding to separate the foreground object from the background.
2. **Adaptive Thresholding:**
   * Apply adaptive thresholding (Gaussian window) to the grayscale version of your normalized image to handle local illumination variations.

*Save the resulting binary masks and the segmented foreground extractions for both methods.*

---

## Part 4: Classical & Optimization-Based Segmentation

To handle the complex textures and shadows of the outdoor scene, move beyond simple pixel intensities. Use the fully normalized color image from Part 2 as the input for these methods:

1. **Color-Space Clustering (K-Means):**
   * Convert the normalized color image to the HSV color space.
   * Apply K-Means clustering to segment the image into $K$ distinct regions (test and select an optimal $K$ value between 3 and 5).
   * Isolate the cluster that most closely captures the "unknown figure."

*Save the resulting binary masks and the segmented foreground extractions for both methods.*

---

## Part 5: Evaluation and Analysis

You now have 3 distinct segmentation outputs (Otsu, Adaptive, and K-Means).

1. **Qualitative Analysis:**
   * Discuss the pros and cons of each approach regarding background noise (e.g., leaves, porch structures) and edge preservation of the central figure. Specifically note how color normalization across all three channels impacted the final segmentation compared to the raw image results from Homework One.
2. **Quantitative Comparison (Pseudo-Ground Truth):**
   * Manually create or define a clean reference mask of the "figure" to serve as your ground truth.
   * Calculate and print the **Intersection over Union (IoU) / Jaccard Index** and the **Dice Coefficient** for each of the 3 segmentation methods against your reference mask.

$$IoU = \frac{|A \cap B|}{|A \cup B|}$$

3. **Visualization:**
   * Create a multi-image comparison plot displaying the original image, the multi-channel normalized color image, and the 4 final segmented masks side-by-side. Include this plot in your updated `README.md`.

---

## Part 6: Submission

**Submit your updated GitHub repository link on BB.** Ensure your master/main branch or pull request clearly reflects the additions made for Homework Two.