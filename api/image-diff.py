from tempfile import SpooledTemporaryFile
import cv2
import numpy as np

def find_diff(reference_image_file: SpooledTemporaryFile, given_image_file: SpooledTemporaryFile):
    reference_image = cv2.imdecode(np.frombuffer(reference_image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    given_image = cv2.imdecode(np.frombuffer(given_image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    
    # Convert images to grayscale for feature matching
    reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    given_gray = cv2.cvtColor(given_image, cv2.COLOR_BGR2GRAY)

    # Detect and match keypoints between the two images
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(reference_gray, None)
    kp2, des2 = orb.detectAndCompute(given_gray, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Estimate an affine transformation (ignores perspective distortion)
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)

    if M is not None:
        # Apply the estimated transformation to the reference image
        aligned_reference = cv2.warpAffine(reference_image, M, (given_image.shape[1], given_image.shape[0]))

        # Compute the absolute difference between the aligned reference and given images
        diff_frame = cv2.absdiff(given_image, aligned_reference)

        diff_gray = cv2.cvtColor(diff_frame, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to highlight the differences
        THRESHOLD_CUTOFF = 10 # was 30
        _, thresholded = cv2.threshold(diff_gray, THRESHOLD_CUTOFF, 255, cv2.THRESH_BINARY)

        # Apply morphological operations to further enhance the differences
        thresholded = cv2.erode(thresholded, None, iterations=2)
        thresholded = cv2.dilate(thresholded, None, iterations=2)

        # Find contours of the differences
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        diffs = []

        # Draw rectangles around the detected differences
        for idx, contour in enumerate(contours):
            if cv2.contourArea(contour) > 20:
                x, y, w, h = cv2.boundingRect(contour)
                diffs.append({"i": idx, "x0": x, "y0": y, "x1": x + w, "y0": y+ h})

        return diffs
    else:
        print("No transformation found; images are too dissimilar.")


# File paths for the reference and given images
reference_image_path = "./Spot_the_difference.png"
given_image_path = "./Spot_the_difference2.png"

with SpooledTemporaryFile(mode="w+b", max_size=500) as sp_file:
    sp_file.write(open(reference_image_path, "rb").read())
    sp_file.seek(0)
    with SpooledTemporaryFile(mode="w+b", max_size=500) as sp_file2:
        sp_file2.write(open(given_image_path, "rb").read())
        sp_file2.seek(0)
        diffs = find_diff(sp_file, sp_file2)
        print(diffs)
