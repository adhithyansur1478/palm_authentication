import os
import sys

import onnxruntime as ort
import numpy as np
import cv2
import matplotlib.pyplot as plt


# ---------------- ONNX session ----------------
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and PyInstaller """
    if hasattr(sys, "_MEIPASS"):  # running inside exe
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

onnx_path = resource_path("palmnet_embedder.onnx")
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

def show(img):
    plt.imshow(img, cmap='gray')
    plt.xticks([])
    plt.yticks([])

# ---------------- ROI extractor (yours, slightly cleaned) ----------------
def roi_preprocessing(img,size=(224,224),debug=False):
    # accept both str/path and numpy arrays
    if isinstance(img, (str, os.PathLike)):
        img_original = cv2.imread(str(img), 0)
        if img_original is None:
            raise ValueError(f"Could not read image: {img}")
    else:
        # numpy array
        arr = img
        if arr.ndim == 3:  # BGR
            img_original = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        elif arr.ndim == 2:
            img_original = arr
        else:
            raise ValueError("img_or_path must be a path or HxW / HxWx3 numpy array")

    #img_original = cv2.imread(img, 0)
    h, w = img_original.shape
    img = np.zeros((h + 160, w), np.uint8)
    img[80:-80, :] = img_original
    """plt.figure(figsize=(15, 5))
    plt.subplot(131)
    show(img)"""
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    """plt.subplot(132)
    show(blur)"""
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    """plt.subplot(133)
    show(th)
    plt.tight_layout()
    plt.show()"""

    M = cv2.moments(th)
    h, w = img.shape
    x_c = M['m10'] // M['m00']
    y_c = M['m01'] // M['m00']
    """plt.figure(figsize=(15, 5))
    plt.subplot(121)
    show(th)
    plt.plot(x_c, y_c, 'bx', markersize=10)"""
    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]]).astype(np.uint8)
    erosion = cv2.erode(th, kernel, iterations=1)
    boundary = th - erosion

    cnt, _ = cv2.findContours(boundary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    img_c = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cnt = cnt[0]
    img_cnt = cv2.drawContours(img_c, [cnt], 0, (255, 0, 0), 2)
    """plt.subplot(122)
    plt.plot(x_c, y_c, 'bx', markersize=10)
    show(img_cnt)
    plt.tight_layout()
    plt.show()"""

    cnt = cnt.reshape(-1, 2)
    left_id = np.argmin(cnt.sum(-1))
    cnt = np.concatenate([cnt[left_id:, :], cnt[:left_id, :]])

    # Use floating-point division for accurate center
    x_c = int(M['m10'] / M['m00'])
    y_c = int(M['m01'] / M['m00'])

    # Define box size (e.g., 100x100)
    box_size = int(min(img.shape[0], img.shape[1]) * 0.6)  # 30% of the smaller side

    half = box_size // 2

    # Calculate top-left and bottom-right corners centered around (x_c, y_c)
    x1 = max(0, x_c - half)
    y1 = max(0, y_c - half)
    x2 = min(img.shape[1], x_c + half)
    y2 = min(img.shape[0], y_c + half)

    # Draw green rectangle
    img_box = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(img_box, (x1, y1), (x2, y2), (0, 255, 0), 5)

    # Draw red centroid marker
    # cv2.circle(img_box, (x_c, y_c), 4, (0, 0, 255), 50)

    # Show result
    """plt.figure(figsize=(4, 6))
    plt.imshow(img_box)
    plt.title("Centroid with Square Box")
    plt.axis("off")
    plt.show()"""

    # Crop the boxed region from the image that includes the marker and rectangle
    cropped_with_marker = img_box[y1:y2, x1:x2]
    cropped_without_marker = img[y1:y2, x1:x2]
    roi_224 = cv2.resize(cropped_without_marker, size, interpolation=cv2.INTER_AREA)
    #cv2.imwrite("croopped.jpg", cropped_without_marker)
    #cv2.imwrite("croopped_224.jpg",roi_224)

    # Optional: Resize if needed
    # cropped_with_marker_resized = cv2.resize(cropped_with_marker, (224, 224))

    # Save the image
    # cv2.imwrite("cropped_with_marker.jpg", cropped_with_marker_resized)

    # Display
    """plt.imshow(cropped_with_marker)
    plt.title("Cropped ROI with Marker and Box")
    plt.axis("off")
    plt.show()"""

    if debug:
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        show(img)
        plt.subplot(132)
        show(blur)
        plt.subplot(133)
        show(th)
        plt.tight_layout()
        plt.show()
        plt.figure(figsize=(15, 5))
        plt.subplot(121)
        show(th)
        plt.plot(x_c, y_c, 'bx', markersize=10)
        plt.subplot(122)
        plt.plot(x_c, y_c, 'bx', markersize=10)
        show(img_cnt)
        plt.tight_layout()
        plt.show()
        plt.figure(figsize=(4, 6))
        plt.imshow(img_box)
        plt.title("Centroid with Square Box")
        plt.axis("off")
        plt.show()
        plt.imshow(cropped_with_marker)
        plt.title("Cropped ROI with Marker and Box")
        plt.axis("off")
        plt.show()

    return roi_224

# ---------------- Preprocess for ONNX ----------------
def preprocess_from_gray224(gray_224):
    # If your model was trained on RGB, repeat channel to 3
    rgb = np.repeat(gray_224[..., None], 3, axis=2)  # HxWx3
    rgb = rgb.astype(np.float32) / 255.0
    rgb = (rgb - 0.5) / 0.5  # normalize to [-1, 1]; change if you used different mean/std
    # HWC -> NCHW
    rgb = np.transpose(rgb, (2, 0, 1))[None, ...].astype(np.float32)
    return rgb

# ---------------- Embedding & similarity ----------------
def get_embedding(img_path,debugg=False):
    roi_224 = roi_preprocessing(img_path, size=(224, 224),debug=debugg)
    x = preprocess_from_gray224(roi_224)
    emb = sess.run(["embedding"], {"input": x})[0][0]
    emb = emb / (np.linalg.norm(emb) + 1e-12)
    return emb

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

# ---------------- Example ----------------
#emb1 = get_embedding("meel1.jpeg")
#print(emb1)
#emb2 = get_embedding("meel3.jpeg")
#print("Cosine similarity:", cosine_sim(emb1, emb2))
