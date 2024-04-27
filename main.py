import os
from tqdm import tqdm

import cv2
from PIL import Image, ImageFilter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from sklearn.cluster import KMeans

product_path = 'product_images'

for images in tqdm(os.listdir(product_path)):
    images = 'product1.jpg'
    product_image_path = os.path.join(product_path, images)
    
    # Stage 1: Skin Detection
    im = cv2.imread(product_image_path)
    
    # Stage 4: Region Growing
    varIm = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, res1 = cv2.threshold(varIm, 250, 255, cv2.THRESH_BINARY)
    res2 = cv2.bitwise_not(res1)
    res = cv2.bitwise_or(res1, res2)
    
    original_image = cv2.imread(product_image_path)
    
    resized_mask = cv2.resize(res, (original_image.shape[1], original_image.shape[0]))
    
    # Convert the mask to 3 channels (RGB) for compatibility
    resized_mask_rgb = cv2.cvtColor(resized_mask, cv2.COLOR_GRAY2RGB)
    
    # Mask out the background from the original image
    masked_image = cv2.bitwise_and(original_image, resized_mask_rgb)
    
    # Reshape the image to a 2D array of pixels
    pixels = masked_image.reshape((-1, 3))
    
    # Perform K-means clustering to find dominant colors
    k = 5  # Number of clusters
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    
    # Get the cluster centers sorted by frequency
    unique_l, counts_l = np.unique(kmeans.labels_, return_counts=True)
    sort_ix = np.argsort(counts_l)
    sort_ix = sort_ix[::-1]
    
    # Plot the dominant colors with adjusted rectangle width and position
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x_from = 0.05
    rectangle_width = 0.19  # Set the width of the rectangles
    gap = 0.05  # Set the gap between rectangles
    
    for i in range(1, 4):  # Ensure that 4 rectangles are drawn
        cluster_center = kmeans.cluster_centers_[sort_ix[i]]
        facecolor = '#%02x%02x%02x' % (int(cluster_center[2]), int(cluster_center[1]), int(cluster_center[0]))
        ax.add_patch(patches.Rectangle((x_from, 0.05), rectangle_width, 0.9, alpha=None, facecolor=facecolor))
        x_from = x_from + rectangle_width + gap  # Add a small gap between rectangles
    
    # Plot the last rectangle (adjusted to fit within the plot boundaries)
    cluster_center = kmeans.cluster_centers_[sort_ix[4]]
    facecolor = '#%02x%02x%02x' % (int(cluster_center[2]), int(cluster_center[1]), int(cluster_center[0]))
    ax.add_patch(patches.Rectangle((x_from, 0.05), 1 - x_from - gap, 0.9, alpha=None, facecolor=facecolor))
    
    plt.show()
    
    break