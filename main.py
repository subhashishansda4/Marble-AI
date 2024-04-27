import os
from tqdm import tqdm

import cv2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from sklearn.cluster import KMeans

product_path = 'product_images'

for images in tqdm(os.listdir(product_path)):
    product_image_path = os.path.join(product_path, images)
    
    # Loading and merging datasets
    photos_df = pd.read_csv('photos.tsv000', sep='\t')
    keywords_df = pd.read_csv('keywords.tsv000', sep='\t')
    colors_df = pd.read_csv('colors.tsv000', sep='\t')
    
    merge_df = pd.merge(keywords_df, colors_df, on='photo_id', how='outer')
    df = pd.merge(photos_df, merge_df, on='photo_id', how='outer')
    df.to_csv('dataset/data.csv', index=False)
    
    
    product_image = cv2.imread(product_image_path)
    
    # Background removal
    gray = cv2.cvtColor(product_image, cv2.COLOR_BGR2GRAY)
    _, mask1 = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    mask2 = cv2.bitwise_not(mask1)
    mask = cv2.bitwise_or(mask1, mask2)
    
    # Resizing
    res_mask = cv2.resize(mask, (product_image.shape[1], product_image.shape[0]))
    
    # Convert the mask to RGB
    res_mask_rgb = cv2.cvtColor(res_mask, cv2.COLOR_GRAY2RGB)
    
    # Masking out
    masked_image = cv2.bitwise_and(product_image, res_mask_rgb)
    
    plt.imshow(masked_image, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.imsave('plots/mask_image.jpg', masked_image, cmap='gray')
    
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
    hexcodes = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x_from = 0.05
    rectangle_width = 0.19
    gap = 0.05
    
    for i in range(1, 4):
        cluster_center = kmeans.cluster_centers_[sort_ix[i]]
        facecolor = '#%02x%02x%02x' % (int(cluster_center[2]), int(cluster_center[1]), int(cluster_center[0]))
        ax.add_patch(patches.Rectangle((x_from, 0.05), rectangle_width, 0.9, alpha=None, facecolor=facecolor))
        x_from = x_from + rectangle_width + gap
        hexcodes.append(facecolor)
    
    # Plot the last rectangle (adjusted to fit within the plot boundaries)
    cluster_center = kmeans.cluster_centers_[sort_ix[4]]
    facecolor = '#%02x%02x%02x' % (int(cluster_center[2]), int(cluster_center[1]), int(cluster_center[0]))
    ax.add_patch(patches.Rectangle((x_from, 0.05), 1 - x_from - gap, 0.9, alpha=None, facecolor=facecolor))
    hexcodes.append(facecolor)
    
    plt.savefig(f'plots/{os.path.splitext(images)[0]}_colors.jpg')
    plt.show()

