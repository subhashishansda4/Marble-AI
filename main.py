import os
from tqdm import tqdm
from PIL import Image, ImageFilter
from io import BytesIO

import cv2
import requests
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from sklearn.cluster import KMeans

from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing import image
from scipy.spatial import distance

product_path = 'product_images'
download_path = 'download_images'


# Loading and merging datasets
photos_df = pd.read_csv('dataset/photos.tsv000', sep='\t')
keywords_df = pd.read_csv('dataset/keywords.tsv000', sep='\t')
colors_df = pd.read_csv('dataset/colors.tsv000', sep='\t')

merge_df = pd.merge(keywords_df, colors_df, on='photo_id', how='outer')
df = pd.merge(photos_df, merge_df, on='photo_id', how='outer')
df.to_csv('dataset/data.csv', index=False)

df = pd.read_csv('dataset/data.csv')

df_ = df.sample(1000000)
# Dropping columns
drop_cls = ['photo_id', 'photo_url', 'photo_submitted_at', 'photo_featured', 'photographer_username', 'photographer_first_name', 'photographer_last_name', 'exif_camera_make', 'exif_camera_model', 'exif_iso', 'exif_aperture_value', 'exif_focal_length', 'exif_exposure_time', 'photo_location_name', 'photo_location_latitude', 'photo_location_longitude', 'photo_location_country', 'photo_location_city', 'ai_primary_landmark_name', 'ai_primary_landmark_latitude', 'ai_primary_landmark_longitude', 'ai_primary_landmark_confidence', 'blur_hash', 'ai_service_1_confidence', 'ai_service_2_confidence', 'suggested_by_user', 'ai_coverage', 'ai_score']
df_ = df_.drop(columns=drop_cls)
# Comining keyword columns
df_['keywords'] = df_[['photo_description', 'ai_description', 'keyword_x', 'keyword_y']].apply(lambda x: ' '.join(x.dropna()), axis=1)
df_ = df_.drop(columns=['photo_description', 'ai_description', 'keyword_x', 'keyword_y'])


'''
# Load pre-trained MobileNet model
mobilenet_model = MobileNet(weights='imagenet')
'''
tags = ['background']


for images in tqdm(os.listdir(product_path)):
    product_image_path = os.path.join(product_path, images)
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
    plt.imsave(f'plots/{os.path.splitext(images)[0]}_mask.jpg', masked_image, cmap='gray')
    
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
    
    img = image.load_img(product_image_path, target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x.copy())
    
    '''
    # Predicting image category
    preds = mobilenet_model.predict(x)
    # Decoding predictions
    decoded_preds = decode_predictions(preds, top=10)[0]
    tags = [label for (_, label, _) in decoded_preds]
    '''
    
    hexcodes = [''.join(c.upper() if c.isalpha() else c for c in s.lstrip('#')) for s in hexcodes]


image_urls = []

# Convert hexcodes to RGB values
target_rgb_values = []
for hexcode in hexcodes:
    rgb_value = tuple(int(hexcode[i:i+2], 16) for i in (0, 2, 4))
    target_rgb_values.append(rgb_value)

# Function to calculate contrast ratio between two colors
def contrast_ratio(color1, color2):
    luminance1 = 0.2126 * color1[0] + 0.7152 * color1[1] + 0.0722 * color1[2]
    luminance2 = 0.2126 * color2[0] + 0.7152 * color2[1] + 0.0722 * color2[2]
    if luminance1 > luminance2:
        return (luminance1 + 0.05) / (luminance2 + 0.05)
    else:
        return (luminance2 + 0.05) / (luminance1 + 0.05)

# Iterate over each image URL
for index, row in df_.iterrows():
    ratios = []
    
    # Extract the RGB values from the DataFrame
    rgb_value = (row['red'], row['green'], row['blue'])
    
    # Calculate Euclidean distance between image RGB values and target RGB values
    distances = [distance.euclidean(target_rgb, rgb_value) for target_rgb in target_rgb_values if not np.isnan(rgb_value).any() and not np.isnan(target_rgb).any()]
    
    # Check if any tag matches the target tags
    if any(tag in row['keywords'] for tag in tags):
        # Calculate contrast ratio between RGB value and target colors

        for target_rgb in target_rgb_values:
            # Check for NaN values and skip if present
            if np.isnan(rgb_value).any() or np.isnan(target_rgb).any():
                continue
            ratios.append(contrast_ratio(target_rgb, rgb_value))
        
        if any(distance <= 5 for distance in distances) and any(ratio > 4.5 for ratio in ratios):  # Set your distance threshold
            image_urls.append(row['photo_image_url'])
            
# Download images
temp_path = os.path.join(download_path, os.path.splitext(images)[0])
os.makedirs(temp_path, exist_ok=True)

# Download only 10 random images for each product image
i = 0
blur_radius = 30

for image_url in random.sample(image_urls, min(10, len(image_urls))):
    i += 1
    response = requests.get(image_url)
    if response.status_code == 200:
        # Save the image
        with open(os.path.join(temp_path, str(i)) + '.jpg', 'wb') as f:
            f.write(response.content)
        
        # Generating product shots
        background_image = Image.open(BytesIO(response.content))
        # Resize product image to fit background
        product_resize = cv2.resize(product_image, (background_image.width, background_image.height))
        
        # Convert product image to RGBA (add alpha channel)
        product_rgba = cv2.cvtColor(product_resize, cv2.COLOR_BGR2RGBA)
        # Convert product image to PIL format
        product_pil = Image.fromarray(product_rgba)
        # Apply Gaussian blur to background
        blurred_background = background_image.filter(ImageFilter.GaussianBlur(blur_radius))
        
        # Blend product with blurred background
        product_shot = Image.alpha_composite(blurred_background.convert('RGBA'), product_pil)
    