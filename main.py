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
final_path = 'final_shots'


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
    print(image, '\n')
    
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
    masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
    
    plt.imshow(masked_image, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.imsave(f'plots/{os.path.splitext(images)[0]}_mask.jpg', masked_image)
    
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
        facecolor = '#%02x%02x%02x' % (int(cluster_center[0]), int(cluster_center[1]), int(cluster_center[2]))
        ax.add_patch(patches.Rectangle((x_from, 0.05), rectangle_width, 0.9, alpha=None, facecolor=facecolor))
        x_from = x_from + rectangle_width + gap
        hexcodes.append(facecolor)
    
    # Plot the last rectangle (adjusted to fit within the plot boundaries)
    cluster_center = kmeans.cluster_centers_[sort_ix[4]]
    facecolor = '#%02x%02x%02x' % (int(cluster_center[0]), int(cluster_center[1]), int(cluster_center[2]))
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



    '''
    # Removing shadows
    lab_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # Extract lightness channel
    lightness, a, b = cv2.split(lab_img)
    # Apply bilateral filtering to preserve edges while smoothing shadows
    bilateral_filtered = cv2.bilateralFilter(lightness, 9, 75, 75)
    
    # Check and potentially resize for consistent size
    if bilateral_filtered.shape != a.shape:
        bilateral_filtered = cv2.resize(bilateral_filtered, dsize=a.shape)
    
    # Verify and potentially convert data types
    if bilateral_filtered.dtype != a.dtype:
        bilateral_filtered = cv2.convertScaleAbs(bilateral_filtered, alpha=1.0, beta=0.0)
    
    # Merge back wiith original a* and b* channels
    lab_img = cv2.merge((bilateral_filtered, a, b))
    # Convert back to BGR color space
    img_ = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)
    cv2.imwrite('_.png', img_)
    '''
    
    # Removing background
    product_image_bgra = cv2.cvtColor(product_image, cv2.COLOR_BGR2BGRA)
    # Create nask
    mask_ = cv2.inRange(product_image_bgra, (220, 220, 220, 220), (255, 255, 255, 255))
    # Invert mask
    mask_ = cv2.bitwise_not(mask_)
    # remove white
    image_ = cv2.bitwise_and(product_image_bgra, product_image_bgra, mask=mask_)
    image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGBA)
    
    print(f'background removed for {images}', '\n')
    
    
    
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
            
            if any(distance <= 8 for distance in distances) and any(ratio > 3.5 for ratio in ratios):  # Set your distance threshold
                image_urls.append(row['photo_image_url'])
                
    # Download images
    temp_path_1 = os.path.join(download_path, os.path.splitext(images)[0])
    temp_path_2 = os.path.join(final_path, os.path.splitext(images)[0])
    os.makedirs(temp_path_1, exist_ok=True)
    os.makedirs(temp_path_2, exist_ok=True)
    
    print(f'downloading images for {images}', '\n')
    
    # Download only 10 random images for each product image
    i = 0
    blur_radius = 7.5
    
    for image_url in random.sample(image_urls, min(20, len(image_urls))):
        i += 1
        response = requests.get(image_url)
        if response.status_code == 200:
            # Save the image
            with open(os.path.join(temp_path_1, str(i)) + '.jpg', 'wb') as f:
                f.write(response.content)
            
            # Generating product shots
            foreground_image = Image.fromarray(image_)
            foreground_image.save("_.png", "PNG")
            
            background_image = Image.open(BytesIO(response.content))
            
            # Calculate the aspect ratio of the product and background images
            product_aspect_ratio = foreground_image.width / foreground_image.height
            background_aspect_ratio = background_image.width / background_image.height
            # Resize the product image to fit within the dimensions of the background image while preserving aspect ratio
            if product_aspect_ratio > background_aspect_ratio:
                # Product image is wider than background image
                new_width = background_image.width
                new_height = int(new_width / product_aspect_ratio)
            else:
                # Product image is taller than or equal to the background image
                new_height = background_image.height
                new_width = int(new_height * product_aspect_ratio)
                
            foreground_image = foreground_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create a new image with the dimensions of the background image
            composite_image = Image.new('RGBA', background_image.size, (255, 255, 255, 0))
            
            # Calculate the position to paste the product image onto the composite image to center it
            offset = ((background_image.width - new_width) // 2, (background_image.height - new_height) // 2)
            
            # Paste the product image onto the composite image
            composite_image.paste(foreground_image, offset, mask=foreground_image.split()[3])
            
            # Apply Gaussian blur to the background
            blurred_background = background_image.filter(ImageFilter.GaussianBlur(blur_radius))
            
            # Blend the composite image with the blurred background
            final_image = Image.alpha_composite(blurred_background.convert('RGBA'), composite_image)
        
        
        
            
            path_ = os.path.join(temp_path_2, str(i)) + '.png'
            final_image.save(path_)