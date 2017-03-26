# using udacity class snippets
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob
from skimage.feature import hog
from skimage import color, exposure
from sklearn.preprocessing import StandardScaler

# pipline output images
path_image  = "test_images/"
path_out    = "output_images/"

def color_transform(img, color_space='RGB'):
    if color_space != 'RGB':
      if color_space == 'HSV':
          feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
      elif color_space == 'LUV':
          feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
      elif color_space == 'HLS':
          feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
      elif color_space == 'YUV':
          feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
      elif color_space == 'RGB2YCrCb':
          feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
      elif color_space == 'BGR2YCrCb':
          feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    else: feature_image = np.copy(img)
    return feature_image

# Define a function to compute color histogram features  
def color_hist2(img, nbins=32, bins_range=(0, 256), color_space='RGB'):
    # Compute the histogram of the RGB channels separately
    ch1_hist = None
    ch2_hist = None
    ch3_hist = None
    # Generating bin centers
    bin_centers = None
    # Concatenate the histograms into a single feature vector
    hist_features = None

    # Return the individual histograms, bin_centers and feature vector
    feature_image = color_transform(img, color_space=color_space)

    ch1_hist = np.histogram(feature_image[:,:,0], nbins, bins_range)
    ch2_hist = np.histogram(feature_image[:,:,1], nbins, bins_range)
    ch3_hist = np.histogram(feature_image[:,:,2], nbins, bins_range)
    
    bin_edges = ch1_hist[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2

    hist_features = np.concatenate((ch1_hist[0], ch2_hist[0], ch3_hist[0]))
    
    return ch1_hist, ch2_hist, ch3_hist, bin_centers, hist_features

def bin_spatial2(img, color_space='RGB', size=(32, 32)):
    # Convert image to new color space (if specified)
    feature_image = color_transform(img, color_space)
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(feature_image, size).ravel() 
    # Return the feature vector
    return features

def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))
                        
def color_hist(img, nbins=32):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # return single feature vector
    return hist_features

def extract_hog_feature(img, orient, pix_per_cell, cell_per_block, feature_vec=False):
    return hog(img, 
                              orientations=orient, 
                              pixels_per_cell=(pix_per_cell, pix_per_cell),
                              cells_per_block=(cell_per_block, cell_per_block), 
                              transform_sqrt=False,
                              visualise=False,
                              feature_vector=feature_vec)

def get_hog_features(img, orient, pix_per_cell, cell_per_block, feature_vec=True):
    features, hog_image = hog(img,
                              orientations=orient, 
                              pixels_per_cell=(pix_per_cell, pix_per_cell),
                              cells_per_block=(cell_per_block, cell_per_block), 
                              transform_sqrt=False,
                              visualise=True,
                              feature_vector=feature_vec)
    return features, hog_image

def combine_features2(img, color_space='RGB', size=(32,32)):
    ch1_h, ch2_h, ch3_h, bincen, color_features = color_hist(img, color_space=color_space)
    bin_feat                                    = bin_spatial(img, color_space=color_space, size=size)
    
    return np.concatenate((bin_feat, color_features))

def combine_features(img, size=(32,32), nbins=32):
    color_features = color_hist(img, nbins=nbins)
    bin_feat       = bin_spatial(img, size=size)
    
    return np.concatenate((bin_feat, color_features))

def features_from_img_list(img_list, color_space='RGB2YCrCb', size=(32,32), orient=9, pix_per_cell=8, cell_per_block=2):
    features        = []

    for img in img_list:
        image   = mpimg.imread(img)*255

        img     = color_transform(np.copy(image), color_space=color_space)

        ch1 = img[:,:,0]
        ch2 = img[:,:,1]
        ch3 = img[:,:,2]

        hog_feat1 = extract_hog_feature(ch1, orient, pix_per_cell, cell_per_block, True)
        hog_feat2 = extract_hog_feature(ch2, orient, pix_per_cell, cell_per_block, True)
        hog_feat3 = extract_hog_feature(ch3, orient, pix_per_cell, cell_per_block, True)

        bin_color_features = combine_features(img, size=size)

        all_features = np.concatenate((bin_color_features, hog_feat1, hog_feat2, hog_feat3))

        features.append(all_features)
    return features

def normalize_features(feature_list):
  X = np.vstack(feature_list).astype(np.float64)
  
  # Fit a per-column scaler
  X_scaler = StandardScaler().fit(X)
  # Apply the scaler to X
  scaled_X = X_scaler.transform(X)
  return X, scaled_X

if __name__ == "__main__":
    # Read in the image
    image = mpimg.imread(path_image+'cutout1.jpg')

    print(image)
    print(image.shape)

    rh, gh, bh, bincen, feature_vec = color_hist2(image, nbins=32, bins_range=(0, 256))

    # Plot a figure with all three bar charts
    fig = plt.figure(figsize=(12,3))
    plt.subplot(131)
    plt.bar(bincen, rh[0])
    plt.xlim(0, 256)
    plt.title('R Histogram')
    plt.subplot(132)
    plt.bar(bincen, gh[0])
    plt.xlim(0, 256)
    plt.title('G Histogram')
    plt.subplot(133)
    plt.bar(bincen, bh[0])
    plt.xlim(0, 256)
    plt.title('B Histogram')

    fig.tight_layout()
    plt.savefig(path_out+"cutout_histograms.jpg")
    plt.show()
    plt.close()

    pix_per_cell = 8
    cell_per_block = 2
    orient = 9

#    img_hog = color_transform(np.copy(image), color_space='RGB2YCrCb')
#    img_hog = color_transform(np.copy(image), color_space='LUV')
#    img_hog = color_transform(np.copy(image), color_space='RGB')
#    img_hog = color_transform(np.copy(image), color_space='HSV')
    # looks good
#    img_hog = color_transform(np.copy(image), color_space='HLS')

    # also good
    img_hog = color_transform(np.copy(image), color_space='YUV')

    ch1 = img_hog[:,:,0]
    ch2 = img_hog[:,:,1]
    ch3 = img_hog[:,:,2]

    features, hog_feat1 = hog(ch1, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block, cell_per_block), visualise=True, feature_vector=True)
    features, hog_feat2 = hog(ch2, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block, cell_per_block), visualise=True, feature_vector=True)
    features, hog_feat3 = hog(ch3, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block, cell_per_block), visualise=True, feature_vector=True)
    # Plot the examples
    fig = plt.figure(figsize=(12,4))
    plt.subplot(141)
    plt.imshow(image, cmap='gray')
    plt.title('Car Image')
    plt.subplot(142)
    plt.imshow(hog_feat1, cmap='gray')
    plt.title('HOG channel 1')
    plt.subplot(143)
    plt.imshow(hog_feat2, cmap='gray')
    plt.title('HOG channel 2')
    plt.subplot(144)
    plt.imshow(hog_feat3, cmap='gray')
    plt.title('HOG channel 3')
    plt.savefig(path_out+"skimage_hog.jpg")
    plt.show()
    plt.close()
