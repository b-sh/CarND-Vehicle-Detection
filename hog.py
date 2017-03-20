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

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # make a copy of the image
    draw_img = np.copy(img)
    # draw each bounding box on your image copy using cv2.rectangle()
    # return the image copy with boxes drawn
    for box in bboxes:
      cv2.rectangle(draw_img, (box[0][0], box[0][1]), (box[1][0], box[1][1]), color, thick)
    return draw_img # Change this line to return image copy with boxes

# Define a function that takes an image and a list of templates as inputs
# then searches the image and returns the a list of bounding boxes 
# for matched templates
def find_matches(img, template_list):
    # Make a copy of the image to draw on
    # Define an empty list to take bbox coords
    bbox_list = []
    # Iterate through template list
    # Read in templates one by one
    # Use cv2.matchTemplate() to search the image
    #     using whichever of the OpenCV search methods you prefer
    # Use cv2.minMaxLoc() to extract the location of the best match
    # Determine bounding box corners for the match
    # Return the list of bounding boxes
    for template in template_list:
        image_templ = mpimg.imread(template)
        h, w, c = image_templ.shape
        res = cv2.matchTemplate(img, image_templ, cv2.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        bbox_list.append((max_loc,(max_loc[0] + w, max_loc[1] + h)))
    return bbox_list

def find_matches(img, template_list):
    # Define an empty list to take bbox coords
    bbox_list = []
    # Define matching method
    # Other options include: cv2.TM_CCORR_NORMED', 'cv2.TM_CCOEFF', 'cv2.TM_CCORR',
    #         'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED'
    method = cv2.TM_CCOEFF_NORMED
    # Iterate through template list
    for temp in template_list:
        # Read in templates one by one
        tmp = mpimg.imread(temp)
        # Use cv2.matchTemplate() to search the image
        result = cv2.matchTemplate(img, tmp, method)
        # Use cv2.minMaxLoc() to extract the location of the best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        # Determine a bounding box for the match
        w, h = (tmp.shape[1], tmp.shape[0])
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        # Append bbox position to list
        bbox_list.append((top_left, bottom_right))
        # Return the list of bounding boxes
        
    return bbox_list

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
    else: feature_image = np.copy(img)
    return feature_image

# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256), color_space='RGB'):
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

def bin_spatial(img, color_space='RGB', size=(32, 32)):
    # Convert image to new color space (if specified)
    feature_image = color_transform(img, color_space)
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(feature_image, size).ravel() 
    # Return the feature vector
    return features

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    features, hog_image = hog(img, 
                              orientations=orient, 
                              pixels_per_cell=(pix_per_cell, pix_per_cell),
                              cells_per_block=(cell_per_block, cell_per_block), 
                              transform_sqrt=False, 
                              visualise=vis,
                              feature_vector=False)
    return features, hog_image

def combine_features(img, color_space='RGB', size=(32,32)):
  ch1_h, ch2_h, ch3_h, bincen, color_features = color_hist(img, color_space=color_space)
  bin_feat                                    = bin_spatial(img, color_space=color_space, size=size)
  
#  print(color_features.shape, bin_feat.shape)
#  print(type(color_features), type(bin_feat))

  feature = []
  feature = np.concatenate((bin_feat, color_features))

  return feature

def normalize_features(feature_list):
  # Create an array stack, NOTE: StandardScaler() expects np.float64
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

    rh, gh, bh, bincen, feature_vec = color_hist(image, nbins=32, bins_range=(0, 256))

    # Plot a figure with all three bar charts
    if rh is not None:
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

#        plt.subplot(144)
#        plt.bar(bincen, feature_vec)
#        plt.xlim(0, 256)
#        plt.title('Feature vector')

        fig.tight_layout()
        plt.savefig(path_out+"cutout_histograms.jpg")
        plt.show()
        plt.close()

        pix_per_cell = 8
        cell_per_block = 2
        orient = 9

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        features, hog_image = hog(gray, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block, cell_per_block), visualise=True, feature_vector=False)
        # Plot the examples
        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(image, cmap='gray')
        plt.title('Example Car Image')
        plt.subplot(122)
        plt.imshow(hog_image, cmap='gray')
        plt.title('HOG Visualization')
        plt.savefig(path_out+"skimage_hog.jpg")
        plt.show()
        plt.close()

#    bboxes = find_matches(image, templist)
#    result = draw_boxes(image, bboxes)
#    plt.imshow(result)

#    for image in os.listdir(path_image):
#        path_image   = "test_images/"
#        plt_img      = mpimg.imread(path_image+image)
#        img          = cv2.imread(path_image+image)

