import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob
from skimage.feature import hog
from skimage import color, exposure
import features

# pipline output images
path_image  = "test_images/"
path_out    = "output_images/"

def data_look(car_list, notcar_list):
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    image = mpimg.imread(car_list[0])
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = image.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = image.dtype
    # Return data_dict
    return data_dict

if __name__ == "__main__":
    cars = glob.glob('data/vehicles/**/*.png')
    notcars = glob.glob('data/non-vehicles/**/*.png')

    data_info = data_look(cars, notcars)
    car_ind     = np.random.randint(0, len(cars))
    notcar_ind  = np.random.randint(0, len(notcars))
#    car_ind = 0
#    notcar_ind = 0
        
    # Read in car / not-car image
    car_image    = mpimg.imread(cars[car_ind])
    notcar_image = mpimg.imread(notcars[notcar_ind])

    # Plot the examples
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(car_image)
    plt.title('Example Car Image')
    plt.subplot(122)
    plt.imshow(notcar_image)
    plt.title('Example Not-car Image')
    plt.savefig(path_out+"examples_data.jpg")
    plt.show()
    plt.close()

    # scale accordently
    car_image       *=255
    notcar_image    *=255

    bin_feature_car     = features.bin_spatial(car_image)
    bin_feature_noncar  = features.bin_spatial(notcar_image)
    plt.subplot(121)
    plt.plot(bin_feature_car)
    plt.title('Spatially binned feature (Car)')
    plt.subplot(122)
    plt.plot(bin_feature_noncar)
    plt.title('Spatially binned feature (Not a car)')
    plt.savefig(path_out+"random_car_noncar_bin_feature.jpg")
    plt.show()
    plt.close()

    rh_car, gh_car, bh_car, bincen_car, feature_vec_car = features.color_hist2(car_image, nbins=32, bins_range=(0, 256))
    rh_noncar, gh_noncar, bh_noncar, bincen_noncar, feature_vec_noncar = features.color_hist2(notcar_image, nbins=32, bins_range=(0, 256))

    # plot car and non car histograms
    fig = plt.figure(figsize=(12,3))
    plt.subplot(231)
    plt.bar(bincen_car, rh_car[0])
    plt.xlim(0, 256)
    plt.title('R Histogram car')
    plt.subplot(232)
    plt.bar(bincen_car, gh_car[0])
    plt.xlim(0, 256)
    plt.title('G Histogram car')
    plt.subplot(233)
    plt.bar(bincen_car, bh_car[0])
    plt.xlim(0, 256)
    plt.title('B Histogram car')
    plt.subplot(234)
    plt.bar(bincen_noncar, rh_noncar[0])
    plt.xlim(0, 256)
    plt.title('R Histogram not car')
    plt.subplot(235)
    plt.bar(bincen_noncar, gh_noncar[0])
    plt.xlim(0, 256)
    plt.title('G Histogram not car')
    plt.subplot(236)
    plt.bar(bincen_noncar, bh_noncar[0])
    plt.xlim(0, 256)
    plt.title('B Histogram not car')
    fig.tight_layout()
    plt.savefig(path_out+"random_car_noncar_histograms.jpg")
    plt.show()
    plt.close()

    orient          = 9
    pix_per_cell    = 8
    cell_per_block  = 2

    color_spaces = ['RGB2YCrCb']
    color_spaces += ['LUV']
    color_spaces += ['RGB']
    color_spaces += ['HSV']
    color_spaces += ['HLS']
    color_spaces += ['YUV']
    for color_space in color_spaces:
        print(color_space)
        img_hog    = features.color_transform(np.copy(car_image), color_space=color_space)

        ch1 = img_hog[:,:,0]
        ch2 = img_hog[:,:,1]
        ch3 = img_hog[:,:,2]

        features_vec, hog_feat1 = hog(ch1, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block, cell_per_block), visualise=True, feature_vector=True)
        features_vec, hog_feat2 = hog(ch2, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block, cell_per_block), visualise=True, feature_vector=True)
        features_vec, hog_feat3 = hog(ch3, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block, cell_per_block), visualise=True, feature_vector=True)
        # Plot the examples
        fig = plt.figure(figsize=(12,4))
        plt.subplot(241)
        plt.imshow(car_image/255, cmap='gray')
        plt.title('Car Image')
        plt.subplot(242)
        plt.imshow(hog_feat1, cmap='gray')
        plt.title('HOG channel 1')
        plt.subplot(243)
        plt.imshow(hog_feat2, cmap='gray')
        plt.title('HOG channel 2')
        plt.subplot(244)
        plt.imshow(hog_feat3, cmap='gray')
        plt.title('HOG channel 3')

        img_hog    = features.color_transform(np.copy(notcar_image), color_space=color_space)

        ch1 = img_hog[:,:,0]
        ch2 = img_hog[:,:,1]
        ch3 = img_hog[:,:,2]

        features_vec, hog_feat1 = hog(ch1, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block, cell_per_block), visualise=True, feature_vector=True)
        features_vec, hog_feat2 = hog(ch2, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block, cell_per_block), visualise=True, feature_vector=True)
        features_vec, hog_feat3 = hog(ch3, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block, cell_per_block), visualise=True, feature_vector=True)

        plt.subplot(245)
        plt.imshow(notcar_image/255, cmap='gray')
        plt.title('Not Car Image')
        plt.subplot(246)
        plt.imshow(hog_feat1, cmap='gray')
        plt.title('HOG channel 1')
        plt.subplot(247)
        plt.imshow(hog_feat2, cmap='gray')
        plt.title('HOG channel 2')
        plt.subplot(248)
        plt.imshow(hog_feat3, cmap='gray')
        plt.title('HOG channel 3')
        
        fig.tight_layout()
        plt.savefig(path_out+"skimage_hog"+color_space+".jpg")
        plt.show()
        plt.close()

    car_features     = features.features_from_img_list(cars, color_space='RGB2YCrCb')
    notcar_features  = features.features_from_img_list(notcars, color_space='RGB2YCrCb')

    X, scaled_X      = features.normalize_features((car_features))

    # Plot an example of raw and scaled features
    fig = plt.figure(figsize=(12,4))
    plt.subplot(131)
    plt.imshow(car_image/255)
    plt.title('Original Image')
    plt.subplot(132)
    plt.plot(X[car_ind])
    plt.title('Raw Features')
    plt.subplot(133)
    plt.plot(scaled_X[car_ind])
    plt.title('Normalized Features')
    fig.tight_layout()
    plt.savefig(path_out+"normalized_features.jpg")
    plt.show()
    plt.close()
