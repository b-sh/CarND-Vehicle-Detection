import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob
from skimage.feature import hog
from skimage import color, exposure
import hog

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
    # Just for fun choose random car / not-car indices and plot example images   
    car_ind     = np.random.randint(0, len(cars))
    notcar_ind  = np.random.randint(0, len(notcars))
        
    # Read in car / not-car images
    car_image = mpimg.imread(cars[car_ind])
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

    car_image *=255
    notcar_image *=255

    rh_car, gh_car, bh_car, bincen_car, feature_vec_car = hog.color_hist(car_image, nbins=32, bins_range=(0, 256))
    rh_noncar, gh_noncar, bh_noncar, bincen_noncar, feature_vec_noncar = hog.color_hist(notcar_image, nbins=32, bins_range=(0, 256))

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

    feature = hog.combine_features(car_image)
    X, scaled_X  = hog.normalize_features([feature])

    # Plot an example of raw and scaled features
    fig = plt.figure(figsize=(12,4))
    plt.subplot(131)
    plt.imshow(car_image)
    plt.title('Original Image')
    plt.subplot(132)
    plt.plot(X[0])
    plt.title('Raw Features')
    plt.subplot(133)
    plt.plot(scaled_X[0])
    plt.title('Normalized Features')
    fig.tight_layout()
    plt.show()
    plt.savefig(path_out+"normalized_features.jpg")
    plt.close()

#    for image in os.listdir(path_image):
#        path_image   = "test_images/"
#        image        = "perspectivestraight_lines1.jpg"
#        image        = "perspectivestraight_lines2.jpg"
#        image        = "perspective_origtest5.jpg"
#        image        = "signs.png"
#        image        = "test5.jpg"
#        image        = "straight_lines1.jpg"
#        image        = "perspectivetest5.jpg"
#        plt_img      = mpimg.imread(path_image+image)
#        img          = cv2.imread(path_image+image)

#        plt.imshow(stack2, cmap='gray')
#        plt.show()

#        plt.imshow(color_binary, cmap='gray')
#        plt.show()
#        plt.title(image)
#        plt.imshow(combined, cmap='gray')
#        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
#        f.tight_layout()
#        ax1.imshow(plt_img)
#        ax1.set_title('Original Image ' + image , fontsize=12)
#        ax2.imshow(plt_img, cmap='gray')
#        ax2.set_title('Thresholded', fontsize=12)
#        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
#        plt.imshow(combined2, cmap='gray')
#        plt.imshow(combined)
#        plt.imshow(dir_binary, cmap='gray')
#        plt.imshow(combined, cmap='gray')
#        plt.savefig('test.jpg')
#        plt.show()

