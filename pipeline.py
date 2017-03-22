# code samples taken from udacity class
import argparse
import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import features

path_out    = "output_images/"

calib = pickle.load(open("calibration.p","rb"))
mtx   = calib['mtx']
dist  = calib['dist']

class Vehicle():
    def __init__(self):
        self.heatmap = []
        self.frame   = 0
        self.bbox_list = []

vehicle          = Vehicle()

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def creat_heatmap(heatmap, bbox_list):
    for box in bbox_list:
        # add +1 for each box
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap

def find_cars(img, ystart, ystop, scale, svc, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, color_space='RGB2YCrCb'):
    
    draw_img    = np.copy(img)
    # based on trained classifier
    img         = img.astype(np.float32)

    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = features.color_transform(img_tosearch, color_space=color_space)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]
  
    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1 
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window              = 64
    nblocks_per_window  = (window // pix_per_cell)-1 
    cells_per_step      = 2
    nxsteps             = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps             = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = features.extract_hog_feature(ch1, orient, pix_per_cell, cell_per_block)
    hog2 = features.extract_hog_feature(ch2, orient, pix_per_cell, cell_per_block)
    hog3 = features.extract_hog_feature(ch3, orient, pix_per_cell, cell_per_block)

    bbox_list = []

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            area       = np.index_exp[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window]
            hog1_patch = hog1[area]
            hog2_patch = hog2[area]
            hog3_patch = hog3[area]
            hog_feat1 = hog1_patch.ravel()
            hog_feat2 = hog2_patch.ravel()
            hog_feat3 = hog3_patch.ravel()

            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop  = ypos*pix_per_cell

            img_part   = ctrans_tosearch[ytop:ytop+window, xleft:xleft+window]
#            img_search = img_tosearch[ytop:ytop+window, xleft:xleft+window]
#            plt.imshow(img_search)
#            plt.show()
#            plt.close()

            # Extract the image patch
            subimg = cv2.resize(img_part, (64,64))

#            plt.imshow(subimg)
#            plt.show()
#            plt.close()
          
            bin_color_features = features.combine_features(subimg, size=spatial_size)

            # combine features and make a prediction
            test_features   = np.hstack((bin_color_features, hog_features)).reshape(1, -1)
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                bbox_list.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                
    return draw_img, bbox_list
    
def image_test(file_path):
    img          = mpimg.imread(file_path)
#    img          = cv2.undistort(np.copy(img), mtx, dist, None, mtx)

    return process_frame(img)

def process_frame(img):

#    img          = cv2.undistort(img, mtx, dist, None, mtx)

    ystart = 400
    ystop  = 656

    orient          = 9
    pix_per_cell    = 8
    cell_per_block  = 2

    spatial_size    = (32,32)
    hist_bins       = 32

    if vehicle.frame == 0:
        scale  = 1.5
        out_img, vehicle.bbox_list = find_cars(img, ystart, ystop, scale, svc, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

        if verbose:
            plt.imshow(out_img)
            plt.savefig(path_out+"result_find_cars_scale1.jpg")
            plt.show()
            plt.close()

        scale  = 1.1
        out_img, bbox_list2 = find_cars(img, ystart, ystop, scale, svc, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        vehicle.bbox_list += bbox_list2

        if verbose:
            plt.imshow(out_img)
            plt.savefig(path_out+"result_find_cars_scale2.jpg")
            plt.show()
            plt.close()

    vehicle.frame += 1
    if vehicle.frame == 10:
        vehicle.frame = 0

    # tip from udacity class
    from scipy.ndimage.measurements import label

    heatmap = np.zeros_like(img[:,:,0]).astype(np.float)
    heatmap = creat_heatmap(heatmap, vehicle.bbox_list)
    heatmap = apply_threshold(heatmap, 3)
    heatmap = np.clip(heatmap, 0, 255)
    labels  = label(heatmap)
    print(labels[1], 'cars found')

    # put boxes around labels
    draw_img = draw_labeled_bboxes(np.copy(img), labels)

    if verbose:
        fig = plt.figure()
        plt.subplot(131)
        plt.imshow(draw_img)
        plt.title('Car Positions')
        plt.subplot(132)
        plt.imshow(heatmap, cmap='hot')
        plt.title('Heat Map')
        plt.subplot(133)
        plt.imshow(labels[0], cmap='gray')
        plt.title('Gray labels')
        fig.tight_layout()
        plt.savefig(path_out+"result_heatmap.jpg")
        plt.show()
        plt.close()
    
    return draw_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Advanced lane line finding')
    parser.add_argument(
        'file',
        type=str,
        help='path to image or video file.'
    )
    parser.add_argument(
        '-v',
        '--verbose',
        default=False,
        action='store_true',
        help='verbose mode with lots of plots.'
    )

    args    = parser.parse_args()
    verbose = args.verbose

    f_svc   = open("svc.p", "rb")
    svc     = pickle.load(f_svc)

    if "mp4" in args.file:
        from moviepy.editor import VideoFileClip

        output          = 'result_' + args.file
        video_file      = VideoFileClip(args.file)
        process_clip    = video_file.fl_image(process_frame)
        process_clip.write_videofile(output, audio=False)

    if ".jpg" in args.file or ".png" in args.file:
        image_test(args.file)