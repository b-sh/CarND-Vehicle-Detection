import numpy as np
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import features

# pipline output images
path_image  = "test_images/"
path_out    = "output_images/"

def train(car_features, notcar_features):
    y = np.hstack((np.ones(len(car_features)), 
        np.zeros(len(notcar_features))))

    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    from sklearn.model_selection import train_test_split
    # using 70% for training and 30% for validation
    rnd = np.random.randint(0, 100)
#   from sklearn.utils import check_random_state
#    rnd = check_random_state(1)
    X_train, X_test, y_train, y_test = train_test_split(
                                            X, y,
                                            test_size=0.3,
                                            random_state=rnd)

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    # http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
    svc = Pipeline([('scaler', StandardScaler()), ('clf', LinearSVC())])
    svc.fit(X_train, y_train)

    print('Train Accuracy of SVC = ', svc.score(X_train, y_train))
    print('Test Accuracy of SVC = ', svc.score(X_test, y_test))
    print('My SVC predicts: ', svc.predict(X_test[0:10]))
    print('For labels: ', y_test[0:10])
    f_svc = open("svc.p", 'wb')
#    data = {}
#    data['svc'] = svc
#    pickle.dump(data, f_svc)
    pickle.dump(svc, f_svc)

if __name__ == "__main__":
    cars = glob.glob('data/vehicles/**/*.png')
    notcars = glob.glob('data/non-vehicles/**/*.png')

#    car_features     = features.features_from_img_list(cars, size=(64, 64))
#    notcar_features  = features.features_from_img_list(notcars, size=(64, 64))
    car_features     = features.features_from_img_list(cars, color_space='RGB2YCrCb', orient=9, pix_per_cell=8, cell_per_block=2)
    notcar_features  = features.features_from_img_list(notcars, color_space='RGB2YCrCb', orient=9, pix_per_cell=8, cell_per_block=2)

    train(car_features, notcar_features)