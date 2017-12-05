import sys
import os
import Image
import numpy as np
import cv2
import cv
from joblib import Parallel, delayed
import features
import random
import dec
import pdb

image_size = (200, 120)

def mode():
    if "MODE" in os.environ:
        return os.environ['MODE']
    return 'validate'


def load_data(images_dir):
    ims = [read(os.path.join(images_dir, filename)) for filename in os.listdir(images_dir)]
    X = np.array(ims, dtype='uint8')
   # dispImg(X[:100, :, :, [2, 1, 0]], 10)
    n_jobs = 10
    cmap_size = (6, 10)
    N = X.shape[0]

    H = np.asarray(Parallel(n_jobs=n_jobs)(delayed(features.hog)(X[i]) for i in xrange(N)))

    H = H.reshape((H.shape[0], H.size / N))

    X_small = np.asarray(Parallel(n_jobs=n_jobs)(delayed(cv2.resize)(X[i], cmap_size) for i in xrange(N)))
    crcb = np.asarray(Parallel(n_jobs=n_jobs)(delayed(cv2.cvtColor)(X_small[i], cv.CV_RGB2YCrCb) for i in xrange(N)))
    crcb = crcb[:, :, :, 1:]
    crcb = crcb.reshape((crcb.shape[0], crcb.size / N))

    feature = np.concatenate(((H - 0.2) * 10.0, (crcb - 128.0) / 10.0), axis=1)
    print feature.shape

    return feature, X[:, :, :, [2, 1, 0]]


def load_label(images_dir, classes, determine):
    return np.array([get_label(classes, filename, determine) for filename in os.listdir(images_dir)], dtype='uint8')


def get_label(classes, filename, determine):
    if mode() == 'validate':
        return classes.index(filename.split(determine)[0])
    return 0


def load_named_label(images_dir):
    return np.array([filename for filename in os.listdir(images_dir)], dtype='str')

if __name__ == '__main__':
    classes = ["heritage", "being", "scenery"]
    images_dir = sys.argv[1]
    read = lambda imname: np.asarray(Image.open(imname).convert("RGB").resize(image_size, Image.ANTIALIAS))
    if os.path.isdir(images_dir):
        X_train, img_train = load_data(images_dir)
        Y = load_label(images_dir, classes, "_")
        labeled = load_named_label(images_dir)
        p = np.random.permutation(X_train.shape[0])
        X_total = X_train[p]
        Y_total = Y[p]
        labeled_total = labeled[p]
        np.save("custom_named_label", labeled_total)
        img_total = img_train[p]
        dec.write_db(X_total, Y_total, 'custom_total')
        dec.write_db(img_total, Y_total, 'custom_img')
        N = X_total.shape[0] * 4 / 5
        dec.write_db(X_total[:N], Y_total[:N], 'custom_train')
        dec.write_db(X_total[N:], Y_total[N:], 'custom_test')
    else:
        raise Exception("Please specific image url")
