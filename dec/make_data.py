import sys
import os
import Image
import numpy as np
import cv2
import cv
from joblib import Parallel, delayed
import features
import dec

def dispImg(X, n, fname=None):
  h = X.shape[1]
  w = X.shape[2]
  c = X.shape[3]
  buff = np.zeros((n*h, n*w, c), dtype=np.uint8)

  for i in xrange(n):
    for j in xrange(n):
      buff[i*h:(i+1)*h, j*w:(j+1)*w, :] = X[i*n+j]

  if fname is None:
    cv2.imshow('a', buff)
    cv2.waitKey(0)
  else:
    cv2.imwrite(fname, buff)

def load_data(images_dir):
    ims = [read(os.path.join(images_dir, filename)) for filename in os.listdir(images_dir)]
    X = np.array(ims, dtype='uint8')
    dispImg(X[:100, :, :, [2, 1, 0]], 10)
    n_jobs = 10
    cmap_size = (8, 8)
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

if __name__ == '__main__':
    images_dir = sys.argv[1]
    read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
    if os.path.isdir(images_dir):
        X_train, img_train = load_data(images_dir)
        p = np.random.permutation(X_train.shape[0])
        X_total = X_train[p]
        img_total = img_train[p]
        Y = np.zeros((X_total.shape[0],))
        dec.write_db(X_total, Y, 'custom_total')
        dec.write_db(img_total, Y, 'custom_img')
        N = X_total.shape[0] * 4 / 5
        dec.write_db(X_total[:N], Y[:N], 'custom_train')
        dec.write_db(X_total[N:], Y[N:], 'custom_test')
    else:
        raise Exception("Please specific image url")
