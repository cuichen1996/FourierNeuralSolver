# %%
import platform

import numpy as np
from scipy.ndimage import gaussian_filter
from six.moves import cPickle as pickle
from skimage.color import rgb2gray
from skimage.transform import resize


# %%
def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == "2":
        return pickle.load(f)
    elif version[0] == "3":
        return pickle.load(f, encoding="latin1")
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, "rb") as f:
        datadict = load_pickle(f)
        X = datadict["data"]
        Y = datadict["labels"]
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y

# %%
x_train, _ = load_CIFAR_batch("data/data/cifar-10-batches-py/data_batch_1")
x_train_gray = rgb2gray(x_train)
# %%
def generate_kappa(n, m, smooth=False, threshold=50, kernel=3):
    # choose a random sample from CIFAR10
    index = np.random.randint(0, x_train.shape[0])
    sample = x_train_gray[index]

    # resize
    sample = resize(sample, (n, m))

    # smooth
    if smooth == True:
        sample = gaussian_filter(sample, sigma=kernel)

    # sample ∈ [random threshold, 1]
    threshold = 0.01 * threshold
    sample_normal = threshold + (((1.0 - threshold) * (sample - np.min(sample))) / (np.max(sample) - np.min(sample)))
    sample_normal1 = np.array(sample_normal, dtype=np.float64)

    return sample_normal1