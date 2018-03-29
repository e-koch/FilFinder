# Licensed under an MIT open source license - see LICENSE

"""
Utility functions for fil-finder package


"""

import itertools
import numpy as np


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]

##########################################################################
# Simple fcns used throughout module
##########################################################################


def eight_con():
    return np.ones((3, 3))


def distance(x, x1, y, y1):
    return np.sqrt((x - x1) ** 2.0 + (y - y1) ** 2.0)


def pad_image(image, extents, pad_size, constant=0):
    '''
    Figure out where an image needs to be padded based on pixel extents.
    '''

    # Now we look for where the extents with padding will run into a boundary
    pad_vector = []
    for i, extent in enumerate(zip(*extents)):

        lower = extent[0] - pad_size
        if lower > 0:
            lower = 0

        # First axis upper
        upper = extent[1] + pad_size - image.shape[i] + 1
        if upper < 0:
            upper = 0

        pad_vector.append((-lower, upper))

    return np.pad(image, pad_vector, mode='constant', constant_values=constant)


def shifter(l, n):
    return l[n:] + l[:n]


def product_gen(n):
    for r in itertools.count(1):
        for i in itertools.product(n, repeat=r):
            yield "".join(i)


def red_chisq(data, fit, nparam, sd):
    N = data.shape[0]
    stat = np.sum(((fit - data) / sd) ** 2.) / float(N - nparam - 1)
    if hasattr(stat, 'unit'):
        stat = stat.value
    return stat


def in_ipynb():
    try:
        cfg = get_ipython().config
        if cfg['IPKernelApp']['parent_appname'] == 'ipython-notebook':
            return True
        else:
            return False
    except NameError:
        return False


def round_to_odd(x):
    return int((np.ceil((np.ceil(x) / 2) + 0.5) * 2) - 1)


def threshold_local(image, *args, **kwargs):
    '''
    skimage changed threshold_adaptive to threshold_local. This wraps both
    to ensure the same behaviour with old and new versions.
    '''
    try:
        from skimage.filters import threshold_local
        mask = image > threshold_local(image, *args, **kwargs)
    except ImportError:
        from skimage.filters import threshold_adaptive
        mask = threshold_adaptive(image, *args, **kwargs)

    return mask
