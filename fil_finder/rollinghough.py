'''

Rolling Hough Transform Implementation
See  Clark et al. 2013 for description

'''

import numpy as np
import scipy.ndimage as nd
from scipy.stats import scoreatpercentile
# from skimage.transform import hough_line, hough_line_peaks
import matplotlib.pyplot as p

def rht(mask, radius, ntheta=180):
    '''

    INPUTS
    ------

    mask - bool array

    '''

    circle, mesh = circular_region(radius)
    pad_mask = np.pad(mask.astype(float), radius, padwithnans)
    theta = np.linspace(0,np.pi, ntheta)

    R = np.empty((ntheta,))
    x, y = np.where(mask!=0.0)
    for i,j in zip(x,y):
        region = circle * pad_mask[i:i+2*radius+1,j:j+2*radius+1]
        for posn, ang in enumerate(theta):
            diff = mesh[0]*np.sin(ang) - mesh[1]*np.cos(ang)
            diff[np.where(np.abs(diff)<1.0)] = 0

            line = region * np.isclose(diff, 0.0)

            R[posn] += np.nansum(line)
    R /= float(ntheta)
    return theta, R



def threshold(img, radius, threshold):
    smoothed = nd.black_tophat(img, radius)
    difference = img - smoothed
    return difference > scoreatpercentile(difference[np.where(~np.isnan(difference))], threshold)


def circular_region(radius):

    xx, yy = np.mgrid[-radius:radius+1,-radius:radius+1]

    circle = xx**2. + yy**2.
    circle = circle < radius**2.

    circle = circle.astype(float)
    circle[np.where(circle==0.)] = np.NaN

    return circle, [xx, yy]

def padwithnans(vector,pad_width,iaxis,kwargs):
  vector[:pad_width[0]] = np.NaN
  vector[-pad_width[1]:] = np.NaN
  return vector






