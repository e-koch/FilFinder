'''

Rolling Hough Transform Implementation
See  Clark et al. 2013 for description

'''

import numpy as np
import scipy.ndimage as nd
from scipy.stats import scoreatpercentile
import matplotlib.pyplot as p
from operator import itemgetter
from itertools import groupby


def rht(mask, radius, ntheta=180, background_percentile=25):
    '''

    Parameters
    **********

    mask : numpy.ndarray
           Boolean or integer array. Transform performed at all non-zero points.

    radius : int
             Radius of circle used around each pixel.

    ntheta : int, optional
             Number of angles to use in transform.

    background_percentile : float, optional
                            Percentile of data to subtract off. Background is
                            due to limits on pixel resolution.

    Returns
    *******

    theta : numpy.ndarray
            Angles transform was performed at.

    R : numpy.ndarray
        Transform output.

    '''

    pad_mask = np.pad(mask.astype(float), radius, padwithnans)

    # The theta=0 case isn't handled properly
    theta = np.linspace(np.pi/2.,1.5*np.pi, ntheta)

    ## Create a cube of all angle positions
    circle, mesh = circular_region(radius)
    circles_cube = np.empty((ntheta, circle.shape[0], circle.shape[1]))
    for posn, ang in enumerate(theta):
        diff = mesh[0]*np.sin(ang) - mesh[1]*np.cos(ang)
        diff[np.where(np.abs(diff)<1.0)] = 0
        circles_cube[posn,:,:] = diff

    R = np.zeros((ntheta,))
    x, y = np.where(mask!=0.0)
    for i,j in zip(x,y):
        region = np.tile(circle * pad_mask[i:i+2*radius+1,j:j+2*radius+1], (ntheta, 1, 1))
        line = region * np.isclose(circles_cube, 0.0)

        if np.isnan(line).all():
            pass ## If all nans, ignore
        else:
            R = R + np.nansum(np.nansum(line, axis=2), axis=1)

    # Check that the ends are close.
    if np.isclose(R[0], R[-1], rtol=1.0):
        R = R[:-1]
        theta = theta[:-1]
    else:
        raise ValueError("R(pi/2) should equal R(3/2*pi). Check input.")

    ## You're likely to get a somewhat constant background, so subtract that out
    R = R - np.median(R[R<=scoreatpercentile(R,background_percentile)])
    if (R<0.0).any():
        R[R<0.0] = 0.0 ## Ignore negative values after subtraction

    # Now we want to set the position to "start" the distribution at
    # We look for minima (or near minima) in the distribution, then see if any are sequentially in line
    # The position used is the median of the longest sequence
    zero_posn = np.where(R<=scoreatpercentile(R, 5))[0]
    mins = np.where(R==R.min())[0]
    if mins.shape[0]>zero_posn.shape[0]: # If there are more minima, use that for a better estimate.
        zero_posn = mins

    if zero_posn.shape[0]>1:
        continuous_sections = []
        for _, g in groupby(enumerate(zero_posn), lambda (i,x): i-x):
            continuous_sections.append(map(itemgetter(1), g))
        try:
            section = max(continuous_sections, key=len)
            zero_posn = int(np.median(section))
        except ValueError:
            # If there are no groups, just choose the first point.
            zero_posn = zero_posn[0]

    # Return to [0, pi] interval and position to the correct zero point.
    theta -= np.pi/2.
    theta = np.roll(theta, zero_posn)
    R = np.roll(R, zero_posn+(ntheta/2))

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






