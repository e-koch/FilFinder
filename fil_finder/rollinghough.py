# Licensed under an MIT open source license - see LICENSE

import numpy as np
from scipy.stats import scoreatpercentile
import matplotlib.pyplot as p


def rht(mask, radius, ntheta=180, background_percentile=25, verbose=False):
    '''

    Parameters
    ----------

    mask : numpy.ndarray
           Boolean or integer array. Transform performed at all
           non-zero points.

    radius : int
             Radius of circle used around each pixel.

    ntheta : int, optional
             Number of angles to use in transform.

    background_percentile : float, optional
                            Percentile of data to subtract off. Background is
                            due to limits on pixel resolution.

    verbose : bool, optional
        Enables plotting.

    Returns
    -------

    theta : numpy.ndarray
            Angles transform was performed at.

    R : numpy.ndarray
        Transform output.

    quantiles : numpy.ndarray.
        Contains the 25%, mean, and 75% percentile angles.

    '''

    pad_mask = np.pad(mask.astype(float), radius, padwithnans)

    # The theta=0 case isn't handled properly
    theta = np.linspace(np.pi/2., 1.5*np.pi, ntheta)

    # Create a cube of all angle positions
    circle, mesh = circular_region(radius)
    circles_cube = np.empty((ntheta, circle.shape[0], circle.shape[1]))
    for posn, ang in enumerate(theta):
        diff = mesh[0]*np.sin(ang) - mesh[1]*np.cos(ang)
        diff[np.where(np.abs(diff) < 1.0)] = 0
        circles_cube[posn, :, :] = diff

    R = np.zeros((ntheta,))
    x, y = np.where(mask != 0.0)
    for i, j in zip(x, y):
        region = np.tile(circle * pad_mask[i:i+2*radius+1,
                                           j:j+2*radius+1], (ntheta, 1, 1))
        line = region * np.isclose(circles_cube, 0.0)

        if not np.isnan(line).all():
            R = R + np.nansum(np.nansum(line, axis=2), axis=1)

    # Check that the ends are close.
    if np.isclose(R[0], R[-1], rtol=1.0):
        R = R[:-1]
        theta = theta[:-1]
    else:
        raise ValueError("R(-pi/2) should equal R(pi/2). Check input.")

    # You're likely to get a somewhat constant background, so subtract it out
    R = R - np.median(R[R <= scoreatpercentile(R, background_percentile)])
    if (R < 0.0).any():
        R[R < 0.0] = 0.0  # Ignore negative values after subtraction

    # Return to [-pi/2, pi/2] interval and position to the correct zero point.
    theta -= np.pi
    R = np.fliplr(R[:, np.newaxis])

    mean_circ = circ_mean(theta, weights=R)
    twofive, sevenfive = circ_CI(theta, weights=R, u_ci=0.67)
    twofive = twofive[0]
    sevenfive = sevenfive[0]
    quantiles = (twofive, mean_circ, sevenfive)

    if verbose:
        p.subplot(1, 2, 1, polar=True)
        p.plot(2*theta, R, "rD")
        p.plot([2*mean_circ]*2, [0, R.max()], "k")
        p.plot([2*twofive]*2, [0, R.max()], "r")
        p.plot([2*sevenfive]*2, [0, R.max()], "r")
        p.subplot(1, 2, 2)
        p.imshow(mask, cmap="binary", origin='lower')
        p.show()

    return theta, R, quantiles


def circular_region(radius):
    '''
    Create a circle of a given radius.
    Values are NaNs outside of the circle.

    Parameters
    ----------
    radius : int
        Circle radius.

    Returns
    -------
    circle : numpy.ndarray
        Array containing the circle.
    [xx, yy] : numpy.ndarray
        Grids used to create the circle.
    '''
    xx, yy = np.mgrid[-radius:radius+1, -radius:radius+1]

    circle = xx**2. + yy**2.
    circle = circle < radius**2.

    circle = circle.astype(float)
    circle[np.where(circle == 0.)] = np.NaN

    return circle, [xx, yy]


def padwithnans(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = np.NaN
    vector[-pad_width[1]:] = np.NaN
    return vector


def find_nearest(array, value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]


def find_nearest_posn(array, value):
    idx = (np.abs(array-value)).argmin()
    return idx


def circ_mean(theta, weights=None):
    """
    Calculates the median of a set of angles on the circle and returns
    a value on the interval from (-pi, pi].  Angles expected in radians.

    Parameters
    ----------
    """

    if len(theta.shape) == 1:
        theta = theta[:, np.newaxis]

    if weights is None:
        weights = np.ones(theta.shape)

    medangle = np.arctan2(np.nansum(np.sin(2*theta) * weights),
                          np.nansum(np.cos(2*theta) * weights))

    medangle /= 2.0

    return medangle


def fourier_shifter(x, shift, axis):
    '''
    Shift an array by some value along its axis.
    '''
    ftx = np.fft.fft(x, axis=axis)
    m = np.fft.fftfreq(x.shape[axis])
    # m_shape = [1] * x.ndim
    # m_shape[axis] = m.shape[0]
    # m = m.reshape(m_shape)
    slices = [slice(None) if ii == axis else None for ii in range(x.ndim)]
    m = m[tuple(slices)]
    phase = np.exp(-2 * np.pi * m * 1j * shift)
    x2 = np.real(np.fft.ifft(ftx * phase, axis=axis))
    return x2


def circ_CI(theta, weights=None, u_ci=0.67, axis=0):
    '''

    '''

    if len(theta.shape) == 1:
        theta = theta[:, np.newaxis]

    if weights is None:
        weights = np.ones(theta.shape)
    else:
        if len(weights.shape) == 1:
            weights = weights[:, np.newaxis]

    assert theta.shape == weights.shape

    # Normalize weights
    weights /= np.sum(weights, axis=axis)

    mean_ang = circ_mean(theta, weights=weights)

    # Now center the data around the mean to find the CI intervals
    diff_val = np.diff(theta[:2, 0])[0]

    theta_mid = theta[theta.shape[0] // 2]

    diff_posn = - (theta_mid - mean_ang) / diff_val

    theta_copy = fourier_shifter(theta, diff_posn, axis=0)

    vec_length2 = np.sum(weights * np.cos(theta_copy), axis=axis)**2. + \
        np.sum(weights * np.sin(theta_copy), axis=axis)**2.

    alpha = np.sum(weights * np.cos(2 * theta_copy), axis=axis)

    var_w = (1 - alpha) / (4 * vec_length2)

    # Make sure the CI stays within the interval. Otherwise assign it to
    # pi/2 (largest possible on interval of pi)
    sin_arg = u_ci * np.sqrt(2 * var_w)

    if sin_arg <= 1:
        ci = np.arcsin(sin_arg)
    else:
        ci = np.pi / 2.

    samp_cis = np.vstack([mean_ang - ci, mean_ang + ci])

    return samp_cis
