# Licensed under an MIT open source license - see LICENSE

from .utilities import find_nearest

import numpy as np
import scipy.ndimage as nd
import scipy.optimize as op
from scipy.integrate import quad
from scipy.stats import scoreatpercentile, percentileofscore
from scipy.interpolate import interp1d
from scipy.signal import argrelmax, argrelmin
from warnings import warn
from astropy.modeling import fitting, models
import astropy.modeling as mod


def dist_transform(labelisofil, filclean_all):
    '''

    Recombines the cleaned skeletons from final analysis and takes the
    Euclidean Distance Transform of each. Since each filament is in an
    array defined by its own size, the offsets need to be taken into account
    when adding back into the master array.

    Parameters
    ----------
    labelisofil : list
        Contains arrays of the cleaned individual skeletons
    filclean_all : numpy.ndarray
        Master array with all final skeletons.

    Returns
    -------
    dist_transform_all : numpy.ndarray
        A Euclidean Distance Transform of all of the skeletons combined.
    dist_transform_sep : list
        Contains the Euclidean Distance Transform of each
        individual skeleton.
    '''

    dist_transform_sep = []

    for skel_arr in labelisofil:
        if np.max(skel_arr) > 1:
            skel_arr[np.where(skel_arr > 1)] = 1
        dist_transform_sep.append(
            nd.distance_transform_edt(np.logical_not(skel_arr)))

    # Distance Transform of all cleaned filaments
    dist_transform_all = nd.distance_transform_edt(
        np.logical_not(filclean_all))

    return dist_transform_all, dist_transform_sep


def gaussian_model(dist, radprof, with_bkg=True):
    '''
    Return a Gaussian model with initial parameter guesses. The model is for
    radial profiles and the peak is assumed to be fixed at the centre.

    Parameters
    ----------
    dist : `~numpy.ndarray`
        Distance bins of the radial profile.
    radprof : `~numpy.ndarray`
        The binned radial profile.
    with_bkg : bool, optional
        Add constant background to the fit when enabled.
    '''

    amp = radprof.max()
    wid_amp = amp / np.exp(0.5)
    idx = np.abs(radprof - wid_amp).argmin()
    inner_width = dist[idx]

    mod = models.Gaussian1D()
    # Check if the version of astropy supports units in the model.
    if mod._supports_unit_fitting:
        mod.amplitude = amp
        mod.mean = 0.0 * inner_width.unit
        mod.stddev = inner_width
    else:
        mod.amplitude = amp.value
        mod.mean = 0.0
        mod.stddev = inner_width.value

    # Fix the mean to 0, since this is for radial profiles.
    mod.mean.fixed = True

    if with_bkg:
        bkg_mod = models.Const1D()
        if bkg_mod._supports_unit_fitting:
            bkg_mod.amplitude = np.min(radprof)
        else:
            bkg_mod.amplitude = np.min(radprof).value
        bkg_mod = bkg_mod.rename("Bkg")
        mod = mod + bkg_mod

        # Check if the compound model can handle units?
        if not mod._supports_unit_fitting:
            # Strip the units out. Hopefully I can get rid of this in a
            # future release...
            mod_new = models.Gaussian1D() + models.Const1D()

            mod_new.amplitude_0 = mod.amplitude_0.value
            mod_new.mean_0 = mod.mean_0.value
            mod_new.mean_0.fixed = True
            mod_new.stddev_0 = mod.stddev_0.value
            mod_new.amplitude_1 = mod.amplitude_1.value

            mod = mod_new

    return mod


def fit_radial_model(dist, radprof, model, fitter=None, weights=None,
                     **fitter_kwargs):
    '''
    Fit a model to the radial profile.

    Parameters
    ----------
    dist :

    radprof :

    model : `~astropy.modeling.models.Fittable1DModel`
        An astropy model object to fit to.
    fitter : `~astropy.modeling.fitting.Fitter`, optional
        One of the fitting classes from astropy. This should probably be a
        non-linear fitting algorithm, but it depends on the chosen model.
        Defaults to a Levenberg-Marquardt fitter.
    weights : `~numpy.ndarray`, optional
        Weights to apply in the fitting.
    '''

    if not isinstance(model, mod.Model):
        raise TypeError("The model must be a 1D astropy model.")

    if fitter is None:
        fitter = fitting.LevMarLSQFitter()
    else:
        if not isinstance(fitter, fitting.Fitter):
            raise TypeError("The fitter must be one of the "
                            "astropy.modeling.fitting classes.")

    if model._supports_unit_fitting:
        xdata = dist
        ydata = radprof
    else:
        xdata = dist.value
        ydata = radprof.value

    fitted_mod = fitter(model, xdata, ydata, weights=weights, **fitter_kwargs)

    return fitted_mod, fitter


def gauss_model(distance, rad_profile, weights, img_beam):
    '''
    Fits a Gaussian to the radial profile of each filament.

    Parameters
    ----------
    distance : list
        Distances from the skeleton.
    rad_profile : list
        Intensity values from the image.
    weights : list
        Weights to be used for the fit.
    img_beam : float
        FWHM of the beam size.

    Returns
    -------
    fit : numpy.ndarray
        Fit values.
    fit_errors : numpy.ndarray
        Fit errors.
    gaussian : function
        Function used to fit the profile.
    parameters : list
        Names of the parameters.
    fail_flag : bool
        Identifies a failed fit.
    '''

    p0 = (np.max(rad_profile), np.std(distance), np.min(rad_profile))
    parameters = ["Amplitude", "Width", "Background", "FWHM"]

    def gaussian(x, *p):
        '''
        Parameters
        ----------
        x : list or numpy.ndarray
            1D array of values where the model is evaluated
        p : tuple
            Components are:
            * p[0] Amplitude
            * p[1] Width
            * p[2] Background
        '''
        return (p[0] - p[2]) * np.exp(-1 * np.power(x, 2) /
                                      (2 * np.power(p[1], 2))) + p[2]

    try:
        fit, cov = op.curve_fit(gaussian, distance, rad_profile, p0=p0,
                                maxfev=100 * (len(distance) + 1), sigma=weights)
        fit_errors = np.sqrt(np.diag(cov))
    except:
        print("curve_fit failed.")
        fit, fit_errors = p0, None
        return fit, fit_errors, gaussian, parameters, True

    # Because of how function is defined,
    # fit function can get stuck at negative widths
    # This doesn't change the output though.
    fit[1] = np.abs(fit[1])

    # Deconvolve the width with the beam size.
    factor = 2 * np.sqrt(2 * np.log(2))  # FWHM factor
    deconv = (factor * fit[1]) ** 2. - img_beam ** 2.
    if deconv > 0:
        fit_errors = np.append(
            fit_errors, (factor**2 * fit[1] * fit_errors[1]) / np.sqrt(deconv))
        fit = np.append(fit, np.sqrt(deconv))
    else:  # Set to zero, can't be deconvolved
        fit = np.append(fit, 0.0)
        fit_errors = np.append(fit_errors, 0.0)

    fail_flag = False
    # The bkg level may be quite uncertain. Only look for poor amplitude or
    # width constraints.
    fail_conditions = fit_errors is None or \
        fit[0] < fit[2] or (fit_errors[:2] > np.abs(fit[:2])).any()
    if fail_conditions:
        fail_flag = True

    return fit, fit_errors, gaussian, parameters, fail_flag


def nonparam_width(distance, rad_profile, unbin_dist, unbin_prof,
                   img_beam=None, bkg_percent=5, peak_percent=99):
    '''
    Estimate the width and peak brightness of a filament non-parametrically.
    The intensity at the peak and background is estimated. The profile is then
    interpolated over in order to find the distance corresponding to these
    intensities. The width is then estimated by finding the distance where
    the intensity drops to 1/e.

    Parameters
    ----------
    distance : list
        Binned distances from the skeleton.
    rad_profile : list
        Binned intensity values from the image.
    unbin_dist : list
        Unbinned distances.
    unbin_prof : list
        Unbinned intensity values.
    img_beam : float, optional
        FWHM of the beam size.
    bkg_percent : float, optional
        Percentile of the data to estimate the background.
    peak_percent : float, optional
        Percentile of the data to estimate the peak of the profile.

    Returns
    -------
    params : numpy.ndarray
        Estimated parameter values.
    param_errors : numpy.ndarray
        Estimated errors.
    fail_flag : bool
        Indicates whether the fit failed.
    '''

    fail_flag = False

    # Find the intensities at the given percentiles
    bkg_intens = scoreatpercentile(rad_profile, bkg_percent)
    peak_intens = scoreatpercentile(rad_profile, peak_percent)

    # Interpolate over the bins in distance
    interp_bins = np.linspace(0.0, np.max(distance), 10 * len(distance))
    interp_profile = np.interp(interp_bins, distance, rad_profile)

    # Find the width by looking for where the intensity drops to 1/e from the
    # peak
    target_intensity = ((peak_intens - bkg_intens) / np.exp(0.5)) + bkg_intens
    width = interp_bins[np.where(interp_profile ==
                                 find_nearest(interp_profile, target_intensity))][0]

    # Estimate the width error by looking +/-5 percentile around the target
    # intensity
    target_percentile = percentileofscore(rad_profile, target_intensity)
    upper = scoreatpercentile(
        rad_profile, np.min((100, target_percentile + 5)))
    lower = scoreatpercentile(rad_profile, np.max((0, target_percentile - 5)))

    error_range_pts = np.logical_and(unbin_prof > lower, unbin_prof < upper)
    width_error = np.max(unbin_dist[error_range_pts]) -\
        np.min(unbin_dist[error_range_pts])

    # Deconvolve the width with the beam size.
    factor = 2 * np.sqrt(2 * np.log(2))  # FWHM factor
    if img_beam is not None:

        deconv = (width * factor) ** 2. - img_beam ** 2.
        if deconv > 0:
            fwhm_width = np.sqrt(deconv)
            fwhm_error = (factor**2 * width * width_error) / fwhm_width
        else:  # Set to zero, can't be deconvolved
            # If you can't devolve it, set it to minimum, which is the
            # beam-size.
            fwhm_width = 0.0
            fwhm_error = 0.0
    else:
        fwhm_width = width * factor
        fwhm_error = width_error * factor

    # Check where the "background" and "peak" are. If the peak distance is
    # greater, we are simply looking at a bad radial profile.
    bkg_dist = np.median(
        interp_bins[np.where(interp_profile ==
                             find_nearest(interp_profile, bkg_intens))])
    peak_dist = np.median(
        interp_bins[np.where(interp_profile ==
                             find_nearest(interp_profile, peak_intens))])
    bkg_error = np.std(
        unbin_prof[unbin_dist >= find_nearest(unbin_dist, bkg_dist)])
    peak_error = np.std(
        unbin_prof[unbin_dist <= find_nearest(unbin_dist, peak_dist)])

    # Check if there are unrealistic estimates
    if peak_dist > bkg_dist or fwhm_error > fwhm_width:
        fail_flag = True

    params = np.array([peak_intens, width, bkg_intens, fwhm_width])
    param_errors = np.array([peak_error, width_error, bkg_error, fwhm_error])

    return params, param_errors, fail_flag


def radial_profile(img, dist_transform_all, dist_transform_sep, offsets,
                   img_scale=1.0, bins=None, bintype="linear",
                   weighting="number", return_unbinned=True, auto_cut=False,
                   pad_to_distance=0.15, max_distance=0.3, auto_cut_kwargs={},
                   debug_mode=False):
    '''
    Fits the radial profiles to all filaments in the image.

    Parameters
    ----------
    img : numpy.ndarray
              The original image.
    dist_transform_all : numpy.ndarray
        The distance transform of all the skeletons.
    dist_transform_sep : list
        The distance transforms of each individual skeleton.
    offsets : list
        Contains the indices where each skeleton was cut out of the original
        array.
    img_scale : float
        Pixel to physical scale conversion.
    bins : numpy.ndarray, optional
        Bins to use for the profile fitting.
    bintype : str, optional
        "linear" for linearly spaced bins; "log" for log-spaced bins.
        Default is "linear".
    weighting : str, optional
        "number" is by the number of points in each bin; "var" is the
        variance of the values in each bin. Default is "number".
    return_unbinned : bool
        If True, returns the unbinned data as well as the binned.
    auto_cut : bool, optional
        Enables the auto cutting routines.
    pad_to_distance : float, optional
        Include pixels in the profile whose distance in dist_transform_sep
        is less than dist_transform_all + pad_to_distance. This is useful
        for creating profiles in crowded regions. If set to 0.0, no padding
        is done. Must be less than max_distance.
    max_distance : float, optional
        Cuts the profile at the specified physical distance (in pc).
    debug_mode : bool, optional
        Enables plotting of which pixels are being used in the radial profile.

    Returns
    -------
    bin_centers : numpy.ndarray
        Center of the bins used in physical units.
    radial_prof : numpy.ndarray
        Binned intensity profile.
    weights : numpy.ndarray
        Weights evaluated for each bin.
    '''

    if max_distance <= 0.0:
        raise ValueError("max_distance must be positive.")

    if pad_to_distance < 0.0 or pad_to_distance > max_distance:
        raise ValueError("pad_to_distance must be positive and less than "
                         "max_distance.")

    width_value = []
    width_distance = []
    x, y = np.where(np.isfinite(dist_transform_sep) *
                    (dist_transform_sep <= max_distance / img_scale))
    # Transform into coordinates of master image
    # Originally had a +1 offset, but new code doesn't need this
    # Correction added directly into fil_finder_2D
    x_full = x + offsets[0][0]  # - 1
    y_full = y + offsets[0][1]  # - 1

    pad_pixel_distance = int(pad_to_distance * img_scale ** -1)

    # Don't necessarily need dist_transform_all. If None, skip some parts
    if dist_transform_all is None:
        check_global = False
    else:
        check_global = True

    # Check if the image has a unit
    if hasattr(img, 'unit'):
        img_vals = img.value
    else:
        img_vals = img

    valids = np.zeros_like(dist_transform_sep, dtype=bool)

    for i in range(len(x)):
        # Check overall distance transform to make sure pixel belongs to proper
        # filament
        img_val = img_vals[x_full[i], y_full[i]]
        sep_dist = dist_transform_sep[x[i], y[i]]
        if check_global:
            glob_dist = dist_transform_all[x_full[i], y_full[i]]
        if np.isfinite(img_val):
            if check_global:
                # Include the point if it falls within the pad distance.
                if sep_dist <= glob_dist + pad_pixel_distance:
                    width_value.append(img_val)
                    width_distance.append(sep_dist)
                    # valids[x_full[i], y_full[i]] = True
                    valids[x[i], y[i]] = True
            else:
                width_value.append(img_val)
                width_distance.append(sep_dist)
                # valids[x_full[i], y_full[i]] = True
                valids[x[i], y[i]] = True

    if debug_mode:
        import matplotlib.pyplot as plt
        plt.subplot(121)
        plt.imshow(dist_transform_sep)
        plt.contour(valids, colors='g')
        plt.contour(dist_transform_all == 0., colors='m')
        plt.contour(dist_transform_sep == 0., colors='c')
        plt.subplot(122)
        plt.imshow(dist_transform_all)
        try:
            plt.contour(dist_transform_sep <= max_distance / img_scale,
                        colors='b')
        except ValueError:
            print("No contour")
        plt.draw()
        raw_input("?")
        plt.clf()

    if len(width_distance) == 0:
        warn("No valid pixels for radial profile found.")
        return None

    width_value = np.asarray(width_value)
    width_distance = np.asarray(width_distance)

    # Binning
    if bins is None:
        nbins = int(np.sqrt(len(width_value)))
        maxbin = np.max(width_distance)
        if bintype is "log":
            # bins must start at 1 if logspaced
            bins = np.logspace(0, np.log10(maxbin), nbins + 1)
        elif bintype is "linear":
            bins = np.linspace(0, maxbin, nbins + 1)

    whichbins = np.digitize(width_distance, bins)
    bin_centers = (bins[1:] + bins[:-1]) / 2.0

    radial_prof = np.zeros_like(bin_centers)
    weights = np.zeros_like(bin_centers)

    for nbin in range(1, int(nbins) + 1):

        bin_posns = whichbins == nbin

        # Skip any empty bins
        if bin_posns.sum() == 0:
            continue

        radial_prof[nbin - 1] = np.median(width_value[bin_posns])

        if weighting == "number":
            weights[nbin - 1] = whichbins[bin_posns].sum()
        elif weighting == "var":
            weights[nbin - 1] = np.nanvar(width_value[bin_posns])

    # Remove all empty bins
    radial_prof = radial_prof[weights > 0]
    bin_centers = bin_centers[weights > 0]
    weights = weights[weights > 0]

    # Put bins in the physical scale.
    bin_centers *= img_scale
    width_distance *= img_scale

    if auto_cut:
        bin_centers, radial_prof, weights = \
            _smooth_and_cut(bin_centers, radial_prof, weights,
                            **auto_cut_kwargs)

    if return_unbinned:
        width_distance = width_distance[np.isfinite(width_value)]
        width_value = width_value[np.isfinite(width_value)]
        return bin_centers, radial_prof, weights, width_distance, width_value
    else:
        return bin_centers, radial_prof, weights


def _smooth_and_cut(bins, values, weights, interp_factor=10,
                    pad_cut=5, smooth_size=0.03, min_width=0.1):
    '''
    Smooth the radial profile and cut if it increases at increasing
    distance. Also checks for profiles with a plateau between two decreasing
    profiles and cut out the last one (as it is not the local profile).

    Parameters
    ----------
    bins : numpy.ndarray
        Bins for the profile.
    values : numpy.ndarray
        Values in each bin.
    weights : numpy.ndarray
        Weights for each bin. These are only clipped to the same position as
        the rest of the profile. Otherwise, no alteration is made.
    interp_factor : int, optional
        The factor to increase the number of bins by for interpolation.
    pad_cut : int, optional
        Add additional bins after the cut is found. The smoothing often cuts
        out some bins which follow the desired profile.
    smooth_size : float, optional
        Set the smoothing size when finding local extrema. The value should
        have the same units as given in `bins`.
    min_width : float, optional
        Ignore local minima below this minimum width.

    Returns
    -------
    cut_bins : numpy.ndarray
        Bins for the profile with a possible cutoff.
    cut_values : numpy.ndarray
        Values in each bin with a possible cutoff.
    cut_weights : numpy.ndarray
        Weights for each bin with a possible cutoff.

    '''

    # Interpolate the points onto a finer spacing
    smooth_bins = np.linspace(bins.min(), bins.max(),
                              interp_factor * bins.size)
    smooth_bin_width = smooth_bins[1] - smooth_bins[0]

    smooth_val = interp1d(bins, values, kind='cubic')(smooth_bins)

    # Adjust size based on interpolation upsample factor
    window_size = int(np.floor(smooth_size / smooth_bin_width))
    # Must be odd!
    if window_size % 2 == 0:
        window_size -= 1

    # Perform a moving average on the interpolated points.
    pad_add = int((window_size - 1) // 2)
    smooth_val = \
        np.convolve(smooth_val, np.ones((window_size,)) / window_size,
                    mode='valid')

    grad = np.gradient(smooth_val, smooth_bins[1] - smooth_bins[0])

    # The shapes need equal so the index to cut at is correct.
    assert smooth_bins.size == smooth_val.size + 2 * pad_add

    cut = crossings_nonzero_all(grad) + pad_add

    # Check for evidence of second drop-off
    new_cut = None

    # Look for local max and mins (must hold True for range of ~0.05 pc)
    bin_diff = smooth_bins[1] - smooth_bins[0]
    loc_mins = argrelmin(grad, order=int(smooth_size / bin_diff))[0] + pad_add
    loc_maxs = argrelmax(grad, order=int(smooth_size / bin_diff))[0] + pad_add

    # Discard below some minimum width (defaults to 0.1 pc).
    loc_mins = loc_mins[smooth_bins[loc_mins] > min_width]
    loc_maxs = loc_maxs

    if loc_mins.size > 0 and loc_maxs.size > 0:
        i = 0
        while True:
            loc_min = loc_mins[i]

            difference = loc_min - loc_maxs
            if (difference > 0).any():
                new_cut = loc_maxs[np.argmin(difference[difference > 0])]
                if smooth_bins[new_cut] > min_width:
                    break

            i += 1

            if i == loc_mins.size:
                break

    if new_cut == 0:
        new_cut = None

    # No cut is found, so no values are sliced off.
    if cut.size == 0 and new_cut is None:
        cut_posn = bins.size
    # Only a plateau cut was found.
    elif cut.size == 0 and new_cut is not None:
        cut_posn = _nearest_idx(bins, smooth_bins[new_cut])
    else:
        # Check for a cut given by the plateau check.
        # If there is isn't one or it is at a greater radius, use the first
        # zero gradient crossing.
        if new_cut is None or new_cut >= cut[0]:
            cut_used = cut[0]
        else:
            cut_used = new_cut

        cut_posn = _nearest_idx(bins, smooth_bins[cut_used])

    # Now adjust by the given pad_cut size
    # Always use +1 to cut to the extrema point.
    cut_posn += 1 + pad_cut

    cut_bins = bins[:cut_posn]
    cut_vals = values[:cut_posn]
    cut_weights = weights[:cut_posn]

    return cut_bins, cut_vals, cut_weights


def _nearest_idx(array, value):
    return (np.abs(array - value)).argmin()


def crossings_nonzero_all(data):
    assert isinstance(data, np.ndarray)
    pos = data > 0
    npos = ~pos
    return ((pos[:-1] & npos[1:]) | (npos[:-1] & pos[1:])).nonzero()[0]
