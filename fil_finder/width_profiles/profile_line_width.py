
import scipy.ndimage as nd
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as p
import astropy.units as u
from astropy.table import QTable

from .profile import profile_line

eight_conn = np.ones((3, 3))

end_structs = [np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 0]]),
               np.array([[0, 1, 0],
                         [0, 1, 0],
                         [0, 0, 0]]),
               np.array([[0, 0, 1],
                         [0, 1, 0],
                         [0, 0, 0]]),
               np.array([[0, 0, 0],
                         [1, 1, 0],
                         [0, 0, 0]]),
               np.array([[0, 0, 0],
                         [0, 1, 1],
                         [0, 0, 0]]),
               np.array([[0, 0, 0],
                         [0, 1, 0],
                         [1, 0, 0]]),
               np.array([[0, 0, 0],
                         [0, 1, 0],
                         [0, 1, 0]]),
               np.array([[0, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]])]

four_conn_posns = [1, 3, 5, 7]
eight_conn_posns = [0, 2, 6, 8]


def filament_profile(skeleton, image, pixscale, max_dist=0.025 * u.pc,
                     distance=250. * u.pc, num_avg=3, verbose=False,
                     bright_unit="Jy km/s", noise=None, fit_profiles=True):
    '''
    Calculate radial profiles along the main extent of a skeleton (ie. the
    longest path). The skeleton must contain a single branch with no
    intersections.

    Parameters
    ----------
    skeleton : np.ndarray
        Boolean array containing the skeleton
    image : np.ndarray
        Image to compute the profiles from. Must match the spatial extent
        of the skeleton array.
    pixscale : `~astropy.units.Quantity`
        Angular size of a pixel in the image. Must have units equivalent to
        degrees.
    max_dist : astropy Quantity, optional
        The angular or physical (when distance is given) extent to create the
        profile away from the centre skeleton pixel. The entire profile will
        be twice this value (for each side of the profile).
    distance : astropy Quantity, optional
        Physical distance to the region in the image. If None is given,
        results will be in angular units based on the header.
    num_avg : int, optional
        Number of points before and after a pixel that is used when computing
        the normal vector. Using at least three points is recommended due to
        small pixel instabilities in the skeletons.
    verbose : bool, optional
        Enable plotting of the profile and the accompanying for each pixel in
        the skeleton.
    bright_unit : string or astropy Unit
        Brightness unit of the image.
    noise : np.ndarray, optional
        RMS array for the accompanying image. When provided, the errors
        are calculated along each of the profiles and used as weights in the
        fitting.
    fit_profiles : bool, optional
        When enabled, fits a Gaussian model to the profiles. Otherwise only
        the profiles are returned.

    Returns
    -------
    line_distances : list
        Distances along the profiles.
    line_profiles : list
        Radial profiles.
    profile_extents : list
        Contains the pixel position of the start of the profile,
        the skeleton pixel, and the end of the profile.
    tab : astropy QTable
        Table of the fit results and errors with appropriate units.
    '''

    deg_per_pix = pixscale.to(u.deg) / u.pixel

    if distance is not None:
        phys_per_pix = distance * (np.pi / 180.) * deg_per_pix / u.deg

        max_pixel = (max_dist / phys_per_pix).value

    else:
        # max_dist should then be in pixel or angular units
        if not isinstance(max_dist, u.Quantity):
            # Assume pixels
            max_pixel = max_dist
        else:
            try:
                max_pixel = max_dist.to(u.pix).value
            except u.UnitConversionError:
                # In angular units
                equiv = [(u.pixel, u.deg, lambda x: x / (pixscale * u.pix),
                          lambda x: x * pixscale * u.pix)]
                max_pixel = max_dist.to(u.pix, equivalencies=equiv).value

    if bright_unit is None:
        bright_unit = u.dimensionless_unscaled
    elif isinstance(bright_unit, str):
        bright_unit = u.Unit(bright_unit)
    elif isinstance(bright_unit, u.UnitBase):
        pass
    else:
        raise TypeError("bright_unit must be compatible with astropy.units.")

    # Make sure the noise array is the same shape
    if noise is not None:
        assert noise.shape == image.shape

    # Get the points in the skeleton (in order)
    skel_pts = walk_through_skeleton(skeleton)

    line_profiles = []
    line_distances = []
    profile_extents = []
    profile_fits = []
    red_chisqs = []

    for j, i in enumerate(range(num_avg, len(skel_pts) - num_avg)):
        # Calculate the normal direction from the surrounding pixels
        pt1 = avg_pts([skel_pts[i + j] for j in range(-num_avg, 0)])
        pt2 = avg_pts([skel_pts[i + j] for j in range(1, num_avg + 1)])

        vec = np.array([float(x2 - x1) for x2, x1 in
                        zip(pt1, pt2)])
        vec /= np.linalg.norm(vec)

        per_vec = perpendicular(vec)

        line_pts = find_path_ends(skel_pts[i], max_pixel, per_vec)

        left_profile, left_dists = \
            profile_line(image, skel_pts[i], line_pts[0])
        right_profile, right_dists = \
            profile_line(image, skel_pts[i], line_pts[1])

        total_profile = np.append(left_profile[::-1], right_profile) * \
            bright_unit

        if noise is not None:
            left_profile, _ = \
                profile_line(noise, skel_pts[i], line_pts[0])
            right_profile, _ = \
                profile_line(noise, skel_pts[i], line_pts[1])
            noise_profile = np.append(left_profile[::-1], right_profile) * \
                bright_unit
        else:
            noise_profile = None

        if distance is not None:
            total_dists = np.append(-left_dists[::-1], right_dists) \
                * u.pix * phys_per_pix
        else:
            total_dists = np.append(-left_dists[::-1], right_dists) \
                * u.pix * deg_per_pix

        if noise is not None:
            if len(total_profile) != len(noise_profile):
                raise ValueError("Intensity and noise profile lengths do not"
                                 " match. Have you applied the same mask to"
                                 " both?")

        line_profiles.append(total_profile)
        line_distances.append(total_dists)
        profile_extents.append([line_pts[0], skel_pts[i], line_pts[1]])

        if fit_profiles:
            # Now fit!
            profile_fit, profile_fit_err, red_chisq = \
                gauss_fit(total_dists.value, total_profile.value,
                          sigma=noise_profile)

            profile_fits.append(np.hstack([profile_fit, profile_fit_err]))
            red_chisqs.append(red_chisq)

        if verbose:
            p.subplot(121)
            p.imshow(image, origin='lower')
            p.contour(skeleton, colors='r')
            p.plot(skel_pts[i][1], skel_pts[i][0], 'bD')
            p.plot(line_pts[0][1], line_pts[0][0], 'bD')
            p.plot(line_pts[1][1], line_pts[1][0], 'bD')

            p.subplot(122)
            p.plot(total_dists, total_profile, 'bD')
            pts = np.linspace(total_dists.min().value,
                              total_dists.max().value, 100)
            if fit_profiles:
                p.plot(pts, gaussian(pts, *profile_fit), 'r')

            if distance is not None:
                unit = (u.pix * phys_per_pix).unit.to_string()
            else:
                unit = (u.pix * deg_per_pix).unit.to_string()
            p.xlabel("Distance from skeleton (" + unit + ")")
            p.ylabel("Surface Brightness (" + bright_unit.to_string() + ")")
            p.tight_layout()
            p.show()

    if fit_profiles:
        profile_fits = np.asarray(profile_fits)
        red_chisqs = np.asarray(red_chisqs)

        # Create an astropy table of the fit results
        param_names = ["Amplitude", "Std Dev", "Background"]
        param_errs = [par + " Error" for par in param_names]
        colnames = param_names + param_errs
        in_bright_units = [True, False, True] * 2
        tab = QTable()

        tab["Number"] = np.arange(profile_fits.shape[0])
        tab.add_index("Number")

        tab["Red Chisq"] = red_chisqs

        for i, (name, is_bright) in enumerate(zip(colnames, in_bright_units)):
            if is_bright:
                col_unit = bright_unit
            else:
                if distance is not None:
                    col_unit = (u.pix * phys_per_pix).unit
                else:
                    col_unit = (u.pix * deg_per_pix).unit

            tab[name] = profile_fits[:, i] * col_unit

        return line_distances, line_profiles, profile_extents, tab
    else:
        return line_distances, line_profiles


def perpendicular(a):
    '''
    Return the perpendicular vector to a given 2D vector.
    '''
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b


def walk_through_skeleton(skeleton):
    '''
    Starting from one end, walk through a skeleton in order. Intended for use
    with skeletons that contain no branches.
    '''

    # Calculate the end points
    end_pts = return_ends(skeleton)
    if len(end_pts) != 2:
        raise ValueError("Skeleton must contain no intersections.")

    # Force the first end point to be closest to the image origin.
    if two_point_dist(end_pts[1], [0, 0]) < two_point_dist(end_pts[0], [0, 0]):
        end_pts = end_pts[::-1]

    all_pts = int(np.sum(skeleton))

    yy, xx = np.mgrid[-1:2, -1:2]
    yy = yy.ravel()
    xx = xx.ravel()

    for i in range(all_pts):
        if i == 0:
            ordered_pts = [end_pts[0]]
            prev_pt = end_pts[0]
        else:
            # Check for neighbors
            y, x = prev_pt
            # Extract the connected region
            neighbors = skeleton[y - 1:y + 2, x - 1:x + 2].ravel()
            # Define the corresponding array indices.
            yy_inds = yy + y
            xx_inds = xx + x

            hits = [int(elem) for elem in np.argwhere(neighbors)]
            # Remove the centre point and any points already in the list
            for pos, (y_ind, x_ind) in enumerate(zip(yy_inds, xx_inds)):
                if (y_ind, x_ind) in ordered_pts:
                    hits.remove(pos)

            num_hits = len(hits)

            if num_hits == 0:
                # You've reached the end. It better be the other end point
                if prev_pt[0] != end_pts[1][0] or prev_pt[1] != end_pts[1][1]:
                    raise ValueError("Final point does not match expected"
                                     " end point. Check input skeleton for"
                                     " intersections.")
                break
            elif num_hits == 1:
                # You have found the next point
                posn = hits[0]
                next_pt = (y + yy[posn], x + xx[posn])
                ordered_pts.append(next_pt)
            else:
                # There's at least a couple neighbours (for some reason)
                # Pick the 4-connected component since it is the closest
                for fours in four_conn_posns:
                    if fours in hits:
                        posn = hits[hits.index(fours)]
                        break
                else:
                    raise ValueError("Disconnected eight-connected pixels?")
                next_pt = (y + yy[posn], x + xx[posn])
                ordered_pts.append(next_pt)
            prev_pt = next_pt

    return ordered_pts


def return_ends(skeleton):
    '''
    Find the endpoints of the skeleton.
    '''

    end_points = []

    for i, struct in enumerate(end_structs):
        hits = nd.binary_hit_or_miss(skeleton, structure1=struct)

        if not np.any(hits):
            continue

        for y, x in zip(*np.where(hits)):
            end_points.append((y, x))

    return end_points


def find_path_ends(posn, max_dist, vector):
    '''
    Find ends of a path for line_profile given
    a vector direction.
    '''

    vector = vector.astype(float) / np.linalg.norm(vector)

    max_size = np.ceil(max_dist).astype(int)

    yy, xx = np.mgrid[-max_size:max_size + 1, -max_size:max_size + 1]

    max_circle = yy**2 + xx**2 <= max_dist**2
    ring = \
        np.logical_xor(max_circle,
                       nd.binary_erosion(max_circle, eight_conn))

    radius_pts = [(y + posn[0], x + posn[1]) for y, x in zip(*np.where(ring))]

    x_step = vector[1]
    y_step = vector[0]

    y_diff = max_size * y_step
    x_diff = max_size * x_step

    neg_line_posn = (posn[0] - y_diff, posn[1] - x_diff)
    pos_line_posn = (posn[0] + y_diff, posn[1] + x_diff)

    # pos_dists = np.array([two_point_dist(pos_line_posn, pt)
    #                       for pt in radius_pts])
    # neg_dists = np.array([two_point_dist(neg_line_posn, pt)
    #                       for pt in radius_pts])

    # These should be the ones used to ensure
    # proper distance from the skeleton point.
    # pos_posn give weird results though...

    # pos_posn = radius_pts[np.argmin(pos_dists)]
    # neg_posn = radius_pts[np.argmin(neg_dists)]

    return [neg_line_posn, pos_line_posn]


def two_point_dist(pt1, pt2):
    return np.linalg.norm([x2 - x1 for x2, x1 in zip(pt1, pt2)])


def avg_pts(pts):
    dims = len(pts[0])
    avg_pt = []
    for dim in range(dims):
        avg_pt.append(np.mean([pt[dim] for pt in pts]).astype(int))
    return avg_pt


def gaussian(x, *p):
    '''
    Parameters
    ----------
    x : list or numpy.ndarray
        1D array of values where the model is evaluated
    p : tuple
        Components are:
        * p[0] Amplitude
        * p[1] Mean
        * p[2] Width
        * p[3] Background
    '''
    return (p[0] - p[2]) * np.exp(-1 * np.power(x, 2) /
                                  (2 * np.power(p[1], 2))) + p[2]


def gauss_fit(distance, rad_profile, sigma=None):
    '''
    Fits a Gaussian to the radial profile.

    Parameters
    ----------
    distance : np.ndarray
        Distances from the skeleton.
    rad_profile : np.ndarray
        Intensity values from the image.
    sigma : np.ndarray
        Errors to be used for the fit. Assumes these are 1-sigma errors.

    Returns
    -------
    fit : numpy.ndarray
        Fit values.
    fit_errors : numpy.ndarray
        Fit errors.
    red_chisq : float
        The reduced chi-squared value for the fit.
    '''

    p0 = (np.max(rad_profile), np.std(distance), np.min(rad_profile))

    try:
        fit, cov, info, _, _ = \
            curve_fit(gaussian, distance, rad_profile, p0=p0,
                      maxfev=100 * (len(distance) + 1), sigma=sigma,
                      absolute_sigma=True, full_output=True)
        fit_errors = np.sqrt(np.diag(cov))
        red_chisq = (info['fvec']**2).sum() / (len(info['fvec']) - len(fit))

    except Exception as e:
        print("curve_fit failed with " + str(e))
        # p.plot(distance, rad_profile)
        # p.show()

        fit = np.asarray([np.NaN] * len(p0))
        fit_errors = np.asarray([np.NaN] * len(p0))
        red_chisq = np.NaN

    return fit, fit_errors, red_chisq
