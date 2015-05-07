# Licensed under an MIT open source license - see LICENSE

from cores import *
from length import *
from pixel_ident import *
from utilities import *
from width import *
from rollinghough import rht
from analysis import Analysis

import numpy as np
import matplotlib.pyplot as p
import scipy.ndimage as nd
from scipy.stats import lognorm
from scipy.ndimage import distance_transform_edt
from skimage.filter import threshold_adaptive
from skimage.morphology import remove_small_objects, medial_axis
from scipy.stats import scoreatpercentile
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
from copy import deepcopy
import os
import time
import warnings


class fil_finder_2D(object):

    """

    fil_finder is intended for use on astronomical images for detecting
    and analyzing filamentary structure in molecular clouds. Our method
    is largely based on mathematical morphology. When properly tuned, it
    is capable of extracting a complete catalog of filaments from an image
    over the complete range of intensities.

    This class acts as an overall wrapper to run the fil-finder algorithm
    on 2D images and enables visualization and saving capabilities.

    Parameters
    ------
    image : numpy.ndarray
        A 2D array of the data to be analyzed.
    hdr   : dictionary
        The header from fits file containing the data.
    beamwidth : float
        The FWHM beamwidth (in arcseconds) of the instrument used to
        take the data.
    skel_thresh : float
        Below this cut off, skeletons with less pixels will be deleted
    branch_thresh : float
        Any branches shorter than this length (in pixels) will be labeled as
        extraneous and pruned off.
    pad_size :  int
        The size of the pad (in pixels) used to pad the individual
        filament arrays.
        This is necessary to build the radial intensity profile.
    flatten_thresh : int
        The percentile of the data used in the normalization of the arctan
        transform. If the data contains regions of a much higher intensity
        than the mean, it is recommended this be set >95 percentile.
    smooth_size : int, optional
        The patch size (in pixels) used to smooth the flatten image before
        adaptive thresholding is performed. Smoothing is necessary to ensure
        the extraneous branches on the skeletons is minimized.
        If None, the patch size is set to ~0.05 pc. This ensures the large
        scale structure is not affected while smoothing extraneous pixels off
        the edges.
    size_thresh : int, optional
        This sets the lower threshold on the size of objects found in the
        adaptive thresholding. If None, the value is set at
        :math:`5\pi (0.1 \text(pc))^2` which is the area of the minimum dimensions
        expected for a filament. Any region smaller than this threshold may be
        safely labeled as an artifact of the thresholding.
    glob_thresh : float, optional
        This is the percentile to cut off searching for filamentary structure.
        Any regions with intensities below this percentile are ignored.
    adapt_thresh : int, optional
        This is the size of the patch used in the adaptive thresholding.
        Bright structure is not very sensitive to the choice of patch size,
        but faint structure is very sensitive. If None, the patch size is set
        to twice the width of a typical filament (~0.2 pc). As the width of
        filaments is somewhat ubiquitous, this patch size generally segments
        all filamentary structure in a given image.
    distance : float, optional
        The distance to the region being examined (in pc). If None, the
        analysis is carried out in pixel and angular units. In this case,
        the physical priors used in other optional parameters is meaningless
        and each must be specified initially.
    region_slice : list, optional
        This gives the option to examine a specific region in the given image.
        The expected input is [xmin,xmax,ymin,max].
    mask : numpy.ndarray, optional
        A pre-made, boolean mask may be supplied to skip the segmentation
        process. The algorithm will skeletonize and run the analysis portions
        only.
    freq : float
           Frequency of the image. This is required for using the cylindrical
           model (cyl_model) for the widths.

    Examples
    --------
    >>> from fil_finder import fil_finder_2D
    >>> from astropy.io.fits import getdata
    >>> img,hdr = \
    >>>     getdata("/srv/astro/erickoch/gould_belt/chamaeleonI-250.fits",
                    header=True)
    >>> filfind = fil_finder_2D(img, hdr, 15.1, 30, 5, 10, 95 , distance=160,
                                region_slice=[620,1400,430,1700])
    >>> filfind.run(verbose=False, save_name="chamaeleonI-250")


    References
    ----------

    """

    def __init__(self, image, hdr, beamwidth, skel_thresh=None,
                 branch_thresh=None, pad_size=10, flatten_thresh=None,
                 smooth_size=None, size_thresh=None, glob_thresh=None,
                 adapt_thresh=None, distance=None, region_slice=None,
                 mask=None, freq=None):

        img_dim = len(image.shape)
        if img_dim < 2 or img_dim > 2:
            raise TypeError(
                "Image must be 2D array. Input array has %s dimensions."
                % (img_dim))
        if region_slice is None:
            self.image = image
        else:
            slices = (slice(region_slice[0], region_slice[1], None),
                      slice(region_slice[2], region_slice[3], None))
            self.image = np.pad(image[slices], 1, padwithzeros)

        self.header = hdr
        self.skel_thresh = skel_thresh
        self.branch_thresh = branch_thresh
        self.pad_size = pad_size
        self.freq = freq

        # If pre-made mask is provided, remove nans if any.
        self.mask = None
        if mask is not None:
            mask[np.isnan(mask)] = 0.0
            self.mask = mask

        # Pad the image by the pad size. Avoids slicing difficulties
        # later on.
        self.image = np.pad(self.image, self.pad_size, padwithnans)

        # Make flattened image
        if flatten_thresh is None:
            # Fit to a log-normal
            fit_vals = lognorm.fit(self.image[~np.isnan(self.image)])

            median = lognorm.median(*fit_vals)
            std = lognorm.std(*fit_vals)
            thresh_val = median + 2*std
        else:
            thresh_val = scoreatpercentile(self.image[~np.isnan(self.image)],
                                           flatten_thresh)

        self.flat_img = np.arctan(self.image / thresh_val)

        if distance is None:
            print "No distance given. Results will be in pixel units."
            self.imgscale = 1.0  # pixel
            # where CDELT2 is in degrees
            self.beamwidth = beamwidth * (hdr["CDELT2"] * 3600) ** (-1)
            self.pixel_unit_flag = True
        else:
            self.imgscale = (hdr['CDELT2'] * (np.pi / 180.0) * distance)  # pc
            self.beamwidth = (
                beamwidth / np.sqrt(8 * np.log(2.))) * \
                (2 * np.pi / 206265.) * distance
            self.pixel_unit_flag = False

        # Angular conversion (sr/pixel^2)
        self.angular_scale = ((hdr['CDELT2'] * u.degree) ** 2.).to(u.sr).value

        self.glob_thresh = glob_thresh
        self.adapt_thresh = adapt_thresh
        self.smooth_size = smooth_size
        self.size_thresh = size_thresh

        self.width_fits = {"Parameters": [], "Errors": [], "Names": None}
        self.rht_curvature = {"Median": [], "IQR": []}
        self.filament_arrays = {}

    def create_mask(self, glob_thresh=None, adapt_thresh=None,
                    smooth_size=None, size_thresh=None, verbose=False,
                    test_mode=False, regrid=True, border_masking=True,
                    zero_border=False):
        '''

        This runs the complete segmentation process and returns a mask of the
        filaments found. The process is broken into six steps:
        *   An arctan tranform is taken to flatten extremely bright regions.
            Adaptive thresholding is very sensitive to local intensity changes
            and small, bright objects(ie. cores) will leave patch-sized holes
            in the mask.
        *   The flattened image is smoothed over with a median filter.
            The size of the patch used here is set to be much smaller than the
            typical filament width. Smoothing is necessary to minimizing
            extraneous branches when the medial axis transform is taken.
        *   A binary opening is performed using an 8-connected structure
            element. This is very successful at removing small regions around
            the edge of the data.
        *   Objects smaller than a certain threshold (set to be ~1/10 the area
            of a small filament) are removed to ensure only regions which are
            sufficiently large enough to be real structure remain.

        The parameters for this function are as previously defined.
        They are included here for fine-tuning purposes only.

        Parameters
        ----------
        smooth_size : int, optional
            See previous definition.
        size_thresh : int, optional
            See previous definition.
        glob_thresh : float, optional
            See previous definition.
        adapt_thresh : int, optional
            See previous definition.
        verbose : bool, optional
            Enables plotting.
        test_mode : bool, optional
            Plot each masking step.
        zero_border : bool, optional
            Replaces the NaN border with zeros for the adaptive thresholding.
            This is useful when emission continues to the edge of the image.

        Returns
        -------
        self.mask : numpy.ndarray
            The mask.

        '''

        if self.mask is not None:
            warnings.warn("Using inputted mask. Skipping creation of a new mask.")
            return self  # Skip if pre-made mask given

        if glob_thresh is not None:
            self.glob_thresh = glob_thresh
        if adapt_thresh is not None:
            self.adapt_thresh = adapt_thresh
        if smooth_size is not None:
            self.smooth_size = smooth_size
        if size_thresh is not None:
            self.size_thresh = size_thresh

        if self.size_thresh is None:
            if self.beamwidth == 0.0:
                warnings.warn("Beam width is set to 0.0."
                              "The size threshold is then 0. It is recommended"
                              "that size_thresh is manually set.")
            self.size_thresh = round(
                np.pi * 5 * (0.1)**2. * self.imgscale ** -2)
            # Area of ellipse for typical filament size. Divided by 10 to
            # incorporate sparsity.
        if self.adapt_thresh is None:
            # twice average FWHM for filaments
            self.adapt_thresh = round(0.2 / self.imgscale)
        if self.smooth_size is None:
            # half average FWHM for filaments
            self.smooth_size = round(0.05 / self.imgscale)

        # Check if regridding is even necessary
        if self.adapt_thresh >= 40 and regrid:
            regrid = False
            warnings.warn("Adaptive thresholding patch is larger than 40"
                          "pixels. Regridding has been disabled.")

        # Adaptive thresholding can't handle nans, so we create a nan mask
        # by finding the large, outer regions, smoothing with a large median
        # filter and eroding it.

        # Make a copy of the flattened image
        flat_copy = self.flat_img.copy()

        # Make the nan mask
        if border_masking:
            nan_mask = np.isnan(flat_copy)
            nan_mask = remove_small_objects(nan_mask, min_size=50,
                                            connectivity=8)
            nan_mask = np.logical_not(nan_mask)

            nan_mask = nd.median_filter(nan_mask, 25)
            nan_mask = nd.binary_erosion(nan_mask, eight_con(),
                                         iterations=15)
        else:
            nan_mask = np.logical_not(np.isnan(flat_copy))

        # Perform regridding
        if regrid:
            # Remove nans in the copy
            flat_copy[np.isnan(flat_copy)] = 0.0

            # Calculate the needed zoom to make the patch size ~40 pixels
            ratio = 40 / self.adapt_thresh
            # Round to the nearest factor of 2
            regrid_factor = np.min([2., int(round(ratio/2.0)*2.0)])

            # Defaults to cubic interpolation
            masking_img = nd.zoom(flat_copy, (regrid_factor, regrid_factor))
        else:
            regrid_factor = 1
            ratio = 1
            masking_img = flat_copy

        smooth_img = nd.median_filter(masking_img,
                                      size=round(self.smooth_size*ratio))

        # Set the border to zeros for the adaptive thresholding. Avoid border
        # effects.
        if zero_border:
            smooth_img[:self.pad_size*ratio+1, :] = 0.0
            smooth_img[-self.pad_size*ratio-1:, :] = 0.0
            smooth_img[:, :self.pad_size*ratio+1] = 0.0
            smooth_img[:, -self.pad_size*ratio-1:] = 0.0

        adapt = threshold_adaptive(smooth_img,
                                   round(ratio * self.adapt_thresh),
                                   method="mean")

        if regrid:
            regrid_factor = float(regrid_factor)
            adapt = nd.zoom(adapt, (1/regrid_factor, 1/regrid_factor), order=0)

        # Remove areas near the image border
        adapt = adapt * nan_mask

        if self.glob_thresh is not None:
            thresh_value = \
                np.max([0.0,
                        scoreatpercentile(self.flat_img[~np.isnan(self.flat_img)],
                                          self.glob_thresh)])
            glob = flat_copy > thresh_value
            adapt = glob * adapt

        opening = nd.binary_opening(adapt, structure=np.ones((3, 3)))
        cleaned = \
            remove_small_objects(opening, min_size=self.size_thresh)

        # Remove small holes within the object
        mask_objs, num, corners = \
            isolateregions(cleaned, fill_hole=True, rel_size=10,
                           morph_smooth=True)
        self.mask = recombine_skeletons(mask_objs,
                                        corners, self.image.shape,
                                        self.pad_size, verbose=True)

        ## WARNING!! Setting some image values to 0 to avoid negative weights.
        ## This may cause issues, however it will allow for proper skeletons
        self.image[np.where((self.mask * self.image) < 0.0)] = 0

        if test_mode:
            # p.subplot(3,3,1)
            p.imshow(np.log10(self.image), origin="lower", interpolation=None)
            p.colorbar()
            p.show()
            # p.subplot(3,3,2)
            p.imshow(masking_img, origin="lower", interpolation=None)
            p.colorbar()
            p.show()
            # p.subplot(3,3,3)
            p.imshow(smooth_img, origin="lower", interpolation=None)
            p.colorbar()
            p.show()
            # p.subplot(3,3,4)
            p.imshow(adapt, origin="lower", interpolation=None)
            p.show()
            # p.subplot(3,3,5)
            p.imshow(opening, origin="lower", interpolation=None)
            p.show()
            # p.subplot(3,3,6)
            p.imshow(cleaned, origin="lower", interpolation=None)
            p.show()
            # p.subplot(3,3,7)
            p.imshow(self.mask, origin="lower", interpolation=None)
            p.show()

        if verbose:
            p.imshow(self.flat_img, interpolation=None, origin="lower")
            p.contour(self.mask, colors="k")
            p.title("Mask on Flattened Image.")
            p.show()

        return self

    def medskel(self, return_distance=True, verbose=False):
        '''

        This function performs the medial axis transform (skeletonization)
        on the mask. This is essentially a wrapper function of
        skimage.morphology.medial_axis with the ability to delete narrow
        regions in the mask.

        If the distance transform is returned from the transform, it is used
        as a pruning step. Regions where the width of a region are far too
        small (set to >0.01 pc) are deleted. This ensures there no unnecessary
        connections between filaments.

        Parameters
        ----------
        return_distance : bool, optional
            This sets whether the distance transform is returned from
            skimage.morphology.medial_axis.

        verbose : bool, optional
            Enables plotting.

        Returns
        -------

        self.skeleton : numpy.ndarray
            The array containing all of the skeletons.

        self.medial_axis_distance : numpy.ndarray
            The distance transform used to create the skeletons.

        '''

        if return_distance:
            self.skeleton, self.medial_axis_distance = medial_axis(
                self.mask, return_distance=return_distance)
            self.medial_axis_distance = self.medial_axis_distance * \
                self.skeleton
            # Delete connection smaller than 2 pixels wide. Such a small
            # connection is more likely to be from limited pixel resolution
            # rather than actual structure.
            width_threshold = 1
            narrow_pts = np.where(self.medial_axis_distance < width_threshold)
            self.skeleton[narrow_pts] = 0  # Eliminate narrow connections
            self.medial_axis_distance[narrow_pts] = 0

        else:
            self.skeleton = medial_axis(self.mask)
            self.medial_axis_skeleton = None

        if verbose:  # For examining results of skeleton
            p.imshow(self.flat_img, interpolation=None, origin="lower")
            p.contour(self.skeleton, colors="k")
            p.show()

        return self

    def analyze_skeletons(self, relintens_thresh=0.2, nbeam_lengths=3,
                          skel_thresh=None, branch_thresh=None,
                          verbose=False):
        '''

        This function wraps most of the skeleton analysis. Several steps are
        completed here:
        *   isolatefilaments is run to separate each skeleton into its own
            array. If the skeletons are under the threshold set by
            self.size_thresh, the region is removed. An updated mask is
            also returned.
        *   pix_identify classifies each of the pixels in a skeleton as a
            body, end, or interestion point. See the documentation on find_filpix
            for a complete explanation. The function labels the branches and
            intersections of each skeletons.
        *   init_lengths finds the length of each branch in each skeleton and
            also returns the coordinates of each of these branches for use in
            the graph representation.
        *   pre_graph turns the skeleton structures into a graphing format
            compatible with networkx. Hubs in the graph are the intersections
            and end points, labeled as letters and numbers respectively. Edges
            define the connectivity of the hubs and they are weighted by their
            length.
        *   longest_path utilizes networkx.shortest_path_length to find the
            overall length of each of the filaments. The returned path is the
            longest path through the skeleton. If loops exist in the skeleton,
            the longest path is chosen (this shortest path algorithm fails when
            used on loops).
        *   extremum_pts returns the locations of the longest path's extent
            so its performance can be evaluated.
        *   final_lengths takes the path returned from longest_path and
            calculates the overall length of the filament. This step also acts
            as to prune the skeletons.
        *   final_analysis combines the outputs and returns the results for
            further analysis.

        Parameters
        ----------

        verbose : bool, optional
            Enables plotting.
        relintens_thresh : float, optional
            Relative intensity threshold for pruning. Sets the importance
            a branch must have in intensity relative to all other branches
            in the skeleton. Must be between (0.0, 1.0].
        nbeam_lengths : float or int, optional
            Sets the minimum skeleton length based on the number of beam
            sizes specified.
        skel_thresh : float, optional
            Manually set the minimum skeleton threshold. Overrides all
            previous settings.
        branch_thresh : float, optional
            Manually set the minimum branch length threshold. Overrides all
            previous settings.

        Returns
        -------
        self.filament_arrays : list of numpy.ndarray
                               Contains individual arrays of each skeleton
        self.number_of_filaments : int
                                   The number of individual filaments.
        self.array_offsets : list
            A list of coordinates for each filament array.This will
            be used to recombine the final skeletons into one array.
        self.filament_extents : list
            This contains the coordinates of the initial and final
            position of the skeleton's extent. It may be used to
            test the performance of the shortest path algorithm.
        self.lengths : list
            Contains the overall lengths of the skeletons
        self.labeled_fil_arrays : list of numpy.ndarray
            Contains the final labeled versions of the skeletons.
        self.branch_properties : dict
            The significant branches of the skeletons have their length
            and number of branches in each skeleton stored here.
            The keys are: *filament_branches*, *branch_lengths*

        '''

        if relintens_thresh > 1.0 or relintens_thresh <= 0.0:
            raise ValueError(
                "relintens_thresh must be set between (0.0, 1.0].")

        # Set the skeleton length threshold to some factor of the beam width
        if self.skel_thresh is None:
            self.skel_thresh = \
                round( self.beamwidth * nbeam_lengths / self.imgscale)
        elif skel_thresh is not None:
            self.skel_thresh = skel_thresh

        # Set the minimum branch length to be the beam size.
        if self.branch_thresh is None:
            self.branch_thresh = \
                round( self.beamwidth / self.imgscale)
        elif branch_thresh is not None:
            self.branch_thresh = branch_thresh

        isolated_filaments, num, offsets = \
            isolateregions(self.skeleton, size_threshold=self.skel_thresh,
                           pad_size=self.pad_size)
        self.number_of_filaments = num
        self.array_offsets = offsets

        interpts, hubs, ends, filbranches, labeled_fil_arrays =  \
            pix_identify(isolated_filaments, num)

        self.branch_properties = init_lengths(
            labeled_fil_arrays, filbranches, self.array_offsets, self.image)
        # Add the number of branches onto the dictionary
        self.branch_properties["number"] = filbranches

        edge_list, nodes = pre_graph(
            labeled_fil_arrays, self.branch_properties, interpts, ends)

        max_path, extremum, G = \
            longest_path(edge_list, nodes,
                         verbose=verbose,
                         skeleton_arrays=labeled_fil_arrays,
                         lengths=self.branch_properties["length"])

        updated_lists = \
            prune_graph(G, nodes, edge_list, max_path, labeled_fil_arrays,
                        self.branch_properties, self.branch_thresh,
                        relintens_thresh=relintens_thresh)

        labeled_fil_arrays, edge_list, nodes, self.branch_properties = \
            updated_lists

        self.filament_extents = extremum_pts(
            labeled_fil_arrays, extremum, ends)

        length_output = main_length(max_path, edge_list, labeled_fil_arrays,
                                    interpts, self.branch_properties[
                                        "length"], self.imgscale,
                                    verbose=verbose)

        self.lengths, self.filament_arrays["long path"] = length_output
        # Convert lengths to numpy array
        self.lengths = np.asarray(self.lengths)

        self.filament_arrays["final"] = make_final_skeletons(
            labeled_fil_arrays, interpts, verbose=verbose)

        self.labelled_filament_arrays = labeled_fil_arrays

        # Convert branch lengths physical units
        for n in range(self.number_of_filaments):
            lengths = self.branch_properties["length"][n]
            self.branch_properties["length"][n] = [
                self.imgscale * length for length in lengths]

        self.skeleton = \
            recombine_skeletons(self.filament_arrays["final"],
                                self.array_offsets, self.image.shape,
                                self.pad_size, verbose=True)

        self.skeleton_longpath = \
            recombine_skeletons(self.filament_arrays["long path"],
                                self.array_offsets, self.image.shape,
                                self.pad_size, verbose=True)

        return self

    def exec_rht(self, radius=10, ntheta=180, background_percentile=25,
                 branches=False, min_branch_length=3, verbose=False):
        '''

        Implements the Rolling Hough Transform (Clark et al., 2013).
        The orientation of each filament is denoted by the mean value of the
        RHT, which from directional statistics can be defined as:
        :math:`\\langle\\theta \\rangle = \\frac{1}{2} \\tan^{-1}\\left(\\frac{\\Sigma_i w_i\\sin2\\theta_i}{\\Sigma_i w_i\\cos2\\theta_i}\\right)`
        where :math:`w_i` is the normalized value of the RHT at
        :math:`\\theta_i`. This definition assumes that :math:`\\Sigma_iw_i=1`.
        :math:`\\theta` is defined on :math:`\\left[-\\pi/2, \\pi/2\\right)`.
        "Curvature" is represented by the IQR confidence interval about the mean,
        :math:`\\langle\\theta \\rangle \\pm \\sin^{-1} \\left( u_{\\alpha} \\sqrt{ \\frac{1-\\alpha}{2R^2} } \\right)`
        where :math:`u_{\\alpha}` is the z-score of the two-tail probability,
        :math:`\\alpha=\\Sigma_i\\cos{\\left[2w_i\\left(\\theta_i-\\langle\\theta\\rangle\\right)\\right]}`
        is the estimated weighted second trigonometric moment and
        :math:`R^2=\\left[\\left(\\Sigma_iw_i\\sin{\\theta_i}\\right)^2 +\\left(\\Sigma_iw_i\\cos{\\theta_i}\\right)^2\\right]`
        is the weighted length of the vector.

        These equations can be found in Fisher & Lewis (1983).

        Parameters
        ----------

        radius : int
            Sets the patch size that the RHT uses.

        ntheta : int, optional
            The number of bins to use for the RHT.

        background : int, optional
            RHT distribution often has a constant background. This sets the
            percentile to subtract off.

        branches : bool, optional
            If enabled, runs the RHT on individual branches in the skeleton.

        min_branch_length : int, optional
            Sets the minimum pixels a branch must have to calculate the RHT

        verbose : bool, optional
            Enables plotting.

        Returns
        -------

        self.rht_curvature : dict
            Contains the median and IQR for each filament.

        References
        ----------

        Clark et al. (2014)

        '''

        if not self.rht_curvature["Median"]:
            pass
        else:
            self.rht_curvature = {"Median": [], "IQR": []}

        # Flag branch output
        self._rht_branches_flag = False
        if branches:
            self._rht_branches_flag = True
            # Set up new dict entries.
            self.rht_curvature["Intensity"] = []
            self.rht_curvature["Length"] = []

        for n in range(self.number_of_filaments):
        # Need to correct for how image is read in
        # fliplr aligns angles with image when shown in ds9
            if branches:
                # We need intermediary arrays now
                medians = np.array([])
                iqrs = np.array([])
                intensity = np.array([])
                lengths = np.array([])
                # See above comment (613-614)
                skel_arr = np.fliplr(self.filament_arrays["final"][n])
                # Return the labeled skeleton without intersections
                output = \
                    pix_identify([skel_arr], 1)[-2:]
                labeled_fil_array = output[1]
                filbranch = output[0]
                branch_properties = init_lengths(labeled_fil_array,
                                                 filbranch,
                                                 [self.array_offsets[n]],
                                                 self.image)
                labeled_fil_array = labeled_fil_array[0]
                filbranch = filbranch[0]

                # Return the labeled skeleton without intersections
                labeled_fil_array = pix_identify([skel_arr], 1)[-1][0]
                branch_labels = \
                    np.unique(labeled_fil_array[np.nonzero(labeled_fil_array)])

                for val in branch_labels:
                    length = branch_properties["length"][0][val-1]
                    # Only include the branches with >10 pixels
                    if length < min_branch_length:
                        continue
                    theta, R, quantiles = \
                        rht(labeled_fil_array == val,
                            radius, ntheta, background_percentile)

                    twofive, median, sevenfive = quantiles

                    medians = np.append(medians, median)
                    if sevenfive > twofive:
                        iqrs = \
                            np.append(iqrs,
                                      np.abs(sevenfive - twofive))
                    else:  #
                        iqrs = \
                            np.append(iqrs,
                                      np.abs(sevenfive - twofive) + np.pi)
                    intensity = np.append(intensity, branch_properties["intensity"][0][val-1])
                    lengths = np.append(lengths, branch_properties["length"][0][val-1])

                self.rht_curvature["Median"].append(medians)
                self.rht_curvature["IQR"].append(iqrs)
                self.rht_curvature["Intensity"].append(intensity)
                self.rht_curvature["Length"].append(lengths)

            else:
                skel_arr = np.fliplr(self.filament_arrays["long path"][n])
                theta, R, quantiles = rht(
                    skel_arr, radius, ntheta, background_percentile)

                twofive, median, sevenfive = quantiles

                self.rht_curvature["Median"].append(median)
                if sevenfive > twofive:
                    self.rht_curvature["IQR"].append(
                        np.abs(sevenfive - twofive))  # Interquartile range
                else:  #
                    self.rht_curvature["IQR"].append(
                        np.abs(sevenfive - twofive + np.pi))

                if verbose:
                    ax1 = p.subplot(121, polar=True)
                    ax1.plot(2 * theta, R / R.max(), "kD")
                    ax1.fill_between(2 * theta, 0,
                                     R[:, 0] / R.max(),
                                     facecolor="blue",
                                     interpolate=True, alpha=0.5)
                    ax1.set_rmax(1.0)
                    ax1.plot([2 * median] * 2, np.linspace(0.0, 1.0, 2), "g")
                    ax1.plot([2 * twofive] * 2, np.linspace(0.0, 1.0, 2),
                             "b--")
                    ax1.plot([2 * sevenfive] * 2, np.linspace(0.0, 1.0, 2),
                             "b--")
                    p.subplot(122)
                    p.imshow(self.filament_arrays["long path"][n],
                             cmap="binary", origin="lower")
                    p.show()

        return self

    def find_widths(self, fit_model=gauss_model, try_nonparam=True,
                    verbose=False):
        '''

        The final step of the algorithm is to find the widths of each
        of the skeletons. We do this by:
        *   A Euclidean Distance Transform is performed on each skeleton.
            The skeletons are also recombined onto a single array. The
            individual filament arrays are padded to ensure a proper radial
            profile is created. If the padded arrays fall outside of the
            original image, they are trimmed.
        *   A user-specified model is fit to each of the radial profiles.
            There are three models included in this package; a gaussian,
            lorentzian and a cylindrical filament model
            (Arzoumanian et al., 2011). This returns the width and central
            intensity of each filament. The reported widths are the
            deconvolved FWHM of the gaussian width. For faint or crowded
            filaments, the fit can fail due to lack of data to fit to.
            In this case, we estimate the width non-parametrically.

        Parameters
        ----------

        fit_model : function
            Function to fit to the radial profile.

        try_nonparam : bool, optional
            If True, uses a non-parametric method to find the properties of
            the radial profile in cases where the model fails.

        verbose : bool, optional
            Enables plotting.

        Returns
        -------

        self.widths : list
            List of the FWHM widths returned from the fits.

        self.width_fits : dict
            Contains the fit parameters and estimations of the errors
            from each fit.

        self.skeleton : numpy.ndarray
            Updated versions of the array of skeletons.

        '''

        dist_transform_all, dist_transform_separate = \
            dist_transform(self.filament_arrays["final"],
                           self.skeleton)

        def red_chisq(data, fit, nparam, sd):
            N = data.shape[0]
            return np.sum(((fit - data) / sd) ** 2.) / float(N - nparam - 1)

        for n in range(self.number_of_filaments):

            # Need the unbinned data for the non-parametric fit.
            dist, radprof, weights, unbin_dist, unbin_radprof = \
                radial_profile(self.image, dist_transform_all,
                               dist_transform_separate[n],
                               self.array_offsets[n], self.imgscale)

            if fit_model == cyl_model:
                if self.freq is None:
                    print('''Image not converted to column density.
                             Fit parameters will not match physical meaning.
                             lease specify frequency.''')
                else:
                    assert isinstance(self.freq, float)
                    radprof = dens_func(
                        planck(20., self.freq), 0.2, radprof) * (5.7e19)

            fit, fit_error, model, parameter_names, fail_flag = \
                fit_model(dist, radprof, weights, self.beamwidth)

            # Get the function's name to track where fit values come from
            fit_type = str(model.__name__)

            if not fail_flag:
                chisq = red_chisq(radprof, model(dist, *fit[:-1]), 3, 1)
            else:
                # Give a value above threshold to try non-parametric fit
                chisq = 11.0

            # If the model isn't doing a good job, try it non-parametrically
            if chisq > 10.0 and try_nonparam:
                fit, fit_error, fail_flag = \
                    nonparam_width(dist, radprof, unbin_dist, unbin_radprof,
                                   self.beamwidth, 5, 99)
                # Change the fit type.
                fit_type = "nonparam"

            if n == 0:
                # Prepare the storage
                self.width_fits["Parameters"] = np.empty(
                    (self.number_of_filaments, len(parameter_names)))
                self.width_fits["Errors"] = np.empty(
                    (self.number_of_filaments, len(parameter_names)))
                self.width_fits["Type"] = np.empty(
                    (self.number_of_filaments), dtype="S")
                self.total_intensity = np.empty(
                    (self.number_of_filaments, ))

            if verbose:
                print "%s in %s" % (n, self.number_of_filaments)
                print "Fit Parameters: %s " % (fit)
                print "Fit Errors: %s" % (fit_error)
                print "Fit Type: %s" % (fit_type)
                p.subplot(121)
                p.plot(dist, radprof, "kD")
                points = np.linspace(np.min(dist), np.max(dist), 2 * len(dist))
                try:  # If FWHM is appended on, will get TypeError
                    p.plot(points, model(points, *fit), "r")
                except TypeError:
                    p.plot(points, model(points, *fit[:-1]), "r")
                p.xlabel(r'Radial Distance (pc)')
                p.ylabel(r'Intensity')
                p.grid(True)
                p.subplot(122)
                xlow, ylow = (
                    self.array_offsets[n][0][0], self.array_offsets[n][0][1])
                xhigh, yhigh = (
                    self.array_offsets[n][1][0], self.array_offsets[n][1][1])
                shape = (xhigh - xlow, yhigh - ylow)
                p.contour(self.filament_arrays["final"][n]
                          [self.pad_size:shape[0] - self.pad_size,
                           self.pad_size:shape[1] - self.pad_size], colors="k")
                img_slice = self.image[xlow + self.pad_size:xhigh - self.pad_size,
                                       ylow + self.pad_size:yhigh - self.pad_size]
                vmin = scoreatpercentile(img_slice[np.isfinite(img_slice)], 10)
                p.imshow(img_slice, interpolation=None, vmin=vmin)
                p.colorbar()
                p.show()

            # Final width check -- make sure length is longer than the width.
            # If it is, add the width onto the length since the adaptive
            # thresholding shortens each edge by the about the same.
            if self.lengths[n] > fit[-1]:
                self.lengths[n] += fit[-1]
            else:
                fail_flag = True

            # If fail_flag has been set to true in any of the fitting steps,
            # set results to nans
            if fail_flag:
                fit = [np.NaN] * len(fit)
                fit_error = [np.NaN] * len(fit)

            # Using the unbinned profiles, we can find the total filament
            # brightness. This can later be used to estimate the mass
            # contained in each filament.
            within_width = np.where(unbin_dist <= fit[1])
            if within_width[0].size:  # Check if its empty
                # Subtract off the estimated background
                fil_bright = unbin_radprof[within_width] - fit[2]
                sum_bright = np.sum(fil_bright[fil_bright >= 0], axis=None)
                self.total_intensity[n] = sum_bright * self.angular_scale
            else:
                self.total_intensity[n] = np.NaN

            self.width_fits["Parameters"][n, :] = fit
            self.width_fits["Errors"][n, :] = fit_error
            self.width_fits["Type"][n] = fit_type
        self.width_fits["Names"] = parameter_names

        return self

    def compute_filament_brightness(self):
        '''
        Returns the median brightness along the skeleton of the filament.
        '''

        self.filament_brightness = []

        labels, n = nd.label(self.skeleton, eight_con())

        for n in range(1, self.number_of_filaments+1):
            values = self.image[np.where(labels == n)]
            self.filament_brightness.append(np.median(values))

        return self

    def filament_model(self, max_radius=25):
        '''
        Returns a model of the diffuse filamentary network based
        on the radial profiles.

        Parameters
        ----------
        max_radius : int, optional
            Number of pixels to extend profiles to.

        Returns
        -------
        model_image : numpy.ndarray
            Array of the model

        '''

        if len(self.width_fits['Parameters']) == 0:
            raise TypeError("Run profile fitting first!")

        params = self.width_fits['Parameters']
        scale = self.imgscale

        # Create the distance transforms
        all_fils = dist_transform(self.filament_arrays["final"],
                                  self.skeleton)[0]

        model_image = np.zeros(all_fils.shape)

        for param, offset, fil_array in zip(params, self.array_offsets,
                                            self.filament_arrays["final"]):
            if np.isnan(param).any():
                continue
            # Avoid issues with the sizes of each filament array
            full_size = np.ones(model_image.shape)
            skel_posns = np.where(fil_array >= 1)
            full_size[skel_posns[0] + offset[0][0],
                      skel_posns[1] + offset[0][1]] = 0
            dist_array = distance_transform_edt(full_size)
            posns = np.where(dist_array < max_radius)
            model_image[posns] += \
                (param[0] - param[2]) * \
                np.exp(-np.power(dist_array[posns], 2) /
                       (2*(param[1]/scale)**2))

        return model_image

    def find_covering_fraction(self, max_radius=25):
        '''
        Compute the fraction of the intensity in the image contained in
        the filamentary structure.

        Parameters
        ----------
        max_radius : int, optional
            Passed to :method:`filament_model`
        '''

        fil_model = self.filament_model(max_radius=25)

        self.covering_fraction = np.nansum(fil_model) / np.nansum(self.image)

        return self

    def save_table(self, table_type="csv", path=None, save_name=None):
        '''

        The results of the algorithm are saved as an Astropy table in a 'csv',
        'fits' or latex format.

        Parameters
        ----------

        table_type : str, optional
            Sets the output type of the table. Supported options are
            "csv", "fits" and "latex".

        path : str, optional
            The path where the file should be saved.
        save_name : str, optional
            The prefix for the saved file.
            If None, the name from the header is used.

        Returns
        -------

        self.dataframe : astropy.Table
            The dataframe is returned for use with the Analysis class.

        '''

        if save_name is None:
            save_name = self.header["OBJECT"]

        save_name = save_name + "_table"

        if table_type == "csv":
            filename = save_name + ".csv"
        elif table_type == "fits":
            filename = save_name + ".fits"
        elif table_type == "latex":
            filename = save_name + ".tex"
        else:
            raise NameError("Only formats supported are 'csv', 'fits' \
                           and 'latex'.")

        # If path is specified, append onto filename.
        if path is not None:
            if path[-1] != "/":
                path = "".join(path, "/")
            filename = path + filename

        data = {"Lengths": self.lengths,
                "Orientation": self.rht_curvature["Median"],
                "Curvature": self.rht_curvature["IQR"],
                "Branches": self.branch_properties["number"],
                "Branch Length": self.branch_properties["length"],
                "Branch Intensity": self.branch_properties["intensity"],
                "Fit Type": self.width_fits["Type"],
                "Total Intensity": self.total_intensity,
                "Median Brightness": self.filament_brightness}

        for i, param in enumerate(self.width_fits["Names"]):
            data[param] = self.width_fits["Parameters"][:, i]
            data[param + " Error"] = self.width_fits["Errors"][:, i]

        if table_type == "csv":
            df = Table(data)
            df.write(filename, format="ascii.csv")

        elif table_type == "fits":
            warnings.warn("Entries containing lists have been deleted from \
                           FITS output due to incompatible format. If you  \
                           need these results, rerun save_table and save to\
                           a CSV file")
            # Branch Lengths and Intensity contains a list for each entry,
            # which aren't accepted for BIN tables.
            if "Branch Length" in data.keys():
                del data["Branch Length"]
                print("Deleted: Branch Length")
            if "Branch Intensity" in data.keys():
                del data["Branch Intensity"]
                print("Deleted: Branch Intensity")
            # If RHT is run on branches, we have to delete that too for FITS
            if self._rht_branches_flag:
                del data["Orientation"]
                del data["Curvature"]
                print("Deleted: Orientation, Curvature")

            df = Table(data)
            df.write(filename)

        elif table_type == "latex":
            df = Table(data)
            df.write(filename, format="ascii.latex")

        self.dataframe = df

        return self

    def save_fits(self, save_name=None, stamps=False, filename=None,
                  model_save=True):
        '''

        This function saves the mask and the skeleton array as FITS files.
        Included in the header are the setting used to create them.

        Parameters
        ----------

        save_name : str, optional
            The prefix for the saved file. If None, the name from the header
            is used.

        stamps : bool, optional
            Enables saving of individual stamps

        filename : str, optional
            File name of the image used. If None, assumes save_name is the
            file name.

        '''

        if not filename:  # Assume save_name is filename if not specified.
            filename = save_name

        # Create header based off of image header.
        new_hdr = deepcopy(self.header)

        try:  # delete the original history
            del new_hdr["HISTORY"]
        except KeyError:
            pass

        new_hdr.update("BUNIT", value="bool", comment="")
        new_hdr["COMMENT"] = "Mask created by fil_finder on " + \
            time.strftime("%c")
        new_hdr["COMMENT"] = \
            "See fil_finder documentation for more info on parameter meanings."
        new_hdr["COMMENT"] = "Smoothing Filter Size: " + \
            str(self.smooth_size) + " pixels"
        new_hdr["COMMENT"] = "Area Threshold: " + \
            str(self.size_thresh) + " pixels^2"
        new_hdr["COMMENT"] = "Global Intensity Threshold: " + \
            str(self.glob_thresh) + " %"
        new_hdr["COMMENT"] = "Size of Adaptive Threshold Patch: " + \
            str(self.adapt_thresh) + " pixels"
        new_hdr["COMMENT"] = "Original file name: " + filename

        # Remove padding
        mask = self.mask[
            self.pad_size:-self.pad_size, self.pad_size:-self.pad_size]

        # Save mask
        fits.writeto(
            "".join([save_name, "_mask.fits"]), mask.astype("float"), new_hdr)

        # Save skeletons. Includes final skeletons and the longest paths.
        new_hdr.update("BUNIT", value="int", comment="")
        new_hdr["COMMENT"] = "Skeleton Size Threshold: " + \
            str(self.skel_thresh)
        new_hdr["COMMENT"] = "Branch Size Threshold: " + \
            str(self.branch_thresh)

        hdu_skel = fits.HDUList()

        # Final Skeletons - create labels which match up with table output

        # Remove padding
        skeleton = self.skeleton[self.pad_size:-self.pad_size,
                                 self.pad_size:-self.pad_size]
        skeleton_long = self.skeleton_longpath[self.pad_size:-self.pad_size,
                                               self.pad_size:-self.pad_size]

        labels = nd.label(skeleton, eight_con())[0]
        hdu_skel.append(fits.PrimaryHDU(labels, header=new_hdr))

        # Longest Paths
        labels_lp = nd.label(skeleton_long, eight_con())[0]
        hdu_skel.append(fits.PrimaryHDU(labels_lp, header=new_hdr))

        hdu_skel.writeto("".join([save_name, "_skeletons.fits"]))

        if stamps:
            # Save stamps of all images. Include portion of image and the
            # skeleton for reference.

            # Make a directory for the stamps
            if not os.path.exists("stamps_" + save_name):
                os.makedirs("stamps_" + save_name)

            final_arrays = self.filament_arrays["final"]
            longpath_arrays = self.filament_arrays["long path"]

            for n, (offset, skel_arr, lp_arr) in \
              enumerate(zip(self.array_offsets,
                            final_arrays,
                            longpath_arrays)):

                xlow, ylow = (offset[0][0], offset[0][1])
                xhigh, yhigh = (offset[1][0], offset[1][1])

                # Create stamp
                img_stamp = self.image[xlow:xhigh,
                                       ylow:yhigh]

                # ADD IN SOME HEADERS!
                prim_hdr = deepcopy(self.header)
                prim_hdr["COMMENT"] = "Outputted from fil_finder."
                prim_hdr["COMMENT"] = \
                    "Extent in original array: (" + \
                    str(xlow + self.pad_size) + "," + \
                    str(ylow + self.pad_size) + ")->" + \
                    "(" + str(xhigh - self.pad_size) + \
                    "," + str(yhigh - self.pad_size) + ")"

                hdu = fits.HDUList()
                # Image stamp
                hdu.append(fits.PrimaryHDU(img_stamp, header=prim_hdr))
                # Stamp of final skeleton
                prim_hdr.update("BUNIT", value="bool", comment="")
                hdu.append(fits.PrimaryHDU(skel_arr, header=prim_hdr))
                # Stamp of longest path
                hdu.append(fits.PrimaryHDU(lp_arr, header=prim_hdr))

                hdu.writeto(
                    "stamps_" + save_name + "/" + save_name + "_object_" + str(n + 1) + ".fits")

        if model_save:
            model = self.filament_model()

            # Remove the padding
            model = model[self.pad_size:-self.pad_size,
                          self.pad_size:-self.pad_size]

            model_hdr = new_hdr.copy()

            model_hdr.update('BUNIT', value=self.header['BUNIT'], comment="")

            model_hdu = fits.PrimaryHDU(model, header=model_hdr)

            model_hdu.writeto("".join([save_name, "_filament_model.fits"]))

        return self

    def __str__(self):
        print("%s filaments found.") % (self.number_of_filaments)
        for fil in range(self.number_of_filaments):
            print "Filament: %s, Width: %s, Length: %s, Curvature: %s,\
                       Orientation: %s" % \
                (fil, self.width_fits["Parameters"][fil, -1][fil],
                 self.lengths[fil], self.rht_curvature["Std"][fil],
                 self.rht_curvature["Std"][fil])

    def run(self, verbose=False, save_name=None, save_plots=False):
        '''
        The whole algorithm in one easy step. Individual parameters have not
        been included in this batch run. If fine-tuning is needed, it is
        recommended to run each step individually.

        Parameters
        ----------
        verbose : bool
            Enables the verbose option for each of the steps.
        save_plots : bool
            Enables the saving of the output plots.
        save_name : str
            The prefix for the saved file.
            If None, the name from the header is used.

        '''

        if verbose:
            print "Best to run in pylab for verbose output."

        if save_name is None:
            save_name = self.header["OBJECT"]

        self.create_mask(verbose=verbose)
        self.medskel(verbose=verbose)

        self.analyze_skeletons(verbose=verbose)
        self.exec_rht(verbose=verbose)
        self.find_widths(verbose=verbose)
        self.compute_filament_brightness()
        self.find_covering_fraction()
        self.save_table(save_name=save_name, table_type="fits")
        self.save_table(save_name=save_name, table_type="csv")
        self.save_fits(save_name=save_name, stamps=False)

        if verbose:
            self.__str__()

        if save_plots:
            Analysis(self.dataframe, save_name=save_name).make_hists()
            # ImageAnalysis(self.image, self.mask, skeleton=self.skeleton, save_name=save_name)

        return self
