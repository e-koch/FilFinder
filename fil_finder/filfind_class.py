# Licensed under an MIT open source license - see LICENSE

import numpy as np
import matplotlib.pyplot as p
import scipy.ndimage as nd
from scipy.stats import lognorm
from scipy.ndimage import distance_transform_edt
from skimage.morphology import remove_small_objects, medial_axis
from scipy.stats import scoreatpercentile
from astropy.io import fits
from astropy.table import Table
from astropy import units as u
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from copy import deepcopy
import os
import time
import warnings

from .cores import *
from .length import *
from .pixel_ident import *
from .utilities import *
from .width import *
from .rollinghough import rht
from .io_funcs import input_data
from .base_conversions import BaseInfoMixin

# The try/except is here to deal with TypeErrors when building the docs on RTD
# This isn't really a solution... but it is lazy and does the job until I
# add astropy_helpers.
try:
    FWHM_FACTOR = 2 * np.sqrt(2 * np.log(2.))
except TypeError:
    FWHM_FACTOR = np.NaN


class fil_finder_2D(BaseInfoMixin):

    """
    This class acts as an overall wrapper to run the fil-finder algorithm
    on 2D images and contains visualization and saving capabilities.

    Parameters
    ----------
    image : numpy.ndarray or astropy.io.fits.PrimaryHDU
        A 2D array of the data to be analyzed. If a FITS HDU is passed, the
        header is automatically loaded.
    header : FITS header, optional
        The header from fits file containing the data. If no header is provided,
        and it could not be loaded from ``image``, all results will be returned
        in pixel units.
    beamwidth : float or astropy.units.Quantity, optional
        The FWHM beamwidth with an appropriately attached unit. By default,
        the beam is read from a provided header. If the beam cannot be read
        from the header, or a header is not provided, this input must be
        given. If a float is given, it is assumed to be in pixel units.
    ang_scale : `~astropy.units.Quantity`, optional
        Give the angular to pixel units conversion. If none is given, it will
        be read from the header. The units must be a valid angular unit.
    skel_thresh : float, optional
        Given in pixel units.Below this cut off, skeletons with less pixels
        will be deleted. The default value is 0.3 pc converted to pixels.
    branch_thresh : float, optional
        Any branches shorter than this length (in pixels) will be labeled as
        extraneous and pruned off. The default value is 3 times the FWHM
        beamwidth.
    pad_size :  int, optional
        The size of the pad (in pixels) used to pad the individual
        filament arrays. By default this is disabled. **If the data continue
        up to the edge of the image, padding should not be enabled as this
        causes deviations in the mask around the edges!**
    skeleton_pad_size : int, optional
        Number of pixels to pad the individual skeleton arrays by. For
        the skeleton to graph conversion, the pad must always be greater
        then 0. The amount of padding can slightly effect the extent of the
        radial intensity profile..
    flatten_thresh : int, optional
        The percentile of the data (0-100) to set the normalization of the arctan
        transform. By default, a log-normal distribution is fit and the
        threshold is set to :math:`\mu + 2\sigma`. If the data contains regions
        of a much higher intensity than the mean, it is recommended this
        be set >95 percentile.
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
        This is the percentile of the data to mask off. All intensities below
        are cut off from being included in the filamentary structure.
    adapt_thresh : int, optional
        This is the size in pixels of the patch used in the adaptive
        thresholding.  Bright structure is not very sensitive to the choice of
        patch size, but faint structure is very sensitive. If None, the patch
        size is set to twice the width of a typical filament (~0.2 pc). As the
        width of filaments is somewhat ubiquitous, this patch size generally
        segments all filamentary structure in a given image.
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
    freq : float, optional
        **Deprecated. Has no effect.**
    save_name : str, optional
        Sets the prefix name that is used for output files. Can be overridden
        in ``save_fits`` and ``save_table``. Default is "FilFinder_output".

    Examples
    --------
    >>> from fil_finder import fil_finder_2D
    >>> from astropy.io import fits
    >>> import astropy.units as u
    >>> img,hdr = fits.open("twod.fits")[0] # doctest: +SKIP
    >>> filfind = fil_finder_2D(img, header=hdr, beamwidth=15*u.arcsec, distance=170*u.pc, save_name='twod_filaments') # doctest: +SKIP
    >>> filfind.run(verbose=False) # doctest: +SKIP

    """

    def __init__(self, image, header=None, beamwidth=None, ang_scale=None,
                 skel_thresh=None, branch_thresh=None, pad_size=0,
                 skeleton_pad_size=1, flatten_thresh=None,
                 smooth_size=None, size_thresh=None, glob_thresh=None,
                 adapt_thresh=None, distance=None, region_slice=None,
                 mask=None, freq=None, save_name="FilFinder_output"):

        # Deprecating this version soon
        warnings.warn("Support for fil_finder_2D will be dropped in v2.0. Use "
                      "the new version 'FilFinder2D'.", DeprecationWarning)

        output = input_data(image, header)

        self._image = output["data"].value
        if "header" in output:
            self._header = output["header"]
        else:
            self._header = None

        if self.header is not None:
            self._wcs = WCS(self.header)

        if region_slice is not None:
            slices = (slice(region_slice[0], region_slice[1], None),
                      slice(region_slice[2], region_slice[3], None))
            self.image = np.pad(self.image[slices], 1, padwithzeros)

        self.skel_thresh = skel_thresh
        self.branch_thresh = branch_thresh
        self.pad_size = pad_size
        self.skeleton_pad_size = skeleton_pad_size
        self.freq = freq
        self.save_name = save_name

        # If pre-made mask is provided, remove nans if any.
        self.mask = None
        if mask is not None:
            mask[np.isnan(mask)] = 0.0
            self.mask = mask

        # Pad the image by the pad size. Avoids slicing difficulties
        # later on.
        if self.pad_size > 0:
            self._image = np.pad(self.image, self.pad_size, padwithnans)

        # Make flattened image
        if flatten_thresh is None:
            # Fit to a log-normal
            fit_vals = lognorm.fit(self.image[~np.isnan(self.image)])

            median = lognorm.median(*fit_vals)
            std = lognorm.std(*fit_vals)
            thresh_val = median + 2 * std
        else:
            thresh_val = scoreatpercentile(self.image[~np.isnan(self.image)],
                                           flatten_thresh)

        self.flat_img = np.arctan(self.image / thresh_val)

        if ang_scale is not None:
            if not isinstance(ang_scale, u.Quantity):
                raise TypeError("ang_scale must be an astropy.units.Quantity.")
            if not ang_scale.unit.is_equivalent(u.deg):
                raise u.UnitsError("ang_scale must be given in angular units.")

            pix_scale = ang_scale.to(u.deg)
        else:
            # Check for a wcs object
            if not hasattr(self, "wcs"):
                pix_scale = 1.
                Warning("No header given. Results will be in pixel units.")
            else:
                pix_scale = \
                    np.abs(proj_plane_pixel_scales(self.wcs.celestial)[0]) * \
                    u.deg

        if self.header is None:
            Warning("No header given. Results will be in pixel units.")

            if beamwidth is None:
                raise TypeError("Beamwidth in pixels must be given when no"
                                " header is provided.")
            elif isinstance(beamwidth, u.Quantity):
                # Can only use pixel inputs in this case
                if beamwidth.unit != u.pix:
                    raise TypeError("Beamwidth must be given in pixel units"
                                    " when no header is given.")
                else:
                    Warning("Assuming given beamwidth is in pixels.")
            self._beamwidth = beamwidth.value / FWHM_FACTOR
            self.angular_scale = 1.0
            self.imgscale = 1.0
            self.pixel_unit_flag = True

        else:
            # Check for the beam info in the header
            if "BMAJ" in self.header:
                beamwidth = self.header["BMAJ"] * u.deg
            else:
                if beamwidth is None:
                    raise TypeError("Beamwidth was not found in the header."
                                    " Must provide a value with the"
                                    " 'beamwidth' argument.")

                if not isinstance(beamwidth, u.Quantity):
                    Warning("No unit provided for the beamwidth. Assuming "
                            "pixels.")
                    beamwidth *= u.pix

            if distance is None:
                Warning("No distance given. Results will be in pixel units.")
                if beamwidth.unit == u.pix:
                    self._beamwidth = beamwidth.value / FWHM_FACTOR
                else:
                    self._beamwidth = ((beamwidth.to(u.deg) / FWHM_FACTOR) /
                                       pix_scale).value
                self.imgscale = 1.0
                self.pixel_unit_flag = True
            else:
                if not isinstance(distance, u.Quantity):
                    Warning("No unit for distance given. Assuming pc.")
                    distance *= u.pc

                # Image scale in pc.
                self.imgscale = pix_scale.to(u.rad).value * \
                    distance.to(u.pc).value

                width = beamwidth / FWHM_FACTOR
                if beamwidth.unit == u.pix:
                    self._beamwidth = width.value * self.imgscale
                else:
                    # Try to convert straight to pc
                    try:
                        self._beamwidth = width.to(u.pc).value
                        _try_ang_units = False
                    except u.UnitConversionError:
                        _try_ang_units = True

                    # If that fails, try converting from an angular unit
                    if _try_ang_units:
                        try:
                            self._beamwidth = \
                                (width.to(u.arcsec).value / 206265.) * \
                                distance.value
                        except u.UnitConversionError:
                            raise u.UnitConversionError("Cannot convert the "
                                                        "given beamwidth in "
                                                        " physical or angular"
                                                        " units.")
                self.pixel_unit_flag = False

            # Angular conversion (sr/pixel^2)
            if pix_scale.unit.is_equivalent(u.deg):
                self.angular_scale = \
                    (pix_scale ** 2.).to(u.sr).value
            else:
                self.angular_scale = 1 * u.pix**2

        self.glob_thresh = glob_thresh
        self.adapt_thresh = adapt_thresh
        self.smooth_size = smooth_size
        self.size_thresh = size_thresh

        self.width_fits = {"Parameters": [], "Errors": [], "Names": None}
        self.rht_curvature = {"Orientation": [], "Curvature": []}
        self.filament_arrays = {}

    @property
    def wcs(self):
        return self._wcs

    @property
    def header(self):
        return self._header

    @property
    def pad_size(self):
        return self._pad_size

    @pad_size.setter
    def pad_size(self, value):
        if value < 0:
            raise ValueError("Pad size must be >=0")
        self._pad_size = value

    @property
    def skeleton_pad_size(self):
        return self._skeleton_pad_size

    @skeleton_pad_size.setter
    def skeleton_pad_size(self, value):
        if value <= 0:
            raise ValueError("Skeleton pad size must be >0")
        self._skeleton_pad_size = value

    def create_mask(self, glob_thresh=None, adapt_thresh=None,
                    smooth_size=None, size_thresh=None, verbose=False,
                    test_mode=False, regrid=True, border_masking=True,
                    zero_border=False, fill_hole_size=None,
                    use_existing_mask=False, save_png=False):
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
            See definition in ``fil_finder_2D`` inputs.
        size_thresh : int, optional
            See definition in ``fil_finder_2D`` inputs.
        glob_thresh : float, optional
            See definition in ``fil_finder_2D`` inputs.
        adapt_thresh : int, optional
            See definition in ``fil_finder_2D`` inputs.
        verbose : bool, optional
            Enables plotting. Default is False.
        test_mode : bool, optional
            Plot each masking step. Default is False.
        regrid : bool, optional
            Enables the regridding of the image to larger sizes when the patch
            size for the adaptive thresholding is less than 40 pixels. This
            decreases fragmentation of regions due to pixellization effects.
            Default is True.
        border_masking : bool, optional
            Dilates a mask of the regions along the edge of the image to remove
            regions dominated by noise. Disabling leads to regions characterized
            at the image boundaries and should only be used if there is not
            significant noise at the edges. Default is True.
        zero_border : bool, optional
            Replaces the NaN border with zeros for the adaptive thresholding.
            This is useful when emission continues to the edge of the image.
            Default is False.
        fill_hole_size : int or float, optional
            Sets the maximum hole size to fill in the skeletons. If <1,
            maximum is that proportion of the total number of pixels in
            skeleton. Otherwise, it sets the maximum number of pixels.
            Defaults to a square area with length of the beamwidth.
        use_existing_mask : bool, optional
            If ``mask`` is already specified, enabling this skips
            recomputing the mask.
        save_png : bool, optional
            Saves the plot made in verbose mode. Disabled by default.

        Attributes
        ----------
        mask : numpy.ndarray
            The mask of filaments.

        '''

        if self.mask is not None and use_existing_mask:
            warnings.warn("Using inputted mask. Skipping creation of a new mask.")
            self.glob_thresh = 'usermask'
            self.adapt_thresh = 'usermask'
            self.size_thresh = 'usermask'
            self.smooth_size = 'usermask'

            return self  # Skip if pre-made mask given

        if glob_thresh is not None:
            self.glob_thresh = glob_thresh
        if adapt_thresh is not None:
            self.adapt_thresh = adapt_thresh
        if smooth_size is not None:
            self.smooth_size = smooth_size
        if size_thresh is not None:
            self.size_thresh = size_thresh

        if self.pixel_unit_flag:
            if smooth_size is None:
                raise ValueError("Distance not given. Must specify smooth_size"
                                 " in pixel units.")
            if adapt_thresh is None:
                raise ValueError("Distance not given. Must specify adapt_thresh"
                                 " in pixel units.")
            if size_thresh is None:
                raise ValueError("Distance not given. Must specify size_thresh"
                                 " in pixel units.")

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

        # Remove nans in the copy
        flat_copy[np.isnan(flat_copy)] = 0.0

        # Perform regridding
        if regrid:
            # Calculate the needed zoom to make the patch size ~40 pixels
            ratio = 40 / self.adapt_thresh
            # Round to the nearest factor of 2
            regrid_factor = np.min([2., int(round(ratio / 2.0) * 2.0)])

            # Defaults to cubic interpolation
            masking_img = nd.zoom(flat_copy, (regrid_factor, regrid_factor))
        else:
            regrid_factor = 1
            ratio = 1
            masking_img = flat_copy

        smooth_img = \
            nd.median_filter(masking_img,
                             size=int(round(self.smooth_size * ratio)))

        # Set the border to zeros for the adaptive thresholding. Avoid border
        # effects.
        if zero_border and self.pad_size > 0:
            pad_size = int(self.pad_size * ratio)
            smooth_img[:pad_size + 1, :] = 0.0
            smooth_img[-pad_size - 1:, :] = 0.0
            smooth_img[:, :pad_size + 1] = 0.0
            smooth_img[:, -pad_size - 1:] = 0.0

        adapt = threshold_local(smooth_img,
                                round_to_odd(ratio * self.adapt_thresh),
                                method="mean")

        if regrid:
            regrid_factor = float(regrid_factor)
            adapt = nd.zoom(adapt, (1 / regrid_factor, 1 / regrid_factor),
                            order=0)

        # Remove areas near the image border
        adapt = adapt * nan_mask

        if self.glob_thresh is not None:
            thresh_value = \
                np.max([0.0,
                        scoreatpercentile(self.flat_img[~np.isnan(self.flat_img)],
                                          self.glob_thresh)])
            glob = flat_copy > thresh_value
            adapt = glob * adapt

        cleaned = \
            remove_small_objects(adapt, min_size=self.size_thresh)

        # Remove small holes within the object

        if fill_hole_size is None:
            fill_hole_size = np.pi * (self.beamwidth / self.imgscale)**2

        mask_objs, num, corners = \
            isolateregions(cleaned, fill_hole=True, rel_size=fill_hole_size,
                           morph_smooth=True, pad_size=self.skeleton_pad_size)
        self.mask = recombine_skeletons(mask_objs,
                                        corners, self.image.shape,
                                        self.skeleton_pad_size)

        # WARNING!! Setting some image values to 0 to avoid negative weights.
        # This may cause issues, however it will allow for proper skeletons
        # Through all the testing and deriving science results, this has not
        # been an issue! EK
        self.image[np.where((self.mask * self.image) < 0.0)] = 0

        if test_mode:
            p.imshow(np.log10(self.image), origin="lower", interpolation=None,
                     cmap='binary')
            p.colorbar()
            p.show()
            p.imshow(masking_img, origin="lower", interpolation=None,
                     cmap='binary')
            p.colorbar()
            p.show()
            p.imshow(smooth_img, origin="lower", interpolation=None,
                     cmap='binary')
            p.colorbar()
            p.show()
            p.imshow(adapt, origin="lower", interpolation=None,
                     cmap='binary')
            p.show()
            p.imshow(cleaned, origin="lower", interpolation=None,
                     cmap='binary')
            p.show()
            p.imshow(self.mask, origin="lower", interpolation=None,
                     cmap='binary')
            p.show()

        if verbose or save_png:
            vmin = np.percentile(self.flat_img[np.isfinite(self.flat_img)], 20)
            vmax = np.percentile(self.flat_img[np.isfinite(self.flat_img)], 90)
            p.clf()
            p.imshow(self.flat_img, interpolation=None, origin="lower",
                     cmap='binary', vmin=vmin, vmax=vmax)
            p.contour(self.mask, colors="r")
            p.title("Mask on Flattened Image.")
            if save_png:
                try_mkdir(self.save_name)
                p.savefig(os.path.join(self.save_name, self.save_name + "_mask.png"))
            if verbose:
                p.show()
            if in_ipynb():
                p.clf()

    def medskel(self, verbose=False, save_png=False):
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
        verbose : bool, optional
            Enables plotting.
        save_png : bool, optional
            Saves the plot made in verbose mode. Disabled by default.

        Attributes
        ----------
        skeleton : numpy.ndarray
            The array containing all of the skeletons.
        medial_axis_distance : numpy.ndarray
            The distance transform used to create the skeletons.
        '''

        self.skeleton, self.medial_axis_distance = \
            medial_axis(self.mask, return_distance=True)
        self.medial_axis_distance = \
            self.medial_axis_distance * self.skeleton
        # Delete connection smaller than 2 pixels wide. Such a small
        # connection is more likely to be from limited pixel resolution
        # rather than actual structure.
        width_threshold = 1
        narrow_pts = np.where(self.medial_axis_distance < width_threshold)
        self.skeleton[narrow_pts] = 0  # Eliminate narrow connections
        self.medial_axis_distance[narrow_pts] = 0

        if verbose or save_png:  # For examining results of skeleton
            vmin = np.percentile(self.flat_img[np.isfinite(self.flat_img)], 20)
            vmax = np.percentile(self.flat_img[np.isfinite(self.flat_img)], 90)
            p.clf()
            p.imshow(self.flat_img, interpolation=None, origin="lower",
                     cmap='binary', vmin=vmin, vmax=vmax)
            p.contour(self.skeleton, colors="r")
            if save_png:
                try_mkdir(self.save_name)
                p.savefig(os.path.join(self.save_name,
                                       self.save_name + "_initial_skeletons.png"))
            if verbose:
                p.show()
            if in_ipynb():
                p.clf()

    def analyze_skeletons(self, prune_criteria='all', relintens_thresh=0.2,
                          nbeam_lengths=5, branch_nbeam_lengths=3,
                          skel_thresh=None, branch_thresh=None,
                          verbose=False, save_png=False):
        '''

        This function wraps most of the skeleton analysis. Several steps are
        completed here:

        *   isolatefilaments is run to separate each skeleton into its own
            array. If the skeletons are under the threshold set by
            self.size_thresh, the region is removed. An updated mask is
            also returned.
        *   pix_identify classifies each of the pixels in a skeleton as a
            body, end, or intersection point. See the documentation on find_filpix
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
        prune_criteria : {'all', 'intensity', 'length'}, optional
            Choose the property to base pruning on. 'all' requires that the
            branch fails to satisfy the length and relative intensity checks.
        relintens_thresh : float, optional
            Relative intensity threshold for pruning. Sets the importance
            a branch must have in intensity relative to all other branches
            in the skeleton. Must be between (0.0, 1.0].
        nbeam_lengths : float or int, optional
            Sets the minimum skeleton length based on the number of beam
            sizes specified.
        branch_nbeam_lengths : float or int, optional
            Sets the minimum branch length based on the number of beam
            sizes specified.
        skel_thresh : float, optional
            Manually set the minimum skeleton threshold. Overrides all
            previous settings.
        branch_thresh : float, optional
            Manually set the minimum branch length threshold. Overrides all
            previous settings.
        save_png : bool, optional
            Saves the plot made in verbose mode. Disabled by default.

        Attributes
        ----------
        filament_arrays : list of numpy.ndarray
            Contains individual arrays of each skeleton
        number_of_filaments : int
            The number of individual filaments.
        array_offsets : list
            A list of coordinates for each filament array.This will
            be used to recombine the final skeletons into one array.
        filament_extents : list
            This contains the coordinates of the initial and final
            position of the skeleton's extent. It may be used to
            test the performance of the shortest path algorithm.
        lengths : list
            Contains the overall lengths of the skeletons
        labeled_fil_arrays : list of numpy.ndarray
            Contains the final labeled versions of the skeletons.
        branch_properties : dict
            The significant branches of the skeletons have their length
            and number of branches in each skeleton stored here.
            The keys are: *filament_branches*, *branch_lengths*

        '''

        if relintens_thresh > 1.0 or relintens_thresh <= 0.0:
            raise ValueError(
                "relintens_thresh must be set between (0.0, 1.0].")

        if self.pixel_unit_flag:
            if self.skel_thresh is None and skel_thresh is None:
                raise ValueError("Distance not given. Must specify skel_thresh"
                                 " in pixel units.")

        # Set the skeleton length threshold to some factor of the beam width
        if self.skel_thresh is None:
            self.skel_thresh = round(0.3 / self.imgscale)
            # round( self.beamwidth * nbeam_lengths / self.imgscale)
        elif skel_thresh is not None:
            self.skel_thresh = skel_thresh

        # Set the minimum branch length to be the beam size.
        if self.branch_thresh is None:
            self.branch_thresh = \
                round(branch_nbeam_lengths * self.beamwidth / self.imgscale)
        elif branch_thresh is not None:
            self.branch_thresh = branch_thresh

        isolated_filaments, num, offsets = \
            isolateregions(self.skeleton, size_threshold=self.skel_thresh,
                           pad_size=self.skeleton_pad_size)
        self.number_of_filaments = num
        self.array_offsets = offsets

        interpts, hubs, ends, filbranches, labeled_fil_arrays =  \
            pix_identify(isolated_filaments, num)

        self.branch_properties = init_lengths(
            labeled_fil_arrays, filbranches, self.array_offsets, self.image)
        # Add the number of branches onto the dictionary
        self.branch_properties["number"] = filbranches

        edge_list, nodes, loop_edges = pre_graph(
            labeled_fil_arrays, self.branch_properties, interpts, ends)

        max_path, extremum, G = \
            longest_path(edge_list, nodes,
                         verbose=verbose,
                         save_png=save_png,
                         save_name=self.save_name,
                         skeleton_arrays=labeled_fil_arrays)

        updated_lists = \
            prune_graph(G, nodes, edge_list, max_path, labeled_fil_arrays,
                        self.branch_properties, loop_edges,
                        length_thresh=self.branch_thresh,
                        relintens_thresh=relintens_thresh,
                        prune_criteria=prune_criteria)

        labeled_fil_arrays, edge_list, nodes, self.branch_properties = \
            updated_lists

        self.filament_extents = extremum_pts(labeled_fil_arrays,
                                             extremum, ends)

        length_output = main_length(max_path, edge_list, labeled_fil_arrays,
                                    interpts,
                                    self.branch_properties["length"],
                                    self.imgscale,
                                    verbose=verbose, save_png=save_png,
                                    save_name=self.save_name)

        self.lengths, self.filament_arrays["long path"] = length_output
        # Convert lengths to numpy array
        self.lengths = np.asarray(self.lengths)

        self.filament_arrays["final"] =\
            make_final_skeletons(labeled_fil_arrays, interpts,
                                 verbose=verbose, save_png=save_png,
                                 save_name=self.save_name)

        self.labelled_filament_arrays = labeled_fil_arrays

        # Convert branch lengths physical units
        for n in range(self.number_of_filaments):
            lengths = self.branch_properties["length"][n]
            self.branch_properties["length"][n] = \
                [self.imgscale * length for length in lengths]

        self.skeleton = \
            recombine_skeletons(self.filament_arrays["final"],
                                self.array_offsets, self.image.shape,
                                self.skeleton_pad_size)

        self.skeleton_longpath = \
            recombine_skeletons(self.filament_arrays["long path"],
                                self.array_offsets, self.image.shape,
                                self.skeleton_pad_size)

    def exec_rht(self, radius=10, ntheta=180, background_percentile=25,
                 branches=False, min_branch_length=3, verbose=False,
                 save_png=False):
        '''

        Implements the Rolling Hough Transform (Clark et al., 2014).
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
        save_png : bool, optional
            Saves the plot made in verbose mode. Disabled by default.

        Attributes
        ----------
        rht_curvature : dict
            Contains the median and IQR for each filament.

        References
        ----------

        `Clark et al. (2014) <http://adsabs.harvard.edu/abs/2014ApJ...789...82C>`_
        `Fisher & Lewis (1983) <http://biomet.oxfordjournals.org/content/70/2/333.short>`_

        '''

        if not self.rht_curvature["Orientation"]:
            pass
        else:
            self.rht_curvature = {"Orientation": [],
                                  "Curvature": []}

        # Flag branch output
        self._rht_branches_flag = False
        if branches:
            self._rht_branches_flag = True
            # Set up new dict entries.
            self.rht_curvature["Intensity"] = []
            self.rht_curvature["Length"] = []

        # Need to correct for how image is read in
        # fliplr aligns angles with image when shown in ds9
        for n in range(self.number_of_filaments):
            if branches:
                # We need intermediary arrays now
                means = np.array([])
                iqrs = np.array([])
                intensity = np.array([])
                lengths = np.array([])

                # for val in branch_labels:
                for val, (pix, length) in enumerate(zip(self.branch_properties['pixels'][n],
                                                        self.branch_properties['length'][n])):

                    # Only include the branches with length > min length
                    if length < (min_branch_length * self.imgscale):
                        continue

                    ymax = pix[:, 0].max()
                    ymin = pix[:, 0].min()
                    xmax = pix[:, 1].max()
                    xmin = pix[:, 1].min()

                    shape = (ymax - ymin + 1 + 2 * radius,
                             xmax - xmin + 1 + 2 * radius)

                    branch_array = np.zeros(shape, dtype=bool)
                    branch_array[pix[:, 0] - ymin + radius,
                                 pix[:, 1] - xmin + radius] = True

                    branch_array = np.fliplr(branch_array)

                    theta, R, quantiles = \
                        rht(branch_array,
                            radius, ntheta, background_percentile)

                    twofive, mean, sevenfive = quantiles

                    means = np.append(means, mean)
                    if sevenfive > twofive:
                        iqrs = \
                            np.append(iqrs,
                                      np.abs(sevenfive - twofive))
                    else:
                        iqrs = \
                            np.append(iqrs,
                                      np.abs(sevenfive - twofive) + np.pi)
                    intensity = \
                        np.append(intensity,
                                  self.branch_properties["intensity"][0][val - 1])
                    lengths = np.append(lengths,
                                        self.branch_properties["length"][0][val - 1])

                self.rht_curvature["Orientation"].append(means)
                self.rht_curvature["Curvature"].append(iqrs)
                self.rht_curvature["Intensity"].append(intensity)
                self.rht_curvature["Length"].append(lengths)

                if verbose or save_png:
                    Warning("No verbose mode available when running RHT on "
                            "individual branches. No plots will be saved.")

            else:
                skel_arr = np.fliplr(self.filament_arrays["long path"][n])
                theta, R, quantiles = rht(
                    skel_arr, radius, ntheta, background_percentile)

                twofive, median, sevenfive = quantiles

                self.rht_curvature["Orientation"].append(median)
                if sevenfive > twofive:
                    self.rht_curvature["Curvature"].append(
                        np.abs(sevenfive - twofive))  # Interquartile range
                else:  #
                    self.rht_curvature["Curvature"].append(
                        np.abs(sevenfive - twofive + np.pi))

                if verbose or save_png:
                    p.clf()
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
                    if save_png:
                        try_mkdir(self.save_name)
                        p.savefig(os.path.join(self.save_name,
                                               self.save_name + "_rht_" + str(n) + ".png"))
                    if verbose:
                        p.show()
                    if in_ipynb():
                        p.clf()

    def find_widths(self, fit_model=gauss_model, try_nonparam=True,
                    use_longest_paths=False, verbose=False, save_png=False,
                    **kwargs):
        '''

        The final step of the algorithm is to find the widths of each
        of the skeletons. We do this by:

        *   A Euclidean Distance Transform is performed on each skeleton.
            The skeletons are also recombined onto a single array. The
            individual filament arrays are padded to ensure a proper radial
            profile is created. If the padded arrays fall outside of the
            original image, they are trimmed.
        *   A user-specified model is fit to each of the radial profiles.
            The default is a Gaussian with a constant background. This returns
            the width and central
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
        use_longest_paths : bool, optional
            Optionally use the longest path skeletons for the width fitting.
            Note that this will disregard all branches off of the longest
            path.
        verbose : bool, optional
            Enables plotting.
        save_png : bool, optional
            Saves the plot made in verbose mode. Disabled by default.

        Attributes
        ----------
        width_fits : dict
            Contains the fit parameters and estimations of the errors
            from each fit.
        total_intensity : list
            Sum of the intensity in each filament within 1 FWHM of the
            skeleton.

        '''

        warnings.warn("An array offset issue is present in the radial profiles"
                      "! Please use the new version in FilFinder2D. "
                      "Double-check all results from this function!")

        if use_longest_paths:
            skel_arrays = self.filament_arrays["long path"]
        else:
            skel_arrays = self.filament_arrays["final"]

        dist_transform_all, dist_transform_separate = \
            dist_transform(skel_arrays,
                           self.skeleton)

        # Prepare the storage
        self.width_fits["Parameters"] = np.empty((self.number_of_filaments, 4))
        self.width_fits["Errors"] = np.empty((self.number_of_filaments, 4))
        self.width_fits["Type"] = np.empty((self.number_of_filaments),
                                           dtype="S")
        self.total_intensity = np.empty((self.number_of_filaments, ))

        self._rad_profiles = []
        self._unbin_rad_profiles = []

        for n in range(self.number_of_filaments):

            # Shift bottom offset by 1. There's a +1 running around somewhere
            # in the old code that isn't in the new code. Just make the
            # correction here.
            low_corner = list(self.array_offsets[n][0])
            low_corner[0] -= 1
            low_corner[1] -= 1
            offsets = (tuple(low_corner), self.array_offsets[n][1])

            # Need the unbinned data for the non-parametric fit.
            out = radial_profile(self.image, dist_transform_all,
                                 dist_transform_separate[n],
                                 offsets, self.imgscale,
                                 **kwargs)

            if out is not None:
                dist, radprof, weights, unbin_dist, unbin_radprof = out
                self._rad_profiles.append([dist, radprof])
                self._unbin_rad_profiles.append([unbin_dist, unbin_radprof])
            else:
                self.total_intensity[n] = np.NaN
                self.width_fits["Parameters"][n, :] = [np.NaN] * 4
                self.width_fits["Errors"][n, :] = [np.NaN] * 4
                self.width_fits["Type"][n] = 'g'
                self._rad_profiles.append([np.NaN, np.NaN])
                continue

            # if fit_model == cyl_model:
            #     if self.freq is None:
            #         print('''Image not converted to column density.
            #                  Fit parameters will not match physical meaning.
            #                  lease specify frequency.''')
            #     else:
            #         assert isinstance(self.freq, float)
            #         radprof = dens_func(
            #             planck(20., self.freq), 0.2, radprof) * (5.7e19)

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

            if verbose or save_png:
                if verbose:
                    print("%s in %s" % (n, self.number_of_filaments))
                    print("Fit Parameters: %s " % (fit))
                    print("Fit Errors: %s" % (fit_error))
                    print("Fit Type: %s" % (fit_type))

                p.clf()
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

                xlow, ylow = (self.array_offsets[n][0][0],
                              self.array_offsets[n][0][1])
                xhigh, yhigh = (self.array_offsets[n][1][0],
                                self.array_offsets[n][1][1])
                shape = (xhigh - xlow, yhigh - ylow)

                p.contour(skel_arrays[n]
                          [self.pad_size:shape[0] - self.pad_size,
                           self.pad_size:shape[1] - self.pad_size], colors="r")

                img_slice = self.image[xlow + self.pad_size:
                                       xhigh - self.pad_size,
                                       ylow + self.pad_size:
                                       yhigh - self.pad_size]

                # Use an asinh stretch to highlight all features
                from astropy.visualization import AsinhStretch
                from astropy.visualization.mpl_normalize import ImageNormalize

                vmin = np.nanmin(img_slice)
                vmax = np.nanmax(img_slice)
                p.imshow(img_slice, cmap='binary', origin='lower',
                         norm=ImageNormalize(vmin=vmin, vmax=vmax,
                                             stretch=AsinhStretch()))
                cbar = p.colorbar()
                cbar.set_label(r'Intensity')

                if save_png:
                    try_mkdir(self.save_name)
                    filename = \
                        "{0}_width_fit_{1}.png".format(self.save_name, n)
                    p.savefig(os.path.join(self.save_name, filename))
                if verbose:
                    p.show()
                if in_ipynb():
                    p.clf()

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

    def compute_filament_brightness(self):
        '''
        Returns the median brightness along the skeleton of the filament.

        Attributes
        ----------
        filament_brightness : list
            Average brightness/intensity over the skeleton pixels
            for each filament.
        '''

        self.filament_brightness = []

        labels, n = nd.label(self.skeleton, eight_con())

        for n in range(1, self.number_of_filaments+1):
            values = self.image[np.where(labels == n)]
            self.filament_brightness.append(np.median(values))

    def filament_model(self, max_radius=25, use_nopad=True):
        '''
        Returns a model of the diffuse filamentary network based
        on the radial profiles.

        Parameters
        ----------
        max_radius : int, optional
            Number of pixels to extend profiles to.
        use_nopad : bool, optional
            Returns the unpadded image size when enabled. Enabled by
            default.

        Returns
        -------
        model_image : numpy.ndarray
            Array of the model

        '''

        if len(self.width_fits['Parameters']) == 0:
            raise TypeError("Run profile fitting first!")

        params = self.width_fits['Parameters']
        scale = self.imgscale

        if use_nopad:
            skel_array = self.skeleton_nopad
        else:
            skel_array = self.skeleton

        # Create the distance transforms
        all_fils = dist_transform(self.filament_arrays["final"],
                                  skel_array)[0]

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
            Passed to `~fil_finder_2D.filament_model`

        Attributes
        ----------
        covering_fraction : float
            Fraction of the total image intensity contained in the
            filamentary structure (based on the local, individual fits)
        '''

        fil_model = self.filament_model(max_radius=max_radius)

        self.covering_fraction = np.nansum(fil_model) / np.nansum(self.image)

    def save_table(self, table_type="csv", path=None, save_name=None,
                   save_branch_props=True, branch_table_type="hdf5",
                   hdf5_path="data"):
        '''

        The results of the algorithm are saved as an Astropy table in a 'csv',
        'fits' or latex format.

        Parameters
        ----------

        table_type : str, optional
            Sets the output type of the table. Supported options are
            "csv", "fits", "latex" and "hdf5".
        path : str, optional
            The path where the file should be saved.
        save_name : str, optional
            The prefix for the saved file. If None, the save name specified
            when ``fil_finder_2D`` was first called.
        save_branch_props : bool, optional
            When enabled, saves the lists of branch lengths and intensities
            in a separate file(s). Default is enabled.
        branch_table_type : str, optional
            Any of the accepted table_types will work here. If using HDF5,
            just one output file is created with each stored within it.
        hdf5_path : str, optional
            Path name within the HDF5 file.

        Attributes
        ----------
        dataframe : astropy.Table
            The dataframe is returned for use with the ``Analysis`` class.

        '''

        if save_name is None:
            save_name = self.save_name

        save_name = save_name + "_table"

        if table_type == "csv":
            filename = save_name + ".csv"
        elif table_type == "fits":
            filename = save_name + ".fits"
        elif table_type == "latex":
            filename = save_name + ".tex"
        elif table_type == "hdf5":
            filename = save_name + ".hdf5"
        else:
            raise NameError("Only formats supported are 'csv', 'fits' \
                           and 'latex'.")

        # If path is specified, append onto filename.
        if path is not None:
            filename = os.path.join(path, filename)

        if not self._rht_branches_flag:
            data = {"Lengths": self.lengths,
                    "Orientation": self.rht_curvature["Orientation"],
                    "Curvature": self.rht_curvature["Curvature"],
                    "Branches": self.branch_properties["number"],
                    "Fit Type": self.width_fits["Type"],
                    "Total Intensity": self.total_intensity,
                    "Median Brightness": self.filament_brightness}

            branch_data = \
                {"Branch Length": self.branch_properties["length"],
                 "Branch Intensity": self.branch_properties["intensity"]}
        else:
            # RHT was ran on branches, and so can only be saved as a branch
            # property due to the table shape

            data = {"Lengths": self.lengths,
                    "Fit Type": self.width_fits["Type"],
                    "Total Intensity": self.total_intensity,
                    "Branches": self.branch_properties["number"],
                    "Median Brightness": self.filament_brightness}

            branch_data = \
                {"Branch Length": self.rht_curvature["Length"],
                 "Branch Intensity": self.rht_curvature["Intensity"],
                 "Curvature": self.rht_curvature["Curvature"],
                 "Orientation": self.rht_curvature["Orientation"]}

        for i, param in enumerate(self.width_fits["Names"]):
            data[param] = self.width_fits["Parameters"][:, i]
            data[param + " Error"] = self.width_fits["Errors"][:, i]

        try_mkdir(self.save_name)

        df = Table(data)

        if table_type == "csv":
            df.write(os.path.join(self.save_name, filename),
                     format="ascii.csv")
        elif table_type == "fits":
            df.write(os.path.join(self.save_name, filename))
        elif table_type == "latex":
            df.write(os.path.join(self.save_name, filename),
                     format="ascii.latex")
        elif table_type == 'hdf5':
            df.write(os.path.join(self.save_name, filename),
                     path=hdf5_path)

        self.dataframe = df

        for n in range(self.number_of_filaments):

            branch_df = \
                Table([branch_data[key][n] for key in branch_data.keys()],
                      names=branch_data.keys())

            branch_filename = save_name + "_branch_" + str(n)

            if branch_table_type == "csv":
                branch_df.write(os.path.join(self.save_name,
                                             branch_filename+".csv"),
                                format="ascii.csv")
            elif branch_table_type == "fits":
                branch_df.write(os.path.join(self.save_name,
                                             branch_filename+".fits"))
            elif branch_table_type == "latex":
                branch_df.write(os.path.join(self.save_name,
                                             branch_filename+".tex"),
                                format="ascii.latex")
            elif branch_table_type == 'hdf5':
                hdf_filename = save_name + "_branch"
                if n == 0:
                    branch_df.write(os.path.join(self.save_name,
                                                 hdf_filename+".hdf5"),
                                    path="branch_"+str(n))
                else:
                    branch_df.write(os.path.join(self.save_name,
                                                 hdf_filename+".hdf5"),
                                    path="branch_"+str(n),
                                    append=True)
    @property
    def mask_nopad(self):
        if self.pad_size == 0:
            return self.mask
        return self.mask[self.pad_size:-self.pad_size,
                         self.pad_size:-self.pad_size]

    @property
    def skeleton_nopad(self):
        if self.pad_size == 0:
            return self.skeleton
        return self.skeleton[self.pad_size:-self.pad_size,
                             self.pad_size:-self.pad_size]

    @property
    def skeleton_longpath_nopad(self):
        if self.pad_size == 0:
            return self.skeleton_longpath
        return self.skeleton_longpath[self.pad_size:-self.pad_size,
                                      self.pad_size:-self.pad_size]

    @property
    def flat_img_nopad(self):
        if self.pad_size == 0:
            self.flat_img
        return self.flat_img[self.pad_size:-self.pad_size,
                             self.pad_size:-self.pad_size]

    @property
    def image_nopad(self):
        if self.pad_size == 0:
            return self.image
        return self.image[self.pad_size:-self.pad_size,
                          self.pad_size:-self.pad_size]

    @property
    def medial_axis_distance_nopad(self):
        if self.pad_size == 0:
            return self.medial_axis_distance
        return self.medial_axis_distance[self.pad_size:-self.pad_size,
                                         self.pad_size:-self.pad_size]

    def save_fits(self, save_name=None, stamps=False, filename=None,
                  model_save=True):
        '''

        This function saves the mask and the skeleton array as FITS files.
        Included in the header are the setting used to create them.

        Parameters
        ----------
        save_name : str, optional
            The prefix for the saved file. If None, the save name specified
            when `~fil_finder_2D` was first called.
        stamps : bool, optional
            Enables saving of individual stamps
        filename : str, optional
            File name of the image used. If None, assumes save_name is the
            file name.
        model_save : bool, optional
            When enabled, calculates the model using `~fil_finder_2D.filament_model`
            and saves it in a FITS file.

        '''

        if save_name is None:
            save_name = self.save_name

        if not filename:  # Assume save_name is filename if not specified.
            filename = save_name

        # Create header based off of image header.
        if self.header is not None:
            new_hdr = deepcopy(self.header)
        else:
            new_hdr = fits.Header()
            new_hdr["NAXIS"] = 2
            new_hdr["NAXIS1"] = self.mask_nopad.shape[1]
            new_hdr["NAXIS2"] = self.mask_nopad.shape[0]

        try:  # delete the original history
            del new_hdr["HISTORY"]
        except KeyError:
            pass

        new_hdr["BUNIT"] = ("bool", "")

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

        try_mkdir(self.save_name)

        # Save mask
        fits.writeto(os.path.join(self.save_name,
                                  "".join([save_name, "_mask.fits"])),
                     self.mask_nopad.astype(">i2"), new_hdr)

        # Save skeletons. Includes final skeletons and the longest paths.
        new_hdr["BUNIT"] = ("int", "")

        new_hdr["COMMENT"] = "Skeleton Size Threshold: " + \
            str(self.skel_thresh)
        new_hdr["COMMENT"] = "Branch Size Threshold: " + \
            str(self.branch_thresh)

        hdu_skel = fits.HDUList()

        # Final Skeletons - create labels which match up with table output

        labels = nd.label(self.skeleton_nopad, eight_con())[0]
        hdu_skel.append(fits.PrimaryHDU(labels.astype(">i2"), header=new_hdr))

        # Longest Paths
        labels_lp = nd.label(self.skeleton_longpath_nopad, eight_con())[0]
        hdu_skel.append(fits.PrimaryHDU(labels_lp.astype(">i2"),
                                        header=new_hdr))

        try_mkdir(self.save_name)

        hdu_skel.writeto(os.path.join(self.save_name,
                                      "".join([save_name, "_skeletons.fits"])))

        if stamps:
            # Save stamps of all images. Include portion of image and the
            # skeleton for reference.

            try_mkdir(self.save_name)

            # Make a directory for the stamps
            out_dir = \
                os.path.join(self.save_name, "stamps_" + save_name)

            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

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
                if self.header is not None:
                    prim_hdr = deepcopy(self.header)
                else:
                    prim_hdr = fits.Header()
                    prim_hdr["NAXIS"] = 2
                    prim_hdr["NAXIS1"] = img_stamp.shape[1]
                    prim_hdr["NAXIS2"] = img_stamp.shape[0]

                prim_hdr["COMMENT"] = "Outputted from fil_finder."
                prim_hdr["COMMENT"] = \
                    "Extent in original array: (" + \
                    str(xlow + self.pad_size) + "," + \
                    str(ylow + self.pad_size) + ")->" + \
                    "(" + str(xhigh - self.pad_size) + \
                    "," + str(yhigh - self.pad_size) + ")"

                hdu = fits.HDUList()
                # Image stamp
                hdu.append(fits.PrimaryHDU(img_stamp.astype(">f4"),
                           header=prim_hdr))
                # Stamp of final skeleton
                try:
                    prim_hdr.update("BUNIT", value="bool", comment="")
                except KeyError:
                    prim_hdr["BUNIT"] = ("int", "")

                hdu.append(fits.PrimaryHDU(skel_arr.astype(">i2"),
                           header=prim_hdr))
                # Stamp of longest path
                hdu.append(fits.PrimaryHDU(lp_arr.astype(">i2"),
                           header=prim_hdr))

                hdu.writeto(os.path.join(out_dir,
                                         save_name+"_object_"+str(n+1)+".fits"))

        if model_save:
            model = self.filament_model(use_nopad=True)

            model_hdr = new_hdr.copy()

            try:
                model_hdr.update("BUNIT", value=self.header['BUNIT'], comment="")
            except KeyError:
                Warning("No BUNIT specified in original header.")

            model_hdu = fits.PrimaryHDU(model.astype(">f4"), header=model_hdr)

            try_mkdir(self.save_name)
            model_hdu.writeto(
                os.path.join(self.save_name,
                             "".join([save_name, "_filament_model.fits"])))

    def __str__(self):
        print("%s filaments found.") % (self.number_of_filaments)
        for fil in range(self.number_of_filaments):
            print("Filament: %s, Width: %s, Length: %s, Curvature: %s,\
                       Orientation: %s" % \
                (fil, self.width_fits["Parameters"][fil, -1][fil],
                 self.lengths[fil], self.rht_curvature["Std"][fil],
                 self.rht_curvature["Std"][fil]))

    def run(self, verbose=False, save_name=None, save_png=False,
            table_type="fits"):
        '''
        The whole algorithm in one easy step. Individual parameters have not
        been included in this batch run. If fine-tuning is needed, it is
        recommended to run each step individually.

        Parameters
        ----------
        verbose : bool, optional
            Enables the verbose option for each of the steps.
        save_name : str, optional
            The prefix for the saved file.
            If None, the name from the header is used.
        save_png : bool, optional
            Saves the plot made in verbose mode. Disabled by default.
        table_type : str, optional
            Sets the output type of the table. Supported options are
            "csv", "fits" and "latex".

        '''

        if save_name is None:
                save_name = self.save_name

        self.create_mask(verbose=verbose, save_png=save_png)
        self.medskel(verbose=verbose, save_png=save_png)

        self.analyze_skeletons(verbose=verbose, save_png=save_png)
        self.exec_rht(verbose=verbose, save_png=save_png)
        self.find_widths(verbose=verbose, save_png=save_png)
        self.compute_filament_brightness()
        self.find_covering_fraction()
        self.save_table(save_name=save_name, table_type=table_type)
        self.save_fits(save_name=save_name, stamps=False)

        if verbose:
            self.__str__()
