# Licensed under an MIT open source license - see LICENSE

import numpy as np
import matplotlib.pyplot as p
import scipy.ndimage as nd
from scipy.stats import lognorm
from scipy.ndimage import distance_transform_edt
from skimage.filters import threshold_adaptive
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
from .base_conversions import (BaseInfoMixin, UnitConverter,
                               find_beam_properties, data_unit_check)
from .filament import Filament2D

# The try/except is here to deal with TypeErrors when building the docs on RTD
# This isn't really a solution... but it is lazy and does the job until I
# add astropy_helpers.
try:
    FWHM_FACTOR = 2 * np.sqrt(2 * np.log(2.))
except TypeError:
    FWHM_FACTOR = np.NaN


class FilFinder2D(BaseInfoMixin):

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
    distance : float, optional
        The distance to the region being examined (in pc). If None, the
        analysis is carried out in pixel and angular units. In this case,
        the physical priors used in other optional parameters is meaningless
        and each must be specified initially.
    mask : numpy.ndarray, optional
        A pre-made, boolean mask may be supplied to skip the segmentation
        process. The algorithm will skeletonize and run the analysis portions
        only.
    save_name : str, optional
        Sets the prefix name that is used for output files. Can be overridden
        in ``save_fits`` and ``save_table``. Default is "FilFinder_output".

    Examples
    --------
    >>> from fil_finder import fil_finder_2D
    >>> from astropy.io import fits
    >>> import astropy.units as u
    >>> hdu = fits.open("twod.fits")[0] # doctest: +SKIP
    >>> filfind = FilFinder2D(hdu, beamwidth=15*u.arcsec, distance=170*u.pc, save_name='twod_filaments') # doctest: +SKIP
    >>> filfind.run(verbose=False) # doctest: +SKIP

    """

    def __init__(self, image, header=None, beamwidth=None, ang_scale=None,
                 distance=None, mask=None, save_name="FilFinder_output"):

        # Accepts a numpy array or fits.PrimaryHDU
        output = input_data(image, header)

        self._image = output["data"]
        if "header" in output:
            self._header = output["header"]
        elif ang_scale is not None:
                if not isinstance(ang_scale, u.Quantity):
                    raise TypeError("ang_scale must be an "
                                    "astropy.units.Quantity.")
                if not ang_scale.unit.is_equivalent(u.deg):
                    raise u.UnitsError("ang_scale must be given in angular "
                                       "units.")

                # Mock up a simple header
                hdr_dict = {"NAXIS": 2,
                            "NAXIS1": self.image.shape[1],
                            "NAXIS2": self.image.shape[0],
                            "CDELT1": ang_scale.to(u.deg),
                            "CDELT2": ang_scale.to(u.deg),
                            }
                self._header = fits.Header(hdr_dict)
        else:
            self._header = None

        if self.header is not None:
            self._wcs = WCS(self.header)
        else:
            self._wcs = None

        self.converter = UnitConverter(self.wcs, distance)

        if beamwidth is None:
            if self.header is not None:
                major = find_beam_properties(self.header)[0]
        else:
            major = beamwidth

        self._beamwidth = self.converter.to_pixel(major)

        self.save_name = save_name

        # If pre-made mask is provided, remove nans if any.
        self.mask = None
        if mask is not None:
            if self.image.shape != mask.shape:
                raise ValueError("The given pre-existing mask must have the "
                                 "same shape as the image.")
            mask[np.isnan(mask)] = 0.0
            self.mask = mask

        # beam width was defined to be the Gaussian width, NOT the FWHM
        # XXX imgscale is the pixel scale of one pixel in pc
        # ang_scale is the pixel area
        self.imgscale = self.converter.physical_size
        self.angular_scale = (self.converter.ang_size**2).to(u.sr)

        self.filament_arrays = {}

    def preprocess_image(self, flatten_percent=None):
        '''
        Preprocess and flatten the image before running the masking routine.

        Parameters
        ----------
        flatten_percent : int, optional
            The percentile of the data (0-100) to set the normalization of the
            arctan transform. By default, a log-normal distribution is fit and
            the threshold is set to :math:`\mu + 2\sigma`. If the data contains
            regions of a much higher intensity than the mean, it is recommended
            this be set >95 percentile.

        '''
        # Make flattened image
        if flatten_percent is None:
            # Fit to a log-normal
            fit_vals = lognorm.fit(self.image[~np.isnan(self.image)])

            median = lognorm.median(*fit_vals)
            std = lognorm.std(*fit_vals)
            thresh_val = median + 2 * std
        else:
            thresh_val = np.percentile(self.image[~np.isnan(self.image)],
                                       flatten_percent)

        self._flatten_threshold = data_unit_check(thresh_val, self.image.unit)

        # Make the units dimensionless
        self.flat_img = thresh_val * \
            np.arctan(self.image / self.flatten_threshold) / u.rad

    @property
    def flatten_threshold(self):
        '''
        Threshold value used in the arctan transform.
        '''
        return self._flatten_threshold

    def create_mask(self, glob_thresh=None, adapt_thresh=None,
                    smooth_size=None, size_thresh=None, verbose=False,
                    test_mode=False, regrid=True, border_masking=True,
                    border_kwargs={'size': 50 * u.pix**2,
                                   'filt_width': 25 * u.pix, 'eros_iter': 15},
                    fill_hole_size=None,
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
            warnings.warn("Using inputted mask. Skipping creation of a"
                          "new mask.")
            # Skip if pre-made mask given
            return

        if not hasattr(self.converter, 'distance'):
            if smooth_size is None:
                raise ValueError("Distance not given. Must specify smooth_size"
                                 " in pixel units.")
            if adapt_thresh is None:
                raise ValueError("Distance not given. Must specify"
                                 "adapt_thresh in pixel units.")
            if size_thresh is None:
                raise ValueError("Distance not given. Must specify size_thresh"
                                 " in pixel units.")

        if glob_thresh is None:
            self.glob_thresh = None
        else:
            self.glob_thresh = data_unit_check(glob_thresh, self.image.unit)

        if size_thresh is None:
            # Adapt a typical filament area as pi * length * width,
            # width width ~ 0.1 pc, and length = 5 * width
            min_fil_area = \
                self.converter.to_pixel_area(np.pi * 5 * 0.1**2 * u.pc**2)
            # Use a threshold rounded to the nearest pixel
            self.size_thresh = round(min_fil_area.value) * u.pix**2
        else:
            self.size_thresh = self.converter.to_pixel_area(size_thresh)

            # Area of ellipse for typical filament size. Divided by 10 to
            # incorporate sparsity.
        if adapt_thresh is None:
            # twice average FWHM for filaments
            fil_width = self.converter.to_pixel(0.2 * u.pc)
            self.adapt_thresh = round(fil_width.value) * u.pix
        else:
            self.adapt_thresh = self.converter.to_pixel(adapt_thresh)

        if smooth_size is None:
            # half average FWHM for filaments
            smooth_width = self.converter.to_pixel(0.05 * u.pc)
            self.smooth_size = round(smooth_width.value) * u.pix
        else:
            self.smooth_size = self.converter.to_pixel(smooth_size)

        # Check if regridding is even necessary
        if self.adapt_thresh >= 40 * u.pix and regrid:
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

            # Convert the size and width to pixel units
            border_size_pix = \
                self.converter.to_pixel_area(border_kwargs['size'])
            border_med_width = \
                np.ceil(self.converter.to_pixel(border_kwargs['filt_width']))

            nan_mask = remove_small_objects(nan_mask,
                                            min_size=border_size_pix.value,
                                            connectivity=8)
            nan_mask = np.logical_not(nan_mask)

            nan_mask = nd.median_filter(nan_mask, int(border_med_width.value))
            nan_mask = nd.binary_erosion(nan_mask, eight_con(),
                                         iterations=border_kwargs['eros_iter'])
        else:
            nan_mask = np.logical_not(np.isnan(flat_copy))

        # Remove nans in the copy
        flat_copy[np.isnan(flat_copy)] = 0.0

        # Perform regridding
        if regrid:
            # Calculate the needed zoom to make the patch size ~40 pixels
            ratio = 40 / self.adapt_thresh.value
            # Round to the nearest factor of 2
            regrid_factor = np.min([2., int(round(ratio / 2.0) * 2.0)])

            # Defaults to cubic interpolation
            masking_img = nd.zoom(flat_copy, (regrid_factor, regrid_factor))
        else:
            regrid_factor = 1
            ratio = 1
            masking_img = flat_copy

        med_filter_size = int(round(self.smooth_size.value * ratio))
        smooth_img = nd.median_filter(masking_img,
                                      size=med_filter_size)

        adapt = threshold_adaptive(smooth_img,
                                   round_to_odd(ratio *
                                                self.adapt_thresh.value),
                                   method="mean")

        if regrid:
            regrid_factor = float(regrid_factor)
            adapt = nd.zoom(adapt, (1 / regrid_factor, 1 / regrid_factor),
                            order=0)

        # Remove areas near the image border
        adapt = adapt * nan_mask

        if self.glob_thresh is not None:
            glob = self.image > self.glob_thresh
            adapt = glob * adapt

        cleaned = remove_small_objects(adapt, min_size=self.size_thresh.value)

        # Remove small holes within the object

        if fill_hole_size is None:
            fill_hole_size = np.pi * (self.beamwidth / FWHM_FACTOR)**2
        else:
            fill_hole_size = self.converter.to_pixel_area(fill_hole_size)

        mask_objs, num, corners = \
            isolateregions(cleaned, fill_hole=True,
                           rel_size=fill_hole_size.value,
                           morph_smooth=True, pad_size=1)

        self.mask = recombine_skeletons(mask_objs,
                                        corners, self.image.shape, 1)

        # WARNING!! Setting some image values to 0 to avoid negative weights.
        # This may cause issues, however it will allow for proper skeletons
        # Through all the testing and deriving science results, this has not
        # been an issue! EK
        # XXX Check this
        # self.image[np.where((self.mask * self.image) < 0.0)] = 0

        if test_mode:
            fig, ax = p.subplots(3, 2, sharex=True, sharey=True)
            im0 = ax[0, 0].imshow(np.log10(self.image.value), origin="lower",
                                  interpolation='nearest',
                                  cmap='binary')
            fig.colorbar(im0, ax=ax[0, 0])

            im1 = ax[1, 0].imshow(masking_img, origin="lower",
                                  interpolation='nearest',
                                  cmap='binary')
            fig.colorbar(im1, ax=ax[1, 0])

            im2 = ax[0, 1].imshow(smooth_img, origin="lower",
                                  interpolation='nearest',
                                  cmap='binary')
            fig.colorbar(im2, ax=ax[0, 1])

            im3 = ax[1, 1].imshow(adapt, origin="lower",
                                  interpolation='nearest',
                                  cmap='binary')
            fig.colorbar(im3, ax=ax[1, 1])

            im4 = ax[2, 0].imshow(cleaned, origin="lower",
                                  interpolation='nearest',
                                  cmap='binary')
            fig.colorbar(im4, ax=ax[2, 0])

            im5 = ax[2, 1].imshow(self.mask, origin="lower",
                                  interpolation='nearest',
                                  cmap='binary')
            fig.colorbar(im5, ax=ax[2, 1])

            p.show()

        if verbose or save_png:
            vmin = np.percentile(self.flat_img[np.isfinite(self.flat_img)], 20)
            vmax = np.percentile(self.flat_img[np.isfinite(self.flat_img)], 90)
            p.clf()
            p.imshow(self.flat_img, interpolation='nearest', origin="lower",
                     cmap='binary', vmin=vmin, vmax=vmax)
            p.contour(self.mask, colors="r")
            p.title("Mask on Flattened Image.")
            if save_png:
                try_mkdir(self.save_name)
                p.savefig(os.path.join(self.save_name,
                                       self.save_name + "_mask.png"))
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
            self.medial_axis_distance * self.skeleton * u.pix
        # Delete connection smaller than 2 pixels wide. Such a small
        # connection is more likely to be from limited pixel resolution
        # rather than actual structure.
        width_threshold = 1 * u.pix
        narrow_pts = np.where(self.medial_axis_distance < width_threshold)
        self.skeleton[narrow_pts] = 0  # Eliminate narrow connections
        self.medial_axis_distance[narrow_pts] = 0 * u.pix

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

    def analyze_skeletons(self, relintens_thresh=0.2, nbeam_lengths=5,
                          branch_nbeam_lengths=3,
                          skel_thresh=None, branch_thresh=None,
                          verbose=False, save_png=False, save_name=None):
        '''

        This function wraps most of the skeleton analysis. Several steps are
        completed here:

        *   isolatefilaments is run to separate each skeleton into its own
            array. If the skeletons are under the threshold set by
            self.size_thresh, the region is removed. An updated mask is
            also returned.
        *   pix_identify classifies each of the pixels in a skeleton as a
            body, end, or intersection point. See the documentation on
            find_filpix for a complete explanation. The function labels the
            branches and intersections of each skeletons.
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
            raise ValueError("relintens_thresh must be set between "
                             "(0.0, 1.0].")

        if not hasattr(self.converter, 'distance'):
            if self.skel_thresh is None and skel_thresh is None:
                raise ValueError("Distance not given. Must specify skel_thresh"
                                 " in pixel units.")

        # Set the skeleton length threshold to some factor of the beam width
        if skel_thresh is None:
            # Double check these defaults.
            # min_length = self.converter.to_pixel(0.3 * u.pc)
            min_length = nbeam_lengths * self.beamwidth
            skel_thresh = round(min_length.value) * u.pix
        else:
            skel_thresh = self.converter.to_pixel(skel_thresh)

        self.skel_thresh = np.ceil(skel_thresh)

        # Set the minimum branch length to be the beam size.
        if branch_thresh is None:
            branch_thresh = branch_nbeam_lengths * self.beamwidth
        else:
            branch_thresh = self.converter.to_pixel(branch_thresh)

        self.branch_thresh = np.ceil(branch_thresh).astype(int)

        # Label individual filaments and define the set of filament objects
        labels, num = nd.label(self.skeleton, eight_con())

        # Find the objects that don't satisfy skel_thresh
        if self.skel_thresh > 0.:
            obj_sums = nd.sum(self.skeleton, labels, range(1, num + 1))
            remove_fils = np.where(obj_sums <= self.skel_thresh.value)[0]

            for lab in remove_fils:
                self.skeleton[np.where(labels == lab + 1)] = 0

            # Relabel after deleting short skeletons.
            labels, num = nd.label(self.skeleton, eight_con())

        self.filaments = [Filament2D(np.where(labels == lab),
                                     converter=self.converter) for lab in
                          range(1, num + 1)]

        # Now loop over the skeleton analysis for each filament object
        for fil in self.filaments:
            fil.skeleton_analysis(self.image, verbose=verbose,
                                  save_png=save_png, save_name=save_name,
                                  relintens_thresh=relintens_thresh,
                                  branch_thresh=self.branch_thresh)

        self.number_of_filaments = num
        self.array_offsets = [fil.pixel_extents for fil in self.filaments]

        branch_properties = {}
        branch_properties['length'] = [fil.branch_properties['length']
                                       for fil in self.filaments]
        branch_properties['intensity'] = [fil.branch_properties['intensity']
                                          for fil in self.filaments]
        branch_properties['pixels'] = [fil.branch_properties['pixels']
                                       for fil in self.filaments]
        branch_properties['number'] = np.array([fil.branch_properties['number']
                                                for fil in self.filaments])

        self.branch_properties = branch_properties

        self.filament_extents = [fil.pixel_extents for fil in self.filaments]

        # Returns lengths in pixels
        self.lengths = \
            np.array([fil.length().value for fil in self.filaments]) * u.pix

        # Create filament arrays still?
        self.filament_arrays["long path"] = [fil.skeleton(out_type='longpath')
                                             for fil in self.filaments]

        self.filament_arrays["final"] = [fil.skeleton()
                                         for fil in self.filaments]

        # Convert branch lengths physical units
        # for n in range(self.number_of_filaments):
        #     lengths = self.branch_properties["length"][n]
        #     self.branch_properties["length"][n] = \
        #         [self.imgscale * length for length in lengths]

        self.skeleton = \
            recombine_skeletons(self.filament_arrays["final"],
                                self.array_offsets, self.image.shape,
                                0)

        self.skeleton_longpath = \
            recombine_skeletons(self.filament_arrays["long path"],
                                self.array_offsets, self.image.shape,
                                0)

    def lengths(self, unit=u.pix):
        '''
        Return longest path lengths of the filaments
        '''
        return self.converter.from_pixel(self._lengths, unit)

    def branch_lengths(self, unit=u.pix):
        '''
        Return the length of all branches in all filaments.
        '''
        branches = []
        for lengths in self.branch_properties['length']:
            branches.append(self.converter.from_pixel(lengths, unit))

        return branches

    def exec_rht(self, radius=10 * u.pix,
                 ntheta=180, background_percentile=25,
                 branches=False, min_branch_length=3 * u.pix,
                 verbose=False, save_png=False):
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

        # Flag branch output
        self._rht_branches_flag = False
        if branches:
            self._rht_branches_flag = True

        for i, fil in enumerate(self.filaments):

            if branches:
                fil.rht_branch_analysis(radius=radius,
                                        ntheta=ntheta,
                                        background_percentile=background_percentile,
                                        min_branch_length=min_branch_length)

            else:
                fil.rht_analysis(radius=radius, ntheta=ntheta,
                                 background_percentile=background_percentile)

                if verbose:
                    if save_png:
                        save_name = "{0}_rht_{1}.png".format(self.save_name, i)
                    else:
                        save_name = None
                    fil.plot_rht_distrib(save_name=save_name)

    @property
    def orientation(self):
        '''
        Returns the orientations of the filament longest paths computed with
        `~FilFinder2D.exec_rht` with `branches=False`.
        '''
        return [fil.orientation.value for fil in self.filaments] * u.rad

    @property
    def curvature(self):
        '''
        Returns the orientations of the filament longest paths computed with
        `~FilFinder2D.exec_rht` with `branches=False`.
        '''
        return [fil.curvature.value for fil in self.filaments] * u.rad

    @property
    def orientation_branches(self):
        '''
        Returns the orientations of the filament longest paths computed with
        `~FilFinder2D.exec_rht` with `branches=False`.
        '''
        return [fil.orientation_branches for fil in self.filaments]

    @property
    def curvature_branches(self):
        '''
        Returns the orientations of the filament longest paths computed with
        `~FilFinder2D.exec_rht` with `branches=False`.
        '''
        return [fil.curvature_branches for fil in self.filaments]

    def find_widths(self, max_dist=10 * u.pix,
                    fit_model='gaussian_bkg',
                    fitter=None,
                    try_nonparam=True,
                    use_longest_path=False,
                    add_width_to_length=True,
                    deconvolve_width=True,
                    fwhm_function=None,
                    chisq_max=10.,
                    set_fail_to_nan=False,
                    verbose=False, save_png=False,
                    xunit=u.pix,
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

        for i, fil in enumerate(self.filaments):
            fil.width_analysis(self.flat_img, all_skeleton_array=self.skeleton,
                               max_dist=max_dist, fit_model=fit_model,
                               fitter=fitter, try_nonparam=try_nonparam,
                               use_longest_path=use_longest_path,
                               add_width_to_length=add_width_to_length,
                               deconvolve_width=deconvolve_width,
                               beamwidth=self.beamwidth,
                               fwhm_function=fwhm_function,
                               chisq_max=chisq_max,
                               **kwargs)

            if verbose:
                if save_png:
                    save_name = "{0}_radprof_{1}.png".format(self.save_name, i)
                else:
                    save_name = None
                fil.plot_radial_profile(save_name=save_name, xunit=xunit)

        # Set failed fits to NaN if enabled
        if set_fail_to_nan:
            raise NotImplementedError("")

    def widths(self, unit=u.pix):
        '''
        Fitted FWHM of the filaments and their uncertainties.

        Parameters
        ----------
        unit : `~astropy.units.Quantity`, optional
            The output unit for the FWHM. Default is in pixel units.
        '''
        pix_fwhm = np.array([fil.radprof_fwhm()[0].value for fil in
                             self.filaments])
        pix_fwhm_err = np.array([fil.radprof_fwhm()[1].value for fil in
                                 self.filaments])
        return self.converter.from_pixel(pix_fwhm * u.pix, unit), \
            self.converter.from_pixel(pix_fwhm_err * u.pix, unit)

    def width_fits(self, xunit=u.pix):
        '''
        Return an `~astropy.table.Table` of the width fit parameters,
        uncertainties, and whether a flag was raised for a bad fit.
        '''

        from astropy.table import vstack as tab_vstack

        for i, fil in enumerate(self.filaments):
            if i == 0:
                tab = fil.radprof_fit_table(xunit=xunit)
                continue

            add_tab = fil.radprof_fit_table(xunit=xunit)

            # Concatenate the row together
            tab = tab_vstack([tab, add_tab])

        return tab

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

        return self

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

        return self

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

        return self

    def __str__(self):
        print("%s filaments found.") % (self.number_of_filaments)
        for fil in range(self.number_of_filaments):
            print("Filament: %s, Width: %s, Length: %s, Curvature: %s,\
                       Orientation: %s" % \
                (fil, self.width_fits["Parameters"][fil, -1][fil],
                 self.lengths[fil], self.rht_curvature["Std"][fil],
                 self.rht_curvature["Std"][fil]))
