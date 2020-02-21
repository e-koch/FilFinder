# Licensed under an MIT open source license - see LICENSE

import numpy as np
import matplotlib.pyplot as p
import scipy.ndimage as nd
from scipy.stats import lognorm
from skimage.morphology import remove_small_objects, medial_axis
from astropy.io import fits
from astropy.table import Table, Column
from astropy import units as u
from astropy.wcs import WCS
from astropy.nddata.utils import overlap_slices
from copy import deepcopy
import os
import time
import warnings

from .pixel_ident import recombine_skeletons, isolateregions
from .utilities import eight_con, round_to_odd, threshold_local, in_ipynb
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
    Extract and analyze filamentary structure from a 2D image.

    Parameters
    ----------
    image : `~numpy.ndarray` or `~astropy.io.fits.PrimaryHDU`
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
    >>> from fil_finder import FilFinder2D
    >>> from astropy.io import fits
    >>> import astropy.units as u
    >>> hdu = fits.open("twod.fits")[0] # doctest: +SKIP
    >>> filfind = FilFinder2D(hdu, beamwidth=15*u.arcsec, distance=170*u.pc, save_name='twod_filaments') # doctest: +SKIP
    >>> filfind.preprocess_image(verbose=False) # doctest: +SKIP
    >>> filfind.create_mask(verbose=False) # doctest: +SKIP
    >>> filfind.medskel(verbose=False) # doctest: +SKIP
    >>> filfind.analyze_skeletons(verbose=False) # doctest: +SKIP
    >>> filfind.exec_rht(verbose=False) # doctest: +SKIP
    >>> filfind.find_widths(verbose=False) # doctest: +SKIP
    >>> fil_table = filfind.output_table(verbose=False) # doctest: +SKIP
    >>> branch_table = filfind.branch_tables(verbose=False) # doctest: +SKIP
    >>> filfind.save_fits() # doctest: +SKIP
    >>> filfind.save_stamp_fits() # doctest: +SKIP
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
                            "CDELT1": - ang_scale.to(u.deg).value,
                            "CDELT2": ang_scale.to(u.deg).value,
                            'CTYPE1': 'GLON-CAR',
                            'CTYPE2': 'GLAT-CAR',
                            'CUNIT1': 'deg',
                            'CUNIT2': 'deg',
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
        else:
            major = beamwidth

        if major is not None:
            self._beamwidth = self.converter.to_pixel(major)
        else:
            warnings.warn("No beam width given. Using 0 pixels.")
            self._beamwidth = 0.0 * u.pix

        self.save_name = save_name

        # If pre-made mask is provided, remove nans if any.
        self.mask = None
        if mask is not None:
            if self.image.shape != mask.shape:
                raise ValueError("The given pre-existing mask must have the "
                                 "same shape as the image.")
            mask[np.isnan(mask)] = 0.0
            self.mask = mask

    def preprocess_image(self, skip_flatten=False, flatten_percent=None):
        '''
        Preprocess and flatten the image before running the masking routine.

        Parameters
        ----------
        skip_flatten : bool, optional
            Skip the flattening step and use the original image to construct
            the mask. Default is False.
        flatten_percent : int, optional
            The percentile of the data (0-100) to set the normalization of the
            arctan transform. By default, a log-normal distribution is fit and
            the threshold is set to :math:`\mu + 2\sigma`. If the data contains
            regions of a much higher intensity than the mean, it is recommended
            this be set >95 percentile.

        '''
        if skip_flatten:
            self._flatten_threshold = None
            self.flat_img = self.image
        else:

            # Make flattened image
            if flatten_percent is None:
                # Fit to a log-normal
                fit_vals = lognorm.fit(self.image[~np.isnan(self.image)].value)

                median = lognorm.median(*fit_vals)
                std = lognorm.std(*fit_vals)
                thresh_val = median + 2 * std
            else:
                thresh_val = np.percentile(self.image[~np.isnan(self.image)],
                                           flatten_percent)

            self._flatten_threshold = data_unit_check(thresh_val,
                                                      self.image.unit)

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
            self.glob_thresh = 'usermask'
            self.adapt_thresh = 'usermask'
            self.size_thresh = 'usermask'
            self.smooth_size = 'usermask'

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

        adapt = threshold_local(smooth_img,
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
            p.imshow(self.flat_img.value, interpolation='nearest',
                     origin="lower", cmap='binary', vmin=vmin, vmax=vmax)
            p.contour(self.mask, colors="r")
            p.title("Mask on Flattened Image.")
            if save_png:
                p.savefig(self.save_name + "_mask.png")
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
            p.imshow(self.flat_img.value, interpolation=None, origin="lower",
                     cmap='binary', vmin=vmin, vmax=vmax)
            p.contour(self.skeleton, colors="r")
            if save_png:
                p.savefig(self.save_name + "_initial_skeletons.png")
            if verbose:
                p.show()
            if in_ipynb():
                p.clf()

    def analyze_skeletons(self, prune_criteria='all', relintens_thresh=0.2,
                          nbeam_lengths=5, branch_nbeam_lengths=3,
                          skel_thresh=None, branch_thresh=None,
                          max_prune_iter=10,
                          verbose=False, save_png=False, save_name=None):
        '''

        Prune skeleton structure and calculate the branch and longest-path
        lengths. See `~Filament2D.skeleton_analysis`.


        Parameters
        ----------
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
            Given in pixel units.Below this cut off, skeletons with less pixels
            will be deleted. The default value is 0.3 pc converted to pixels.
        branch_thresh : float, optional
            Any branches shorter than this length (in pixels) will be labeled as
            extraneous and pruned off. The default value is 3 times the FWHM
            beamwidth.
        max_prune_iter : int, optional
            Maximum number of pruning iterations to apply.
        verbose : bool, optional
            Enables plotting.
        save_png : bool, optional
            Saves the plot made in verbose mode. Disabled by default.
        save_name : str, optional
            Prefix for the saved plots.
        '''

        if relintens_thresh > 1.0 or relintens_thresh <= 0.0:
            raise ValueError("relintens_thresh must be set between "
                             "(0.0, 1.0].")

        if not hasattr(self.converter, 'distance') and skel_thresh is None:
            raise ValueError("Distance not given. Must specify skel_thresh"
                             " in pixel units.")

        if save_name is None:
            save_name = self.save_name

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
        for n, fil in enumerate(self.filaments):
            savename = "{0}_{1}".format(save_name, n)
            if verbose:
                print("Filament: %s / %s" % (n + 1, self.number_of_filaments))

            fil.skeleton_analysis(self.image, verbose=verbose,
                                  save_png=save_png,
                                  save_name=savename,
                                  prune_criteria=prune_criteria,
                                  relintens_thresh=relintens_thresh,
                                  branch_thresh=self.branch_thresh,
                                  max_prune_iter=max_prune_iter)

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

        long_path_skel = [fil.skeleton(out_type='longpath')
                          for fil in self.filaments]

        final_skel = [fil.skeleton() for fil in self.filaments]

        self.skeleton = \
            recombine_skeletons(final_skel,
                                self.array_offsets, self.image.shape,
                                0)

        self.skeleton_longpath = \
            recombine_skeletons(long_path_skel,
                                self.array_offsets, self.image.shape,
                                0)

    def lengths(self, unit=u.pix):
        '''
        Return longest path lengths of the filaments.

        Parameters
        ----------
        unit : `~astropy.units.Unit`, optional
            Pixel, angular, or physical unit to convert to.
        '''
        pix_lengths = np.array([fil.length().value
                                for fil in self.filaments]) * u.pix
        return self.converter.from_pixel(pix_lengths, unit)

    def branch_lengths(self, unit=u.pix):
        '''
        Return the length of all branches in all filaments.

        Parameters
        ----------
        unit : `~astropy.units.Unit`, optional
            Pixel, angular, or physical unit to convert to.
        '''
        branches = []
        for lengths in self.branch_properties['length']:
            branches.append(self.converter.from_pixel(lengths, unit))

        return branches

    def filament_positions(self, world_coord=False):
        '''
        Return the median pixel or world positions of the filaments.

        Parameters
        ----------
        world_coord : bool, optional
            Return the world coordinates, defined by the WCS information. If no
            WCS information is given, the output stays in pixel units.

        Returns
        -------
        filament positions : list of tuples
            The median positions of each filament.
        '''
        return [fil.position(world_coord=world_coord) for fil in
                self.filaments]

    @property
    def intersec_pts(self):
        '''
        Intersection pixels for each filament.
        '''
        return [fil.intersec_pts for fil in self.filaments]

    @property
    def end_pts(self):
        '''
        End pixels for each filament.
        '''
        return [fil.end_pts for fil in self.filaments]

    def exec_rht(self, radius=10 * u.pix,
                 ntheta=180, background_percentile=25,
                 branches=False, min_branch_length=3 * u.pix,
                 verbose=False, save_png=False, save_name=None):
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
        save_name : str, optional
            Prefix for the saved plots.

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

        if save_name is None:
            save_name = self.save_name

        for n, fil in enumerate(self.filaments):
            if verbose:
                print("Filament: %s / %s" % (n + 1, self.number_of_filaments))

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
                        savename = "{0}_{1}_rht.png".format(save_name, n)
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
                    pad_to_distance=0 * u.pix,
                    fit_model='gaussian_bkg',
                    fitter=None,
                    try_nonparam=True,
                    use_longest_path=False,
                    add_width_to_length=True,
                    deconvolve_width=True,
                    fwhm_function=None,
                    chisq_max=10.,
                    verbose=False, save_png=False, save_name=None,
                    xunit=u.pix,
                    **kwargs):
        '''

        Create an average radial profile for each filaments and fit a given
        model. See `~Filament2D.width_analysis`.

        *   Radial profiles are created from a Euclidean Distance Transform
            on the skeleton.
        *   A user-specified model is fit to each of the radial profiles.
            The default model is a Gaussian with a constant background
            ('gaussian_bkg'). Other built-in models include a Gaussian with
            no background ('gaussian_nobkg') or a non-parametric estimate
            ('nonparam'). Any 1D astropy model (or compound model) can be
            passed for fitting.

        Parameters
        ----------
        image : `~astropy.unit.Quantity` or `~numpy.ndarray`
            The image from which the filament was extracted.
        all_skeleton_array : np.ndarray
            An array with the skeletons of other filaments. This is used to
            avoid double-counting pixels in the radial profiles in nearby
            filaments.
        max_dist : `~astropy.units.Quantity`, optional
            Largest radius around the skeleton to create the profile from. This
            can be given in physical, angular, or physical units.
        pad_to_distance : `~astropy.units.Quantity`, optional
            Force all pixels within this distance to be kept, even if a pixel
            is closer to another skeleton, as given in `all_skeleton_array`.
        fit_model : str or `~astropy.modeling.Fittable1DModel`, optional
            The model to fit to the profile. Built-in models include
            'gaussian_bkg' for a Gaussian with a constant background,
            'gaussian_nobkg' for just a Gaussian, 'nonparam' for the
            non-parametric estimator. Defaults to 'gaussian_bkg'.
        fitter : `~astropy.modeling.fitting.Fitter`, optional
            One of the astropy fitting classes. Defaults to a
            Levenberg-Marquardt fitter.
        try_nonparam : bool, optional
            If the chosen model fit fails, fall back to a non-parametric
            estimate.
        use_longest_path : bool, optional
            Only fit profile to the longest path skeleton. Disabled by
            default.
        add_width_to_length : bool, optional
            Add the FWHM to the filament length. This accounts for the
            expected shortening in the medial axis transform. Enabled by
            default.
        deconvolve_width : bool, optional
            Deconvolve the beam width from the FWHM. Enabled by default.
        fwhm_function : function, optional
            Convert the width parameter to the FWHM. Must take the fit model
            as an argument and return the FWHM and its uncertainty. If no
            function is given, the Gaussian FWHM is used.
        chisq_max : float, optional
            Enable the fail flag if the reduced chi-squared value is above
            this limit.
        verbose : bool, optional
            Enables plotting.
        save_png : bool, optional
            Saves the plot made in verbose mode. Disabled by default.
        save_name : str, optional
            Prefix for the saved plots.
        xunit : `~astropy.units.Unit`, optional
            Pixel, angular, or physical unit to convert to in the plot.
        kwargs : Passed to `~fil_finder.width.radial_profile`.
        '''
        if save_name is None:
            save_name = self.save_name

        for n, fil in enumerate(self.filaments):

            if verbose:
                print("Filament: %s / %s" % (n + 1, self.number_of_filaments))

            fil.width_analysis(self.image, all_skeleton_array=self.skeleton,
                               max_dist=max_dist,
                               pad_to_distance=pad_to_distance,
                               fit_model=fit_model,
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
                    save_name = "{0}_{1}_radprof.png".format(self.save_name, n)
                else:
                    save_name = None
                fil.plot_radial_profile(save_name=save_name, xunit=xunit)

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

        Parameters
        ----------
        xunit : `~astropy.units.Unit`, optional
            Pixel, angular, or physical unit to convert to.

        Returns
        -------
        tab : `~astropy.table.Table`
            Table with width fit results.
        '''

        from astropy.table import vstack as tab_vstack

        for i, fil in enumerate(self.filaments):
            if i == 0:
                tab = fil.radprof_fit_table(unit=xunit)
                continue

            add_tab = fil.radprof_fit_table(unit=xunit)

            # Concatenate the row together
            tab = tab_vstack([tab, add_tab])

        return tab

    def total_intensity(self, bkg_subtract=False, bkg_mod_index=2):
        '''
        Return the sum of all pixels within the FWHM of the filament.

        .. warning::
            `fil_finder_2D` multiplied the total intensity by the angular size
            of a pixel. This function is just the sum of pixel values. Unit
            conversions can be applied on the output if needed.

        Parameters
        ----------
        bkg_subtract : bool, optional
            Subtract off the fitted background level.
        bkg_mod_index : int, optional
            Indicate which element in `Filament2D.radprof_params` is the
            background level. Defaults to 2 for the Gaussian with background
            model.

        Returns
        -------
        total_intensity : `~astropy.units.Quantity`
            Array of the total intensities for the filament.
        '''

        total_intensity = []

        for i, fil in enumerate(self.filaments):
            total_fil = fil.total_intensity(bkg_subtract=bkg_subtract,
                                            bkg_mod_index=bkg_mod_index)
            if i == 0:
                unit = total_fil.unit

            total_intensity.append(total_fil.value)

        return total_intensity * unit

    def median_brightness(self):
        '''
        Returns the median brightness along the skeleton of the filament.

        Returns
        -------
        filament_brightness : list
            Average brightness/intensity over the skeleton pixels
            for each filament.
        '''

        if len(self.filaments) == 0:
            return np.array([])

        med_bright0 = self.filaments[0].median_brightness(self.image)

        median_bright = np.zeros(len(self.filaments))

        if hasattr(med_bright0, 'unit'):
            median_bright = median_bright * med_bright0.unit
            median_bright[0] = med_bright0

        for i, fil in enumerate(self.filaments):
            median_bright[i] = fil.median_brightness(self.image)

        return median_bright

    def filament_model(self, max_radius=None, bkg_subtract=True,
                       bkg_mod_index=2):
        '''
        Returns a model of the diffuse filamentary network based
        on the radial profiles.

        Parameters
        ----------
        max_radius : `~astropy.units.Quantity`, optional
            Number of pixels to extend profiles to. If None is given, each
            filament model is computed to 3 * FWHM.
        bkg_subtract : bool, optional
            Subtract off the fitted background level.
        bkg_mod_index : int, optional
            Indicate which element in `Filament2D.radprof_params` is the
            background level. Defaults to 2 for the Gaussian with background
            model.

        Returns
        -------
        model_image : `~numpy.ndarray`
            Array of the model

        '''

        model_image = np.zeros(self.image.shape)

        for i, fil in enumerate(self.filaments):

            if max_radius is None:
                max_rad = 3 * fil.radprof_fwhm()[0]
            else:
                max_rad = max_radius

            fil_model = fil.model_image(max_radius=max_rad,
                                        bkg_subtract=bkg_subtract,
                                        bkg_mod_index=bkg_mod_index)

            # Add to the global model.
            if i == 0 and hasattr(fil_model, 'unit'):
                model_image = model_image * fil_model.unit

            pad_size = int(max_rad.value)
            arr_cent = [(fil_model.shape[0] - pad_size * 2 - 1) / 2. +
                        fil.pixel_extents[0][0],
                        (fil_model.shape[1] - pad_size * 2 - 1) / 2. +
                        fil.pixel_extents[0][1]]

            big_slice, small_slice = overlap_slices(model_image.shape,
                                                    fil_model.shape,
                                                    arr_cent)
            model_image[big_slice] += fil_model[small_slice]

        return model_image

    def covering_fraction(self, max_radius=None, bkg_subtract=True,
                          bkg_mod_index=2):
        '''
        Compute the fraction of the intensity in the image contained in
        the filamentary structure.

        Parameters
        ----------
        max_radius : `~astropy.units.Quantity`, optional
            Number of pixels to extend profiles to. If None is given, each
            filament model is computed to 3 * FWHM.
        bkg_subtract : bool, optional
            Subtract off the fitted background level.
        bkg_mod_index : int, optional
            Indicate which element in `Filament2D.radprof_params` is the
            background level. Defaults to 2 for the Gaussian with background
            model.

        Returns
        -------
        covering_fraction : float
            Fraction of the total image intensity contained in the
            filamentary structure (based on the local, individual fits)
        '''

        fil_model = self.filament_model(max_radius=max_radius,
                                        bkg_subtract=bkg_subtract,
                                        bkg_mod_index=bkg_mod_index)

        frac = np.nansum(fil_model) / np.nansum(self.image)
        if hasattr(frac, 'value'):
            frac = frac.value

        return frac

    def ridge_profiles(self):
        '''
        Return the image values along the longest path of the skeleton.
        See `~Filament2D.ridge_profile`.

        Returns
        -------
        ridges : list
            List of the ridge values for each filament.
        '''

        return [fil.ridge_profile(self.image) for fil in self.filaments]

    def output_table(self, xunit=u.pix, world_coord=False, **kwargs):
        '''
        Return the analysis results as an astropy table.

        If `~FilFinder2D.exec_rht` was run on the whole skeleton, the
        orientation and curvature will be included in the table. If the RHT
        was run on individual branches, use `~FilFinder2D.save_branch_tables`
        with `include_rht=True` to save the curvature and orientations.

        Parameters
        ----------
        xunit : `~astropy.units.Unit`, optional
            Unit for spatial properties. Defaults to pixel units.
        world_coord : bool, optional
            Return the median filament position in world coordinates.
        kwargs : Passed to `~FilFinder2D.total_intensity`.

        Return
        ------
        tab : `~astropy.table.Table`
            Table with all analyzed parameters.
        '''

        tab = Table()

        tab["lengths"] = Column(self.lengths(xunit))
        tab['branches'] = Column(self.branch_properties["number"])
        tab['total_intensity'] = Column(self.total_intensity(**kwargs))
        tab['median_brightness'] = Column(self.median_brightness())

        if not self._rht_branches_flag:
            tab['orientation'] = Column(self.orientation)
            tab['curvature'] = Column(self.curvature)

        # Return centres
        fil_centres = self.filament_positions(world_coord=world_coord)

        if fil_centres[0][0].unit == u.pix:
            yposn = [centre[0].value for centre in fil_centres] * u.pix
            xposn = [centre[1].value for centre in fil_centres] * u.pix
            tab['X_posn'] = Column(xposn)
            tab['Y_posn'] = Column(yposn)
        else:
            ra_unit = fil_centres[0][0].unit
            ra = [centre[0].value for centre in fil_centres] * ra_unit
            dec_unit = fil_centres[0][1].unit
            dec = [centre[1] for centre in fil_centres] * dec_unit
            tab['RA'] = Column(ra)
            tab['Dec'] = Column(dec)

        # Join with the width table
        width_table = self.width_fits(xunit=xunit)

        from astropy.table import hstack as tab_hstack

        tab = tab_hstack([tab, width_table])
        return tab

    def branch_tables(self, include_rht=False):
        '''
        Return the branch properties of each filament. If the RHT was run
        on individual branches (`branches=True` in `~FilFinder2D.exec_rht`),
        the orientation and curvature of each branch can be included in the
        saved table.

        A table will be returned for each filament in order of the filaments
        in `~FilFinder2D.filaments`.

        Parameters
        ----------
        include_rht : bool, optional
            Include RHT orientation and curvature if `~FilFinder2D.exec_rht`
            is run with `branches=True`.

        Returns
        -------
        tables : list
            List of `~astropy.table.Table` for each filament.
        '''

        tables = []

        for n, fil in enumerate(self.filaments):

            tables.append(fil.branch_table(include_rht=include_rht))

        return tables

    def save_fits(self, save_name=None, **kwargs):
        '''
        Save the mask and the skeleton array as FITS files. The header includes
        the settings used to create them.

        The mask, skeleton, longest skeletons, and model are included in the
        outputted file. The skeletons are labeled to match their order in
        `~FilFinder2D.filaments`.

        Parameters
        ----------
        save_name : str, optional
            The prefix for the saved file. If None, the save name specified
            when `~FilFinder2D` was first called.
        kwargs : Passed to `~FilFinder2D.filament_model`.
        '''

        if save_name is None:
            save_name = self.save_name
        else:
            save_name = os.path.splitext(save_name)[0]

        # Create header based off of image header.
        if self.header is not None:
            new_hdr = deepcopy(self.header)
        else:
            new_hdr = fits.Header()
            new_hdr["NAXIS"] = 2
            new_hdr["NAXIS1"] = self.image.shape[1]
            new_hdr["NAXIS2"] = self.image.shape[0]

        try:  # delete the original history
            del new_hdr["HISTORY"]
        except KeyError:
            pass

        from fil_finder.version import version

        new_hdr["BUNIT"] = ("bool", "")

        new_hdr["COMMENT"] = \
            "Mask created by fil_finder at {0}. Version {1}"\
            .format(time.strftime("%c"), version)
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
        new_hdr['BITPIX'] = "8"

        mask_hdu = fits.PrimaryHDU(self.mask.astype(int), new_hdr)

        out_hdu = fits.HDUList([mask_hdu])

        # Skeletons

        new_hdr_skel = new_hdr.copy()
        new_hdr_skel["BUNIT"] = ("int", "")
        new_hdr_skel['BITPIX'] = "16"

        new_hdr_skel["COMMENT"] = "Skeleton Size Threshold: " + \
            str(self.skel_thresh)
        new_hdr_skel["COMMENT"] = "Branch Size Threshold: " + \
            str(self.branch_thresh)

        # Final Skeletons - create labels which match up with table output

        labels = nd.label(self.skeleton, eight_con())[0]
        out_hdu.append(fits.ImageHDU(labels, header=new_hdr_skel))

        # Longest Paths
        labels_lp = nd.label(self.skeleton_longpath, eight_con())[0]
        out_hdu.append(fits.ImageHDU(labels_lp,
                                     header=new_hdr_skel))

        model = self.filament_model(**kwargs)
        if hasattr(model, 'unit'):
            model = model.value

        model_hdr = new_hdr.copy()
        model_hdr['COMMENT'] = "Image generated from fitted filament models."
        if self.header is not None:
            bunit = self.header.get('BUNIT', None)
            if bunit is not None:
                model_hdr['BUNIT'] = bunit
            else:
                model_hdr['BUNIT'] = ""
        else:
            model_hdr['BUNIT'] = ""

        model_hdr['BITPIX'] = fits.DTYPE2BITPIX[str(model.dtype)]
        model_hdu = fits.ImageHDU(model, header=model_hdr)

        out_hdu.append(model_hdu)

        out_hdu.writeto("{0}_image_output.fits".format(save_name))

    def save_stamp_fits(self, save_name=None, pad_size=20 * u.pix,
                        **kwargs):
        '''
        Save stamps of each filament image, skeleton, longest-path skeleton,
        and the model image.

        A suffix of "stamp_{num}" is added to each file, where the number is
        is the order in the list of `~FilFinder2D.filaments`.

        Parameters
        ----------
        save_name : str, optional
            The prefix for the saved file. If None, the save name specified
            when `~FilFinder2D` was first called.
        stamps : bool, optional
            Enables saving of individual stamps
        kwargs : Passed to `~Filament2D.save_fits`.
        '''
        if save_name is None:
            save_name = self.save_name
        else:
            save_name = os.path.splitext(save_name)[0]

        for n, fil in enumerate(self.filaments):

            savename = "{0}_stamp_{1}.fits".format(save_name, n)

            fil.save_fits(savename, self.image, pad_size=pad_size,
                          **kwargs)
