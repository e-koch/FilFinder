
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import skimage.morphology as mo
import networkx as nx
import warnings
import astropy.units as u

from .filament import FilamentPPP
from .skeleton3D import Skeleton3D
from .base_conversions import (BaseInfoMixin, UnitConverter,
                               find_beam_properties, data_unit_check)
from .threshold_local_3D import threshold_local

class FilFinderPPP(BaseInfoMixin, Skeleton3D):
    """
    Extract and analyze filamentary structure from a 3D dataset.

    Parameters
    ----------
    image: `~numpy.ndarray`
        A 3D array of the data to be analyzed.
    mask: numpy.ndarray, optional
        A pre-made, boolean mask may be supplied to skip the segmentation
        process. The algorithm will skeletonize and run the analysis portions
        only.
    save_name: str, optional
        Sets the prefix name that is used for output files.

    """

    def __init__(self, image, wcs=None, mask=None, distance=None, save_name='FilFinderPPP_output'):

        # Add warning that this is under development
        warnings.warn("This algorithm is under development. Not all features are implemented"
                      " or tested. Use with caution.")

        self._has_skan()

        # TODO add image checking here
        self._image = image

        self._wcs = wcs

        self.save_name = save_name

        # Mask Initialization
        self.mask = None
        if mask is not None:
            if self.image.shape != mask.shape:
                raise ValueError("The given pre-existing mask must"
                                 " have the same shape as input image.")
            # Clearing NaN entries
            mask[np.isnan(mask)] = 0.0
            self.mask = mask

        # TODO: need to handle cases without wcs info for the unit conversion
        # TODO: minimum should be a pixel scale for 3D products.
        if self.wcs is not None:
            self.converter = UnitConverter(self.wcs, distance)
        else:
            self.converter = lambda x: x

    def preprocess_image(self, skip_flatten=False, flatten_percent=None):
        """
        Preprocess and flatten the dataset before running the masking process.

        Parameters
        ----------
        skip_flatten : bool, optional
            Skip the flattening process and use the original image to
            construct the mask. Default is False.
        flatten_percent : int, optional
            The percentile of the data (0-100) to set the normalization.
            Default is None.

        """
        if skip_flatten:
            self._flatten_threshold = None
            self.flat_img = self._image

        else:
            # TODO Add in here
            pass

    def create_mask(self, adapt_thresh=9, glob_thresh=0.0,
                    ball_radius=2, min_object_size=27*3,
                    max_hole_size=100,
                    verbose=False,
                    save_png=False, use_existing_mask=False,
                    **adapt_kwargs):
        """
        Runs the segmentation process and returns a mask of the filaments found.

        Parameters
        ----------
        glob_thresh : float, optional
            Minimum value to keep in mask. Default is None.
        verbose : bool, optional
            Enables plotting. Default is False.
        save_png : bool, optional
            Saves the plot in verbose mode. Default is False.
        use_existing_mask : bool, optional
            If ``mask`` is already specified, enabling this skips
            recomputing the mask.

        Attributes
        -------
        mask : numpy.ndarray
            The mask of the filaments.

        """

        if self.mask is not None and use_existing_mask:
            warnings.warn("Using inputted mask. Skipping creation of a"
                          "new mask.")
            # Skip if pre-made mask given
            self.glob_thresh = 'usermask'
            self.adapt_thresh = 'usermask'
            self.size_thresh = 'usermask'
            self.smooth_size = 'usermask'

            return

        if glob_thresh is None:
            self.glob_thresh = None
        else:
            # TODO Check if glob_thresh is proper
            self.glob_thresh = glob_thresh

        # Here starts the masking process
        flat_copy = self.flat_img.copy()

        # Removing NaNs in copy
        flat_copy[np.isnan(flat_copy)] = 0.0

        # Create the adaptive thresholded mask
        adapt_mask = flat_copy > threshold_local(flat_copy, adapt_thresh, **adapt_kwargs)

        # Add in global threshold mask
        adapt_mask = np.logical_and(adapt_mask, flat_copy > glob_thresh)

        # TODO should we use other shape here?
        # Create slider object
        selem = mo.ball(ball_radius)

        # Dilate the image
        # dilate = mo.dilation(adapt_mask, selem)

        # NOTE: Look into mo.diameter_opening and mo.diameter_closing
        dilate = mo.opening(adapt_mask, selem)

        # Removing dark spots and small bright cracks in image
        close = mo.closing(dilate)

        # Don't allow small holes: these lead to "shell"-shaped skeleton features
        mo.remove_small_objects(close, min_size=min_object_size, connectivity=1, in_place=True)
        mo.remove_small_holes(close, area_threshold=max_hole_size, connectivity=1, in_place=True)

        self.mask = close

    def analyze_skeletons(self, compute_longest_path=True,
                          do_prune=True,
                          verbose=False, save_png=False,
                          save_name=None, prune_criteria='all',
                          relintens_thresh=0.2, max_prune_iter=10,
                          branch_thresh=0 * u.pix, test_print=False):
        '''
        '''

        self._compute_longest_path = compute_longest_path

        # Define the skeletons

        num = self._skel_labels.max()

        self.filaments = []

        for i in range(1, num + 1):

            coords = np.where(self._skel_labels == i)

            self.filaments.append(FilamentPPP(coords,
                                              converter=self.converter))

        # Calculate lengths and find the longest path.
        # Followed by pruning.
        for num, fil in enumerate(self.filaments):
            if test_print:
                print(f"Skeleton analysis for {num} of {len(self.filaments)}")

            fil._make_skan_skeleton()

            fil.skeleton_analysis(self._image,
                                  compute_longest_path=compute_longest_path,
                                  do_prune=do_prune,
                                  verbose=verbose, save_png=save_png,
                                  save_name=save_name, prune_criteria=prune_criteria,
                                  relintens_thresh=relintens_thresh, max_prune_iter=max_prune_iter,
                                  branch_thresh=branch_thresh, test_print=test_print)

        # Update the skeleton array
        new_skel = np.zeros_like(self.skeleton)

        if self._compute_longest_path:
            new_skel_longpath = np.zeros_like(self.skeleton)

        for fil in self.filaments:

            new_skel[fil.pixel_coords[0],
                     fil.pixel_coords[1],
                     fil.pixel_coords[2]] = True

            if self._compute_longest_path:

                new_skel_longpath[fil.longpath_pixel_coords[0],
                                fil.longpath_pixel_coords[1],
                                fil.longpath_pixel_coords[2]] = True

        self.skeleton = new_skel

        if self._compute_longest_path:
            self.skeleton_longpath = new_skel_longpath

