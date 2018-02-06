# Licensed under an MIT open source license - see LICENSE

import numpy as np
import astropy.units as u

from .length import (init_lengths, main_length, make_final_skeletons,
                     pre_graph, longest_path, prune_graph)
from .pixel_ident import pix_identify
from .utilities import pad_image
from .base_conversions import UnitConverter


class FilamentNDBase(object):
    """
    Analysis and properties of a single filament object.
    """
    @property
    def pixel_coords(self):
        return self._pixel_coords

    @property
    def pixel_extents(self):
        return [tuple([coord.min() for coord in self._orig_pixel_coords]),
                tuple([coord.max() for coord in self._orig_pixel_coords])]


class Filament2D(FilamentNDBase):
    """
    Analysis and properties of a 2D filament.

    Parameters
    ----------
    pixel_coords : tuple of `~np.ndarray`
        Pixel coordinates as a set of arrays (i.e., the output from
        `~numpy.where`).
    converter : `~fil_finder.base_conversions.UnitConverter`, optional
        Unit converter class.
    wcs : `~astropy.wcs.WCS`, optional
        WCS information for the pixel set.
    distance : `~astropy.units.Quantity`, optional
        Distance to the region described by the pixel set. Requires for
        conversions to physical units.
    """
    def __init__(self, pixel_coords, converter=None, wcs=None, distance=None):
        super(Filament2D, self).__init__()

        self._pixel_coords = pixel_coords

        # Create a separate account of the initial skeleton pixels
        self._orig_pixel_coords = pixel_coords

        if converter is not None:
            self._converter = converter
        else:
            self._converter = UnitConverter(wcs=wcs, distance=distance)

    def image_slice(self, pad_size=0):
        '''
        Returns a slice for the original image to cut out the filament region.
        '''
        return tuple([slice(extent[0] - pad_size, extent[1] + pad_size + 1)
                      for extent in zip(*self.pixel_extents)])

    def skeleton(self, pad_size=0, corner_pix=None, out_type='all'):
        '''
        Create a mask from the pixel coordinates.

        Parameters
        ----------
        pad_size : int, optional
            Number of pixels to pad along each edge.
        corner_pix : tuple of ints, optional
            The position of the left-bottom corner of the pixels in the
            skeleton. Used for offsetting the location of the pixels.
        out_type : {"all", "longpath"}, optional
            Return the entire skeleton or just the longest path. Default is to
            return the whole skeleton.

        Returns
        -------
        mask : `~numpy.ndarray`
            Boolean mask containing the skeleton pixels.
        '''

        pad_size = int(pad_size)
        if pad_size < 0:
            raise ValueError("pad_size must be a positive integer.")

        if corner_pix is None:
            # Place the smallest pixel in the set at the pad size
            corner_pix = [pad_size, pad_size]

        out_types = ['all', 'longpath']
        if out_type not in out_types:
            raise ValueError("out_type must be 'all' or 'longpath'.")

        y_shape = self.pixel_extents[1][0] - self.pixel_extents[0][0] + \
            2 * pad_size + 1
        x_shape = self.pixel_extents[1][1] - self.pixel_extents[0][1] + \
            2 * pad_size + 1

        mask = np.zeros((y_shape, x_shape), dtype=bool)

        if out_type == 'all':
            pixels = self.pixel_coords
        else:
            if not hasattr(self, '_longpath_pixel_coords'):
                raise AttributeError("longest path is not defined. Run "
                                     "`Filament2D.skeleton_analysis` first.")
            pixels = self.longpath_pixel_coords

        mask[pixels[0] - self.pixel_extents[0][0] + corner_pix[0],
             pixels[1] - self.pixel_extents[0][1] + corner_pix[1]] = True

        return mask

    def skeleton_analysis(self, image, verbose=False, save_png=False,
                          save_name=None, prune_criteria='all',
                          relintens_thresh=0.2,
                          branch_thresh=0 * u.pix):
        '''
        Run the skeleton analysis. See a full description in
        `~fil_finder.FilFinder2D`.

        Parameters
        ----------
        img : `~numpy.ndarray`
            Data the filament was extracted from.

        prune_criteria : {'all', 'intensity', 'length'}, optional
            Choose the property to base pruning on. 'all' requires that the
            branch fails to satisfy the length and relative intensity checks.

        Attributes
        ----------
        branch_properties :

        length :

        longpath_pixel_coords :

        graph :
        '''

        # NOTE:
        # All of these functions are essentially the same as those used for
        # fil_finder_2D. For now, they all are expecting lists with each
        # filament property as an element. Everything is wrapped to be a list
        # because of this, but will be removed once fil_finder_2D is removed.
        # A lot of this can be streamlined in that process.

        # Must have a pad size of 1 for the morphological operations.
        pad_size = 1

        interpts, hubs, ends, filbranches, labeled_mask =  \
            pix_identify([self.skeleton(pad_size=pad_size)], 1)

        # Do we need to pad the image before slicing?
        input_image = pad_image(image, self.pixel_extents, pad_size)

        # If the padded image matches the mask size, don't need additional
        # slicing
        if input_image.shape != labeled_mask[0].shape:
            input_image = input_image[self.image_slice(pad_size=pad_size)]

        # The mask and sliced image better have the same shape!
        if input_image.shape != labeled_mask[0].shape:
            raise AssertionError("Sliced image shape does not equal the mask "
                                 "shape. This should never happen! If you see"
                                 " this issue, please report it as a bug!")

        branch_properties = init_lengths(labeled_mask, filbranches,
                                         [[(0, 0), (0, 0)]],
                                         input_image)

        # Add the number of branches onto the dictionary
        branch_properties["number"] = filbranches

        edge_list, nodes = pre_graph(labeled_mask,
                                     branch_properties,
                                     interpts, ends)

        max_path, extremum, G = longest_path(edge_list, nodes,
                                             verbose=verbose,
                                             save_png=save_png,
                                             save_name=save_name,
                                             skeleton_arrays=labeled_mask)

        updated_lists = \
            prune_graph(G, nodes, edge_list, max_path, labeled_mask,
                        branch_properties, prune_criteria=prune_criteria,
                        length_thresh=branch_thresh.value,
                        relintens_thresh=relintens_thresh)

        labeled_mask, edge_list, nodes, branch_properties = \
            updated_lists

        self._graph = G[0]

        length_output = main_length(max_path, edge_list, labeled_mask,
                                    interpts,
                                    branch_properties["length"],
                                    1.,
                                    verbose=verbose, save_png=save_png,
                                    save_name=save_name)
        lengths, long_path_array = length_output

        good_long_pix = np.where(long_path_array[0])
        self._longpath_pixel_coords = \
            (good_long_pix[0] + self.pixel_extents[0][0] - pad_size,
             good_long_pix[1] + self.pixel_extents[0][1] - pad_size)

        self._length = lengths[0] * u.pix

        final_fil_arrays =\
            make_final_skeletons(labeled_mask, interpts,
                                 verbose=verbose, save_png=save_png,
                                 save_name=save_name)

        # Update the skeleton pixels
        good_pix = np.where(final_fil_arrays[0])
        self._pixel_coords = \
            (good_pix[0] + self.pixel_extents[0][0] - pad_size,
             good_pix[1] + self.pixel_extents[0][1] - pad_size)

        self._branch_properties = \
            {'length': branch_properties['length'][0] * u.pix,
             'intensity': np.array(branch_properties['intensity'][0]),
             'number': branch_properties['number'][0]}

    @property
    def branch_properties(self):
        return self._branch_properties

    @property
    def intersec_pts(self):
        '''
        Skeleton pixels associated intersections.
        '''
        return self.pixel_coords[self._intersec_idx]

    @property
    def end_pts(self):
        '''
        Skeleton pixels associated branch end.
        '''
        return self.pixel_coords[self._ends_idx]

    def length(self, unit=u.pixel):
        return self._converter.from_pixel(self._length, unit)

    @property
    def longpath_pixel_coords(self):
        '''
        Pixel coordinates of the longest path
        '''
        return self._longpath_pixel_coords

    @property
    def graph(self):
        '''
        The networkx graph for the filament.
        '''
        return self._graph

    def plot_graph(self, save_name=None):
        pass

    def orientation_analysis(self):
        '''
        Use the RHT to find the filament orientation and dispersion.
        '''
        pass

    def width_analysis(self):
        pass

    def profile_analysis(self):
        pass