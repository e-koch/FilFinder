# Licensed under an MIT open source license - see LICENSE

import numpy as np
import astropy.units as u
import networkx as nx
import warnings
import scipy.ndimage as nd
from astropy.nddata import extract_array
import astropy.modeling as mod
from astropy.modeling.models import Gaussian1D, Const1D
import sys

if sys.version_info[0] >= 3:
    import _pickle as pickle
else:
    import cPickle as pickle

from .length import (init_lengths, main_length, make_final_skeletons,
                     pre_graph, longest_path, prune_graph)
from .pixel_ident import pix_identify
from .utilities import pad_image, in_ipynb, red_chisq
from .base_conversions import UnitConverter
from .rollinghough import rht
from .width import (radial_profile, gaussian_model, fit_radial_model,
                    nonparam_width)


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

    def position(self, world_coord=False):
        '''
        Return the centre position of the filament based on the pixel
        coordinates.
        '''
        centres = [np.median(coord) for coord in self._orig_pixel_coords]

        if world_coord:
            if hasattr(self._converter, '_wcs'):
                wcs = self._converter._wcs
                # Convert to world coordinates
                posn_tuple = centres + [0]
                w_centres = wcs.all_pix2world(*posn_tuple)

                # Attach units
                wu_centres = [val * u.Unit(wcs.wcs.cunit[i]) for i, val
                              in enumerate(w_centres)]
                return wu_centres
            else:
                warnings.warn("No WCS information given. Returning pixel"
                              " position.")
                return [centre * u.pix for centre in centres]
        else:
            return [centre * u.pix for centre in centres]


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

    def image_slicer(self, image, out_shape, pad_size=0):
        '''
        Create a cut-out of a given image to some output shape with optional
        padding on the edges. The given image must be on the same pixel grid
        as the image used to create the skeleton.

        Parameters
        ----------
        image : `~numpy.ndarray` or `~astropy.units.Quantity`
            Image to slice out around the skeleton.
        out_shape : tuple
            2D output shape.
        pad_size : int, optional
            Number of pixels to pad.

        Returns
        -------
        out_arr : `~numpy.ndarray` or `~astropy.units.Quantity`
            Output array with given shape.
        '''

        arr_cent = [(out_shape[0] - pad_size * 2 - 1) / 2. +
                    self.pixel_extents[0][0],
                    (out_shape[1] - pad_size * 2 - 1) / 2. +
                    self.pixel_extents[0][1]]

        out_arr = extract_array(image, out_shape, arr_cent)

        # astropy v4.0 now retains the unit. So only add a unit
        # when out_arr isn't a Quantity
        if hasattr(image, "unit") and not hasattr(out_arr, 'unit'):
            out_arr = out_arr * image.unit

        return out_arr

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
                          relintens_thresh=0.2, max_prune_iter=10,
                          branch_thresh=0 * u.pix):
        '''
        Run the skeleton analysis.

        Separates skeleton structures into branches and intersections. Branches
        below the pruning criteria are removed. The structure is converted into
        a graph object to find the longest path. The pruned skeleton is used in
        the subsequent analysis steps.

        Parameters
        ----------
        image : `~numpy.ndarray` or `~astropy.units.Quantity`
            Data the filament was extracted from.
        verbose : bool, optional
            Show intermediate plots.
        save_png : bool, optional
            Save the plots in verbose mode.
        save_name : str, optional
            Prefix for the saved plots.
        prune_criteria : {'all', 'intensity', 'length'}, optional
            Choose the property to base pruning on. 'all' requires that the
            branch fails to satisfy the length and relative intensity checks.
        relintens_thresh : float, optional
            Value between 0 and 1 that sets the relative importance of the
            intensity-to-length criteria when pruning. Only used if
            `prune_criteria='all'`.
        max_prune_iter : int, optional
            Maximum number of pruning iterations to apply.
        branch_thresh : `~astropy.units.Quantity`, optional
            Minimum length for a branch to be eligible to be pruned.
        '''

        # NOTE:
        # All of these functions are essentially the same as those used for
        # fil_finder_2D. For now, they all are expecting lists with each
        # filament property as an element. Everything is wrapped to be a list
        # because of this, but will be removed once fil_finder_2D is removed.
        # A lot of this can be streamlined in that process.

        if save_png and save_name is None:
            raise ValueError("save_name must be given when save_png=True.")

        # Must have a pad size of 1 for the morphological operations.
        pad_size = 1
        self._pad_size = pad_size

        branch_thresh = self._converter.to_pixel(branch_thresh)

        # Do we need to pad the image before slicing?
        input_image = pad_image(image, self.pixel_extents, pad_size)

        skel_mask = self.skeleton(pad_size=pad_size)

        # If the padded image matches the mask size, don't need additional
        # slicing
        if input_image.shape != skel_mask.shape:
            input_image = self.image_slicer(input_image, skel_mask.shape,
                                            pad_size=pad_size)

        # The mask and sliced image better have the same shape!
        if input_image.shape != skel_mask.shape:
            raise AssertionError("Sliced image shape does not equal the mask "
                                 "shape. This should never happen! If you see"
                                 " this issue, please report it as a bug!")

        iter = 0

        while True:

            skel_mask = self.skeleton(pad_size=pad_size)

            interpts, hubs, ends, filbranches, labeled_mask =  \
                pix_identify([skel_mask], 1)

            branch_properties = init_lengths(labeled_mask, filbranches,
                                             [[(0, 0), (0, 0)]],
                                             input_image)

            edge_list, nodes, loop_edges = \
                pre_graph(labeled_mask, branch_properties, interpts, ends)

            max_path, extremum, G = \
                longest_path(edge_list, nodes,
                             verbose=False,
                             skeleton_arrays=labeled_mask)

            # Skip pruning if skeleton has only one branch
            if len(G[0].nodes()) > 1:
                updated_lists = \
                    prune_graph(G, nodes, edge_list, max_path, labeled_mask,
                                branch_properties, loop_edges,
                                prune_criteria=prune_criteria,
                                length_thresh=branch_thresh.value,
                                relintens_thresh=relintens_thresh,
                                max_iter=1)
                labeled_mask, edge_list, nodes, branch_properties = \
                    updated_lists

            final_fil_arrays =\
                make_final_skeletons(labeled_mask, interpts,
                                     verbose=False)

            # Update the skeleton pixels
            good_pix = np.where(final_fil_arrays[0])
            self._pixel_coords = \
                (good_pix[0] + self.pixel_extents[0][0] - pad_size,
                 good_pix[1] + self.pixel_extents[0][1] - pad_size)

            if iter == 0:
                prev_G = G[0]
                iter += 1
                if iter == max_prune_iter:
                    break
                else:
                    continue

            # Isomorphic comparison is failing for networkx 2.1
            # I don't understand the error, so we'll instead require
            # that the nodes be the same. This should be safe as
            # pruning can only remove nodes.

            # edge_match = iso.numerical_edge_match('weight', 1)
            # if nx.is_isomorphic(prev_G, G[0],
            #                     edge_match=edge_match):

            # the node attribute was removed in 2.4.
            if hasattr(G, 'node'):
                if prev_G.node == G[0].node:
                    break

            if hasattr(G, 'nodes'):
                if prev_G.nodes == G[0].nodes:
                    break

            prev_G = G[0]
            iter += 1

            if iter >= max_prune_iter:
                warnings.warn("Graph pruning reached max iterations.")
                break

        self._graph = G[0]

        # Run final analyses for plotting, etc.
        max_path, extremum, G = \
            longest_path(edge_list, nodes,
                         verbose=verbose,
                         save_png=save_png,
                         save_name="{0}_graphstruct.png".format(save_name),
                         skeleton_arrays=labeled_mask)

        length_output = main_length(max_path, edge_list, labeled_mask,
                                    interpts,
                                    branch_properties["length"],
                                    1.,
                                    verbose=verbose, save_png=save_png,
                                    save_name="{0}_longestpath.png".format(save_name))
        lengths, long_path_array = length_output

        good_long_pix = np.where(long_path_array[0])
        self._longpath_pixel_coords = \
            (good_long_pix[0] + self.pixel_extents[0][0] - pad_size,
             good_long_pix[1] + self.pixel_extents[0][1] - pad_size)

        self._length = lengths[0] * u.pix

        final_fil_arrays =\
            make_final_skeletons(labeled_mask, interpts,
                                 verbose=verbose, save_png=save_png,
                                 save_name="{0}_finalskeleton.png".format(save_name))

        # Track the final intersection and end points
        interpts, hubs, ends =  \
            pix_identify([final_fil_arrays[0].copy()], 1)[:3]

        # Adjust intersection and end points to be in the original array
        # positions
        corr_inters = []
        for inter in interpts[0]:
            per_inter = []

            for ints in inter:
                per_inter.append((ints[0] + self.pixel_extents[0][0] - pad_size,
                                  ints[1] + self.pixel_extents[0][1] - pad_size))

            corr_inters.append(per_inter)
        self._interpts = corr_inters

        corr_ends = []
        for end in ends[0]:
            corr_ends.append((end[0] + self.pixel_extents[0][0] - pad_size,
                              end[1] + self.pixel_extents[0][1] - pad_size))
        self._endpts = corr_ends

        # Update the skeleton pixels
        good_pix = np.where(final_fil_arrays[0])
        self._pixel_coords = \
            (good_pix[0] + self.pixel_extents[0][0] - pad_size,
             good_pix[1] + self.pixel_extents[0][1] - pad_size)

        self._branch_properties = \
            {'length': branch_properties['length'][0] * u.pix,
             'intensity': np.array(branch_properties['intensity'][0]),
             'number': branch_properties['number'][0],
             'pixels': branch_properties['pixels'][0]}

    @property
    def branch_properties(self):
        '''
        Dictionary with branch lengths, average intensity, and pixels.
        '''
        return self._branch_properties

    def branch_pts(self, img_coords=False):
        '''
        Pixels within each skeleton branch.

        Parameters
        ----------
        img_coords : bool
            Return the branch pts in coordinates of the original image.
        '''
        if not img_coords:
            return self.branch_properties['pixels']

        # Transform from per-filament to image coords
        img_branch_pts = []
        for bpts in self.branch_properties['pixels']:

            bpts_copy = bpts.copy()

            bpts_copy[:, 0] = bpts[:, 0] + self.pixel_extents[0][0] - self._pad_size
            bpts_copy[:, 1] = bpts[:, 1] + self.pixel_extents[0][1] - self._pad_size

            img_branch_pts.append(bpts_copy)

        return img_branch_pts


    @property
    def intersec_pts(self):
        '''
        Skeleton pixels associated intersections.
        '''
        return self._interpts

    @property
    def end_pts(self):
        '''
        Skeleton pixels associated branch end.
        '''
        return self._endpts

    def length(self, unit=u.pixel):
        '''
        The longest path length of the skeleton

        Parameters
        ----------
        unit : `~astropy.units.Unit`, optional
            Pixel, angular, or physical unit to convert to.
        '''
        return self._converter.from_pixel(self._length, unit)

    @property
    def longpath_pixel_coords(self):
        '''
        Pixel coordinates of the longest path.
        '''
        return self._longpath_pixel_coords

    @property
    def graph(self):
        '''
        The networkx graph for the filament.
        '''
        return self._graph

    def plot_graph(self, save_name=None, layout_func=nx.spring_layout):
        '''
        Plot the graph structure.

        Parameters
        ----------
        save_name : str, optional
            Name of saved plot. A plot is only saved if a name is given.
        layout_func : networkx layout function, optional
            Layout function from networkx. Defaults to `spring_layout`.
        '''
        import matplotlib.pyplot as plt

        G = self.graph

        elist = [(u, v) for (u, v, d) in G.edges(data=True)]
        posns = layout_func(G)

        nx.draw_networkx_nodes(G, posns, node_size=200)
        nx.draw_networkx_edges(G, posns, edgelist=elist, width=2)
        nx.draw_networkx_labels(G, posns, font_size=10,
                                font_family='sans-serif')
        plt.axis('off')

        if save_name is not None:
            # Save the plot
            plt.savefig(save_name)
            plt.close()
        else:
            plt.show()

        # Add in the ipynb checker

    def rht_analysis(self, radius=10 * u.pix, ntheta=180,
                     background_percentile=25):
        '''
        Use the RHT to find the filament orientation and dispersion of the
        longest path.

        Parameters
        ----------
        radius : `~astropy.units.Quantity`, optional
            Radius of the region to compute the orientation within. Converted
            to pixel units and rounded to the nearest integer.
        ntheta : int, optional
            Number of angles to sample at. Default is 180.
        background_percentile : float, optional
            Float between 0 and 100 that sets a background level for the RHT
            distribution before calculating orientation and curvature.
        '''

        if not hasattr(radius, 'unit'):
            warnings.warn("Radius has no given units. Assuming pixel units.")
            radius *= u.pix

        radius = int(round(self._converter.to_pixel(radius).value))

        longpath_arr = self.skeleton(out_type='longpath')

        longpath_arr = np.fliplr(longpath_arr)

        theta, R, quant = rht(longpath_arr, radius, ntheta,
                              background_percentile)

        twofive, mean, sevenfive = quant

        self._orientation = mean * u.rad

        if sevenfive > twofive:
            self._curvature = np.abs(sevenfive - twofive) * u.rad
        else:
            self._curvature = (np.abs(sevenfive - twofive) + np.pi) * u.rad

        self._orientation_hist = [theta, R]
        self._orientation_quantiles = [twofive, sevenfive]

    @property
    def orientation_hist(self):
        '''
        Distribution of orientations from the RHT along the longest path.

        Contains the angles of the distribution bins and the values in those
        bins.
        '''
        return self._orientation_hist

    @property
    def orientation(self):
        '''
        Mean orientation of the filament along the longest path.
        '''
        return self._orientation

    @property
    def curvature(self):
        '''
        Interquartile range of the RHT orientation distribution along the
        longest path.
        '''
        return self._curvature

    def plot_rht_distrib(self, save_name=None):
        '''
        Plot the RHT distribution from `Filament2D.rht_analysis`.

        Parameters
        ----------
        save_name : str, optional
            Name of saved plot. A plot is only saved if a name is given.
        '''

        theta = self.orientation_hist[0]
        R = self.orientation_hist[1]

        import matplotlib.pyplot as plt

        median = self.orientation.value
        twofive, sevenfive = self._orientation_quantiles

        ax1 = plt.subplot(121, polar=True)
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
        plt.subplot(122)
        plt.imshow(self.skeleton(out_type='longpath'),
                   cmap="binary", origin="lower")
        if save_name is not None:
            plt.savefig(save_name)
            plt.close()
        else:
            plt.show()

    def rht_branch_analysis(self, radius=10 * u.pix, ntheta=180,
                            background_percentile=25,
                            min_branch_length=3 * u.pix):
        '''
        Use the RHT to find the filament orientation and dispersion of each
        branch in the filament.

        Parameters
        ----------
        radius : `~astropy.units.Quantity`, optional
            Radius of the region to compute the orientation within. Converted
            to pixel units and rounded to the nearest integer.
        ntheta : int, optional
            Number of angles to sample at. Default is 180.
        background_percentile : float, optional
            Float between 0 and 100 that sets a background level for the RHT
            distribution before calculating orientation and curvature.
        min_branch_length : `~astropy.units.Quantity`, optional
            Minimum length of a branch to run the RHT on. Branches that are
            too short will cause spikes along the axis angles or 45 deg. off.
        '''

        # Convert length cut to pixel units

        if not hasattr(radius, 'unit'):
            warnings.warn("Radius has no given units. Assuming pixel units.")
            radius *= u.pix

        if not hasattr(min_branch_length, 'unit'):
            warnings.warn("min_branch_length has no given units. Assuming "
                          "pixel units.")
            min_branch_length *= u.pix

        radius = int(round(self._converter.to_pixel(radius).value))
        min_branch_length = self._converter.to_pixel(min_branch_length).value

        means = []
        iqrs = []

        # Make padded arrays from individual branches
        for i, (pix, length) in enumerate(zip(self.branch_pts(img_coords=False),
                                              self.branch_properties['length'])):

            if length.value < min_branch_length:
                means.append(np.NaN)
                iqrs.append(np.NaN)
                continue

            # Setup size of array
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

            theta, R, quant = rht(branch_array, radius, ntheta,
                                  background_percentile)

            twofive, mean, sevenfive = quant

            means.append(mean)
            if sevenfive > twofive:
                iqrs.append(np.abs(sevenfive - twofive))
            else:
                iqrs.append(np.abs(sevenfive - twofive) + np.pi)

        self._orientation_branches = np.array(means) * u.rad
        self._curvature_branches = np.array(iqrs) * u.rad

    @property
    def orientation_branches(self):
        '''
        Orientations along each branch in the filament.
        '''
        return self._orientation_branches

    @property
    def curvature_branches(self):
        '''
        Curvature along each branch in the filament.
        '''
        return self._curvature_branches

    def width_analysis(self, image, all_skeleton_array=None,
                       max_dist=10 * u.pix,
                       pad_to_distance=0 * u.pix,
                       fit_model='gaussian_bkg',
                       fitter=None,
                       try_nonparam=True,
                       use_longest_path=False,
                       add_width_to_length=False,
                       deconvolve_width=True,
                       beamwidth=None,
                       fwhm_function=None,
                       chisq_max=10.,
                       **kwargs):
        '''

        Create an average radial profile for the filament and fit a given
        model.

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
        beamwidth : `~astropy.units.Quantity`, optional
            The beam width to deconvolve the FWHM from. Required if
            `deconvolve_width = True`.
        fwhm_function : function, optional
            Convert the width parameter to the FWHM. Must take the fit model
            as an argument and return the FWHM and its uncertainty. If no
            function is given, the Gaussian FWHM is used.
        chisq_max : float, optional
            Enable the fail flag if the reduced chi-squared value is above
            this limit.
        kwargs : Passed to `~fil_finder.width.radial_profile`.

        '''

        # Convert quantities to pixel units.
        max_dist = self._converter.to_pixel(max_dist).value
        pad_to_distance = self._converter.to_pixel(pad_to_distance).value

        if deconvolve_width and beamwidth is None:
            raise ValueError("beamwidth must be given when deconvolve_width is"
                             " enabled.")

        if beamwidth is not None:
            beamwidth = self._converter.to_pixel(beamwidth)

        # Use the max dist as the pad size
        pad_size = int(np.ceil(max_dist))

        # if given a master skeleton array, require it to be the same shape as
        # the image
        if all_skeleton_array is not None:
            if all_skeleton_array.shape != image.shape:
                raise ValueError("The shape of all_skeleton_array must match"
                                 " the given image.")

        if use_longest_path:
            skel_array = self.skeleton(pad_size=pad_size, out_type='longpath')
        else:
            skel_array = self.skeleton(pad_size=pad_size, out_type='all')

        out_shape = skel_array.shape
        input_image = self.image_slicer(image, out_shape, pad_size=pad_size)

        if all_skeleton_array is not None:
            input_all_skeleton_array = \
                self.image_slicer(all_skeleton_array, out_shape,
                                  pad_size=pad_size)
        else:
            input_all_skeleton_array = None

        # Create distance arrays to build profile from
        dist_skel_arr = nd.distance_transform_edt(np.logical_not(skel_array))

        # And create a distance array from the full skeleton array if given
        if input_all_skeleton_array is not None:
            dist_skel_all = nd.distance_transform_edt(np.logical_not(input_all_skeleton_array))
        else:
            dist_skel_all = None

        # Need the unbinned data for the non-parametric fit.
        out = radial_profile(input_image, dist_skel_all,
                             dist_skel_arr,
                             [(0, 0), (0, 0)],
                             max_distance=max_dist,
                             pad_to_distance=pad_to_distance,
                             **kwargs)

        if out is None:
            raise ValueError("Building radial profile failed. Check the input"
                             " image for NaNs.")
        else:
            dist, radprof, weights, unbin_dist, unbin_radprof = out

        # Attach units
        xunit = u.pix
        if hasattr(image, 'unit'):
            yunit = image.unit
        else:
            yunit = u.dimensionless_unscaled

        self._yunit = yunit

        radprof = radprof * yunit
        dist = dist * xunit

        self._radprofile = [dist, radprof]
        self._unbin_radprofile = [unbin_dist * xunit,
                                  unbin_radprof * yunit]

        # Make sure the given model is valid
        if not isinstance(fit_model, mod.Model):
            skip_fitting = False
            self._radprof_type = fit_model
            # Check the default types
            if fit_model == "gaussian_bkg":
                fit_model = gaussian_model(dist, radprof, with_bkg=True)
            elif fit_model == "gaussian_nobkg":
                fit_model = gaussian_model(dist, radprof, with_bkg=False)
            elif fit_model == "nonparam":
                skip_fitting = True
            else:
                raise ValueError("fit_model must be an "
                                 "astropy.modeling.Fittable1DModel or "
                                 "one of the default models: 'gaussian_bkg',"
                                 " 'gaussian_nobkg', or 'nonparam'.")
        else:
            # Record the fit type
            self._radprof_type = fit_model.name

        if not skip_fitting:
            fitted_model, fitter = fit_radial_model(dist, radprof, fit_model,
                                                    weights=weights)

            # Only keep the non-fixed parameters. The fixed parameters won't
            # appear in the covariance matrix.
            params = []
            names = []
            for name in fitted_model.param_names:
                # Check if it is fixed:
                if fitted_model.fixed[name]:
                    continue

                param = getattr(fitted_model, name)

                if param.quantity is not None:
                    params.append(param.quantity)
                else:
                    # Assign a dimensionless unit
                    params.append(param.value * u.dimensionless_unscaled)

                names.append(name)

            self._radprof_params = params
            npar = len(self.radprof_params)

            self._radprof_parnames = names

            self._radprof_model = fitted_model
            self._radprof_fitter = fitter

            # Fail checks
            fail_flag = False

            param_cov = fitter.fit_info.get('param_cov')
            if param_cov is not None:
                fit_uncert = list(np.sqrt(np.diag(param_cov)))
            else:
                fit_uncert = [np.NaN] * npar
                fail_flag = True

            if len(fit_uncert) != len(params):
                raise ValueError("The number of parameters does not match the "
                                 "number from the covariance matrix. Check for"
                                 " fixed parameters.")

            # Add units to errors
            for i, par in enumerate(params):
                fit_uncert[i] = fit_uncert[i] * par.unit

            self._radprof_errors = fit_uncert

            # Check if units should be kept
            if fitted_model._supports_unit_fitting:
                modvals = fitted_model(dist)
                radprof_vals = radprof
            else:
                modvals = fitted_model(dist.value)
                radprof_vals = radprof.value

            chisq = red_chisq(radprof_vals, modvals, npar, 1)
            if chisq > chisq_max:
                fail_flag = True

        if (skip_fitting or fail_flag) and try_nonparam:
            fit, fit_error, fail_flag = \
                nonparam_width(dist.value, radprof.value,
                               unbin_dist, unbin_radprof,
                               None, 5, 99)
            self._radprof_type = 'nonparam'

            # Make the equivalent Gaussian model w/ a background
            self._radprof_model = Gaussian1D() + Const1D()
            if self._radprof_model._supports_unit_fitting:
                self._radprof_model.amplitude_0 = fit[0] * yunit
                self._radprof_model.mean_0 = 0.0 * xunit
                self._radprof_model.sigma_0 = fit[1] * xunit
                self._radprof_model.amplitude_1 = fit[2] * yunit
            else:
                self._radprof_model.amplitude_0 = fit[0]
                self._radprof_model.mean_0 = 0.0
                self._radprof_model.sigma_0 = fit[1]
                self._radprof_model.amplitude_1 = fit[2]

            # Slice out the FWHM and add units
            params = [fit[0] * yunit, fit[1] * xunit, fit[2] * yunit]
            errs = [fit_error[0] * yunit, fit_error[1] * xunit,
                    fit_error[2] * yunit]
            self._radprof_params = params
            self._radprof_errors = errs
            self._radprof_parnames = ['amplitude_0', 'stddev_0', 'amplitude_1']

        if fwhm_function is not None:
            fwhm = fwhm_function(fitted_model)
        else:
            # Default to Gaussian FWHM
            for idx, name in enumerate(self.radprof_parnames):
                if "stddev" in name:
                    found_width = True
                    break

            if found_width:
                fwhm = self.radprof_params[idx].value * np.sqrt(8 * np.log(2)) * xunit
                fwhm_err = self.radprof_errors[idx].value * np.sqrt(8 * np.log(2)) * xunit
            else:
                raise ValueError("Could not automatically identify which "
                                 "parameter in the model corresponds to the "
                                 "width. Please pass a function to "
                                 "'fwhm_function' to identify the width "
                                 "parameter.")

        if deconvolve_width:
            fwhm_deconv_sq = fwhm**2 - beamwidth**2
            if fwhm_deconv_sq > 0:
                fwhm_deconv = np.sqrt(fwhm_deconv_sq)
                fwhm_deconv_err = fwhm * fwhm_err / fwhm_deconv
            else:
                fwhm_deconv = np.NaN
                fwhm_deconv_err = np.NaN
                warnings.warn("Width could not be deconvolved from the beam "
                              "width.")
        else:
            fwhm_deconv = fwhm
            fwhm_deconv_err = fwhm_err

        self._fwhm = fwhm_deconv
        self._fwhm_err = fwhm_deconv_err

        # Final width check -- make sure length is longer than the width.
        # If it is, add the width onto the length since the adaptive
        # thresholding shortens each edge by the about the same.
        if self.length() < self._fwhm:
            fail_flag = True

        # Add the width onto the length if enabled
        if add_width_to_length:
            if fail_flag:
                warnings.warn("Ignoring adding the width to the length because"
                              " the fail flag was raised for the fit.")
            else:
                self._length += self._fwhm

        self._radprof_failflag = fail_flag

    @property
    def radprof_fit_fail_flag(self):
        '''
        Flag to catch poor fits.
        '''
        return self._radprof_failflag

    @property
    def radprof_type(self):
        '''
        The model type used to fit the radial profile.
        '''
        return self._radprof_type

    @property
    def radprofile(self):
        '''
        The binned radial profile created in `~FilFinder2D.width_analysis`.
        This contains the distances and the profile value in the distance bin.
        '''
        return self._radprofile

    @property
    def radprof_params(self):
        '''
        Fit parameters from `~FilFinder2D.width_analysis`.
        '''
        return self._radprof_params

    @property
    def radprof_errors(self):
        '''
        Fit uncertainties from `~FilFinder2D.width_analysis`.
        '''
        return self._radprof_errors

    def radprof_fwhm(self, unit=u.pixel):
        '''
        The FWHM of the fitted radial profile and its uncertainty.

        Parameters
        ----------
        unit : `~astropy.units.Unit`, optional
            Pixel, angular, or physical unit to convert to.
        '''
        return self._converter.from_pixel(self._fwhm, unit), \
            self._converter.from_pixel(self._fwhm_err, unit)

    @property
    def radprof_parnames(self):
        '''
        Parameter names from `~FilFinder2D.radprof_model`.
        '''
        return self._radprof_parnames

    def radprof_fit_table(self, unit=u.pix):
        '''
        Return an `~astropy.table.Table` with the fit parameters and
        uncertainties.

        Parameters
        ----------
        unit : `~astropy.units.Unit`, optional
            Pixel, angular, or physical unit to convert to.
        '''

        from astropy.table import Table, Column

        tab = Table()

        for name, val, err in zip(self.radprof_parnames, self.radprof_params,
                                  self.radprof_errors):

            # Try converting to the given unit. Assume failures are not length
            # units.
            try:
                conv_val = self._converter.from_pixel(val, unit)
                conv_err = self._converter.from_pixel(err, unit)
            except u.UnitsError:
                conv_val = val
                conv_err = err

            tab[name] = Column(conv_val.reshape((1,)))
            tab[name + "_err"] = Column(conv_err.reshape((1,)))

        # Add on the FWHM
        tab['fwhm'] = Column(self.radprof_fwhm(unit)[0].reshape((1,)))
        tab['fwhm_err'] = Column(self.radprof_fwhm(unit)[1].reshape((1,)))

        # Add on whether the fit was "successful"
        tab['fail_flag'] = Column([self.radprof_fit_fail_flag])

        # Add the type of fit based on the model type
        tab['model_type'] = Column([self.radprof_type])

        return tab

    @property
    def radprof_model(self):
        '''
        The fitted radial profile model.
        '''
        return self._radprof_model

    def plot_radial_profile(self, save_name=None, xunit=u.pix,
                            ax=None):
        '''
        Plot the radial profile of the filament and the fitted model.

        Parameters
        ----------
        xunit : `~astropy.units.Unit`, optional
            Pixel, angular, or physical unit to convert to.
        ax : `~matplotlib.axes`, optional
            Use an existing set of axes to plot the profile.
        '''

        dist, radprof = self.radprofile

        model = self.radprof_model

        conv_dist = self._converter.from_pixel(dist, xunit)

        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.subplot(111)

        ax.plot(conv_dist, radprof, "kD")
        points = np.linspace(np.min(dist),
                             np.max(dist), 5 * len(dist))
        # Check if units should be kept when evaluating the model
        if not model._supports_unit_fitting:
            points = points.value

        conv_points = np.linspace(np.min(conv_dist),
                                  np.max(conv_dist), 5 * len(conv_dist))
        ax.plot(conv_points, model(points), "r")
        ax.set_xlabel(r'Radial Distance ({})'.format(xunit))
        ax.set_ylabel(r'Intensity ({})'.format(self._yunit))
        ax.grid(True)

        plt.tight_layout()

        if save_name is not None:
            plt.savefig(save_name)

        plt.show()
        if in_ipynb():
            plt.clf()

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
            The total intensity for the filament.
        '''

        within_fwhm = self._unbin_radprofile[0] <= \
            0.5 * self.radprof_fwhm()[0]
        total_intensity = np.sum(self._unbin_radprofile[1][within_fwhm])

        if bkg_subtract:
            bkg = self.radprof_params[bkg_mod_index]
            if not self.radprof_model._supports_unit_fitting:
                bkg = bkg.value * total_intensity.unit

            total_intensity -= bkg * within_fwhm.sum()

        return total_intensity

    def model_image(self, max_radius=20 * u.pix, bkg_subtract=True,
                    bkg_mod_index=2):
        '''
        Return a model image from the radial profile fit.

        Parameters
        ----------
        max_radius : `~astropy.units.Quantity`, optional
            Set the radius to compute the model to. The outputted array
            will be padded by the number of pixels the max_radius corresponds
            to.
        bkg_subtract : bool, optional
            Subtract off the fitted background level.
        bkg_mod_index : int, optional
            Indicate which element in `Filament2D.radprof_params` is the
            background level. Defaults to 2 for the Gaussian with background
            model.

        Returns
        -------
        model_array : `~astropy.units.Quantity`
            A 2D array computed using the radial profile model.
        '''

        max_radius = self._converter.to_pixel(max_radius).value

        pad_size = int(max_radius)
        skel_arr = self.skeleton(pad_size)

        dists = nd.distance_transform_edt(~skel_arr)
        if self.radprof_model._supports_unit_fitting:
            dists = dists * u.pix

        if not bkg_subtract:
            return self.radprof_model(dists)
        else:
            bkg = self.radprof_params[bkg_mod_index]
            if not self.radprof_model._supports_unit_fitting:
                bkg = bkg.value
            return self.radprof_model(dists) - bkg

    def median_brightness(self, image):
        '''
        Return the median brightness along the skeleton of the filament.

        Parameters
        ----------
        image : `~numpy.ndarray` or `~astropy.units.Quantity`
            The image from which the filament was extracted.

        Returns
        -------
        median_brightness : float or `~astropy.units.Quantity`
            Median brightness along the skeleton.
        '''
        pad_size = 1

        # Do we need to pad the image before slicing?
        input_image = pad_image(image, self.pixel_extents, pad_size)

        skels = self.skeleton(pad_size=pad_size)

        # If the padded image matches the mask size, don't need additional
        # slicing
        if input_image.shape != skels.shape:
            input_image = self.image_slicer(input_image, skels.shape,
                                            pad_size=pad_size)

        assert input_image.shape == skels.shape

        return np.nanmedian(input_image[skels])

    def ridge_profile(self, image):
        '''
        Return the image values along the longest path extent of a filament, or
        from radial slices along the longest path.

        Parameters
        ----------
        image : `~numpy.ndarray` or `~astropy.units.Quantity`
            The image from which the filament was extracted.
        '''

        pad_size = 1

        # Do we need to pad the image before slicing?
        input_image = pad_image(image, self.pixel_extents, pad_size) * \
            u.dimensionless_unscaled

        skels = self.skeleton(pad_size=pad_size, out_type='longpath')

        # If the padded image matches the mask size, don't need additional
        # slicing
        if input_image.shape != skels.shape:
            input_image = self.image_slicer(input_image, skels.shape,
                                            pad_size=pad_size)

        # These should have the same shape now.
        assert input_image.shape == skels.shape

        from .width_profiles.profile_line_width import walk_through_skeleton

        order_pts = walk_through_skeleton(skels)

        if hasattr(image, 'unit'):
            unit = image.unit
        else:
            unit = u.dimensionless_unscaled
            input_image = input_image * unit

        values = []
        for pt in order_pts:
            values.append(input_image[pt[0], pt[1]].value)

        return values * unit

    def profile_analysis(self, image, max_dist=20 * u.pix,
                         num_avg=3, xunit=u.pix):
        '''
        Create profiles of radial slices along the longest path skeleton.
        Profiles created from `~fil_finder.width_profiles.filament_profile`.

        .. note::
            Does not include fitting to the radial profiles. Limited fitting
            of Gaussian profiles is provided in
            `~fil_finder.width_profiles.filament_profile`. See a dedicated
            package like `radfil <https://github.com/catherinezucker/radfil>`_
            for modeling profiles.

        Parameters
        ----------
        image : `~numpy.ndarray` or `~astropy.units.Quantity`
            The image from which the filament was extracted.
        max_dist : astropy Quantity, optional
            The angular or physical (when distance is given) extent to create
            the profile away from the centre skeleton pixel. The entire
            profile will be twice this value (for each side of the profile).
        num_avg : int, optional
            Number of points before and after a pixel that is used when
            computing the normal vector. Using at least three points is
            recommended due to small pixel instabilities in the skeletons.

        Returns
        -------
        dists : `~astropy.units.Quantity`
            Distances in the radial profiles from the skeleton. Units set by
            `xunit`.
        profiles : `~astropy.units.Quantity`
            Radial image profiles.
        '''

        from .width_profiles import filament_profile

        max_dist = self._converter.to_pixel(max_dist)
        pad_size = int(max_dist.value)

        # Do we need to pad the image before slicing?
        input_image = pad_image(image, self.pixel_extents, pad_size)

        if hasattr(image, 'unit'):
            input_image = input_image * image.unit
        else:
            input_image = input_image * u.dimensionless_unscaled

        skels = self.skeleton(pad_size=pad_size, out_type='longpath')

        # If the padded image matches the mask size, don't need additional
        # slicing
        if input_image.shape != skels.shape:
            input_image = self.image_slicer(input_image, skels.shape,
                                            pad_size=pad_size)

        # Check if angular conversions are defined. If not, stay in pixel units
        if hasattr(self._converter, '_ang_size'):
            pixscale = self._converter.to_angular(1 * u.pix)
            ang_conv = True
        else:
            pixscale = 1.0 * u.deg
            ang_conv = False

        dists, profiles = filament_profile(skels, input_image, pixscale,
                                           max_dist=max_dist,
                                           distance=None,
                                           fit_profiles=False,
                                           bright_unit=input_image.unit)

        # First put the distances into pixel units
        if ang_conv:
            dists = [self._converter.to_pixel(dist) for dist in dists]
        else:
            # Already in pixel units.
            dists = [dist.value * u.pix for dist in dists]

        # Convert the distance units
        dists = [self._converter.from_pixel(dist, xunit) for dist in dists]

        return dists, profiles

    def radprof_table(self, xunit=u.pix):
        '''
        Return the radial profile as a table.

        Parameters
        ----------
        xunit : `~astropy.units.Unit`, optional
            Spatial unit to convert radial profile distances.

        Returns
        -------
        tab : `~astropy.table.Table`
            Table with the radial profile distance and values.
        '''
        from astropy.table import Column, Table

        dists = Column(self._converter.from_pixel(self._radprofile[0], xunit))
        vals = Column(self._radprofile[1])

        tab = Table()
        tab['distance'] = dists
        tab['values'] = vals

        return tab

    def branch_table(self, include_rht=False):
        '''
        Save the branch properties of the filament.

        Parameters
        ----------
        include_rht : bool, optional
            If `branches=True` is used in `Filament2D.exec_rht`, the branch
            orientation and curvature will be added to the table.

        Returns
        -------
        tab : `~astropy.table.Table`
            Table with the branch properties.
        '''

        from astropy.table import Table, Column

        branch_data = self.branch_properties.copy()
        del branch_data['pixels']
        del branch_data['number']

        if include_rht:
            branch_data['orientation'] = self.orientation_branches
            branch_data['curvature'] = self.curvature_branches

        tab = Table([Column(branch_data[key]) for key in branch_data],
                    names=branch_data.keys())
        return tab

    def save_fits(self, savename, image, pad_size=20 * u.pix, header=None,
                  **model_kwargs):
        '''
        Save a stamp of the image centered on the filament, the skeleton,
        the longest path skeleton, and the model.

        Parameters
        ----------
        image : `~numpy.ndarray` or `~astropy.units.Quantity`
            The image from which the filament was extracted.
        pad_size : `~astropy.units.Quantity`, optional
            Size to pad the saved arrays by.
        header : `~astropy.io.fits.Header`, optional
            Provide a FITS header to save to. If `~Filament2D` was
            given WCS information, this will be used if no header is given.
        model_kwargs : Passed to `~Filament2D.model_image`.

        '''

        pad_size = int(self._converter.to_pixel(pad_size).value)

        # Do we need to pad the image before slicing?
        input_image = pad_image(image, self.pixel_extents, pad_size)

        skels = self.skeleton(pad_size=pad_size, out_type='all')
        skels_lp = self.skeleton(pad_size=pad_size, out_type='longpath')

        # If the padded image matches the mask size, don't need additional
        # slicing
        if input_image.shape != skels.shape:
            input_image = self.image_slicer(input_image, skels.shape,
                                            pad_size=pad_size)

        model = self.model_image(max_radius=pad_size * u.pix,
                                 **model_kwargs)
        if hasattr(model, 'unit'):
            model = model.value

        from astropy.io import fits
        import time

        if header is None:
            if hasattr(self._converter, "_wcs"):
                header = self._converter._wcs.to_header()
            else:
                header = fits.Header()

        # Strip off units if the image is a Quantity
        if hasattr(input_image, 'unit'):
            input_image = input_image.value.copy()

        hdu = fits.PrimaryHDU(input_image, header)

        skel_hdr = header.copy()
        skel_hdr['BUNIT'] = ("", "bool")
        skel_hdr['COMMENT'] = "Skeleton created by fil_finder on " + \
            time.strftime("%c")

        skel_hdu = fits.ImageHDU(skels.astype(int), skel_hdr)

        skel_lp_hdu = fits.ImageHDU(skels_lp.astype(int), skel_hdr)

        model_hdu = fits.ImageHDU(model, header)

        hdulist = fits.HDUList([hdu, skel_hdu, skel_lp_hdu, model_hdu])

        hdulist.writeto(savename)

    def to_pickle(self, savename):
        '''
        Save a Filament2D class as a pickle file.

        Parameters
        ----------
        savename : str
            Name of the pickle file.
        '''

        with open(savename, 'wb') as output:
                pickle.dump(self, output, -1)

    @staticmethod
    def from_pickle(filename):
        '''
        Load a Filament2D from a pickle file.

        Parameters
        ----------
        filename : str
            Name of the pickle file.
        '''
        with open(filename, 'rb') as input:
                self = pickle.load(input)

        return self


class Filament3D(FilamentNDBase):
    """docstring for Filament3D"""
    def __init__(self, arg):
        super(Filament3D, self).__init__()
        self.arg = arg

