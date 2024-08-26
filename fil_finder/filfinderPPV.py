#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 22:07:25 2020

@author: samuelfielder
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import skimage.morphology as mo
import networkx as nx
import warnings
import astropy.units as u

from .filament import FilamentPPV
from .skeleton3D import Skeleton3D
from .threshold_local_3D import threshold_local


class FilFinderPPV(Skeleton3D):
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

    def __init__(self, image, mask=None, save_name='FilFinderPPV_output'):

        # Add warning that this is under development
        warnings.warn("This algorithm is under development. Not all features are implemented"
                      " or tested. Use with caution.")

        self._has_skan()

        self.image = image

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

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, value):
        if not value.ndim == 3:
            raise ValueError(f"The array must be 3D. Given a {value.ndim} array.")

        self._image = value

    def preprocess_image(self, skip_flatten=False,
                         flatten_percent=85):
        """
        Preprocess and flatten the dataset before running the masking process.

        Parameters
        ----------
        skip_flatten : bool, optional
            Skip the flattening process and use the original image to
            construct the mask. Default is False.
        flatten_percent : int, optional
            The percentile of the data (0-100) to set the normalization.
            Default is 85th.

        """
        if skip_flatten:
            self._flatten_threshold = None
            self.flat_img = self._image

        else:
            # TODO Add in here
            thresh_val = np.nanpercentile(self.image,
                                          flatten_percent)

            # self._flatten_threshold = data_unit_check(thresh_val,
            #                                           self.image.unit)
            self._flatten_threshold = thresh_val

            # Make the units dimensionless
            self.flat_img = thresh_val * np.arctan(self.image / self._flatten_threshold)
            # / u.rad



    def create_mask(self, adapt_thresh=9, glob_thresh=0.0,
                    selem_disc_radius=2, selem_spectral_width=1,
                    min_object_size=27*3,
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

        warnings.warn("Units are not yet supported for kwargs. Please provide pixel units",
                      UserWarning)

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
        thresh_image = threshold_local(flat_copy, adapt_thresh, **adapt_kwargs)
        if hasattr(flat_copy, 'unit'):
            thresh_image = thresh_image * flat_copy.unit

        adapt_mask = flat_copy > thresh_image

        # Add in global threshold mask
        adapt_mask = np.logical_and(adapt_mask, flat_copy > glob_thresh)

        selem = mo.disk(selem_disc_radius)
        if selem_spectral_width > 1:
            selem = np.tile(selem, (selem_spectral_width, 1, 1))
        else:
            selem = selem[np.newaxis, ...]

        # Dilate the image
        # dilate = mo.dilation(adapt_mask, selem)

        # NOTE: Look into mo.diameter_opening and mo.diameter_closing
        dilate = mo.opening(adapt_mask, selem)

        # Removing dark spots and small bright cracks in image
        close = mo.closing(dilate)

        # Don't allow small holes: these lead to "shell"-shaped skeleton features
        # Specify out to avoid duplicating the mask in memory
        close = mo.remove_small_objects(close,
                                        min_size=min_object_size,
                                        connectivity=1,
                                        out=close)
        close = mo.remove_small_holes(close,
                                      area_threshold=max_hole_size,
                                      connectivity=1,
                                      out=close)

        self.mask = close

    def analyze_skeletons(self, compute_longest_path=True,
                          do_prune=True,
                          verbose=False, save_png=False,
                          save_name=None, prune_criteria='all',
                          relintens_thresh=0.2,
                          max_prune_iter=10,
                          branch_spatial_thresh=0 * u.pix,
                          branch_spectral_thresh=0 * u.pix,
                          test_print=0):
        '''
        '''

        self._compute_longest_path = compute_longest_path

        # Define the skeletons

        if not hasattr(self, '_skel_labels'):
            raise ValueError("Run create_skeleton() before analyze_skeletons()")

        num = self._skel_labels.max()

        self.filaments = []

        for i in range(1, num + 1):

            coords = np.where(self._skel_labels == i)

            self.filaments.append(FilamentPPV(coords,))
                                            #   converter=self.converter))

        # Calculate lengths and find the longest path.
        # Followed by pruning.
        for num, fil in enumerate(self.filaments):
            if test_print:
                print(f"Skeleton analysis for {num} of {len(self.filaments)}")

            fil._make_skan_skeleton(self._image)

            fil.skeleton_analysis(self._image,
                                  compute_longest_path=compute_longest_path,
                                  verbose=verbose, save_png=save_png,
                                  save_name=save_name, prune_criteria=prune_criteria,
                                  relintens_thresh=relintens_thresh,
                                  max_prune_iter=max_prune_iter,
                                  branch_spatial_thresh=branch_spatial_thresh,
                                  branch_spectral_thresh=branch_spectral_thresh,
                                  test_print=test_print)

        # Check for now empty skeletons:
        del_fil_inds = [ii for ii, fil in enumerate(self.filaments) if fil.pixel_coords[0].size == 0]
        for ii in sorted(del_fil_inds, reverse=True):
            self.filaments.pop(ii)

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

    def network_plot_3D(self, filament=None, angle=40, filename='plot.pdf', save=False):
        '''
        Gives a 3D plot for networkX using coordinates information of the nodes

        Parameters
        ----------
        filament : Filament
            Filament object or list of objects from `self.filaments`. The default is None
            and will plot all the filaments in the network.
        angle : int
            Angle to view the graph plot
        filename : str
            Filename to save the plot
        save : bool
            boolen value when true saves the plot

        '''

        if filament is None:
            filament = self.filaments

        # 3D network plot
        with plt.style.context(('ggplot')):

            fig = plt.figure(figsize=(10, 7))
            ax = Axes3D(fig)

            for this_filament in filament:

                G = this_filament.graph

                # Get node positions
                pos = nx.get_node_attributes(G, 'pos')

                # Get the maximum number of edges adjacent to a single node
                edge_max = max([G.degree(i) for i in G.nodes])

                # Define color range proportional to number of edges adjacent to a single node
                # colors = [plt.cm.plasma(G.degree(i) / edge_max) for i in G.nodes]

                # Loop on the pos dictionary to extract the x,y,z coordinates of each node
                for i, (key, value) in enumerate(pos.items()):
                    xi = value[0]
                    yi = value[1]
                    zi = value[2]

                    # Scatter plot
                    ax.scatter(xi, yi, zi, # c=colors[i],
                               s=20 + 20 * G.degree(key),
                               edgecolors='k', alpha=0.7)

                # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
                # Those two points are the extrema of the line to be plotted
                for i, j in enumerate(G.edges()):

                    x = np.array((pos[j[0]][0], pos[j[1]][0]))
                    y = np.array((pos[j[0]][1], pos[j[1]][1]))
                    z = np.array((pos[j[0]][2], pos[j[1]][2]))

                    # Plot the connecting lines
                    ax.plot(x, y, z, c='black', alpha=0.5)

            # Set the initial view
            ax.view_init(30, angle)

            # Hide the axes
            # ax.set_axis_off()

            if save is not False:
                plt.savefig(filename)
                plt.close('all')
            else:
                plt.show()

        return


    def network_plot_3D_plotly(self, filament=None, angle=40, filename='plot.pdf', save=False):
        '''
        Gives a 3D plot for networkX using coordinates information of the nodes

        Parameters
        ----------
        filament : Filament
            Filament object or list of objects from `self.filaments`. The default is None
            and will plot all the filaments in the network.
        angle : int
            Angle to view the graph plot
        filename : str
            Filename to save the plot
        save : bool
            boolen value when true saves the plot

        '''

        import plotly.graph_objects as go

        if filament is None:
            filament = self.filaments

        # 3D network plot
        edge_traces = []
        node_traces = []

        for this_filament in filament:
            G = this_filament.graph

            # Get node positions
            pos = nx.get_node_attributes(G, 'pos')

            x_nodes = [pos[node][0] for node in G.nodes()]
            y_nodes = [pos[node][1] for node in G.nodes()]
            z_nodes = [pos[node][2] for node in G.nodes()]

            # Create a 3D scatter plot for the nodes
            node_trace = go.Scatter3d(
                x=x_nodes,
                y=y_nodes,
                z=z_nodes,
                mode='markers',
                marker=dict(size=5, color='blue')
            )
            node_traces.append(node_trace)

            # Create a 3D line plot for the edges
            edge_x = []
            edge_y = []
            edge_z = []
            for edge in G.edges():
                source, target = edge
                x1, y1, z1 = pos[source]
                x2, y2, z2 = pos[target]
                edge_x.extend([x1, x2, None])
                edge_y.extend([y1, y2, None])
                edge_z.extend([z1, z2, None])

            edge_trace = go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                mode='lines',
                line=dict(color='black')
            )
            edge_traces.append(edge_trace)

        # Create the figure
        fig = go.Figure(data=edge_traces + node_traces)

        fig.show()

        return fig

    def plot_data_mask_slice(self, slice_number,
                             show_flat_img=False,):
        """
        Plots slice of mask, alongside image.

        Parameters
        ----------
        slice_number : int
            Array indice in major axis to slice 3D set.

        """
        fig, axs = plt.subplots(2, 1)

        axs[0].imshow(self.image[slice_number])
        axs[0].contour(self.skeleton[slice_number],
                       levels=[0.5], colors='r')
        axs[1].imshow(self.mask[slice_number])
        axs[1].contour(self.skeleton[slice_number],
                       levels=[0.5], colors='r')


