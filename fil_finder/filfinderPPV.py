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

from .filament import FilamentPPV
from .skeleton3D import Skeleton3D


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

    def __init__(self, image, mask=None, save_name='FilFinder3D_output'):

        self._has_skan()

        # TODO add image checking here
        self._image = image

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
            return

    def create_mask(self, glob_thresh=None, verbose=False,
                    save_png=False, use_existing_mask=False):
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
            warnings.warn("Using inputted mask. Skipping creation "
                          " of a new mask.")

        if glob_thresh is None:
            self.glob_thresh = None
        else:
            # TODO Check if glob_thresh is proper
            self.glob_thresh = glob_thresh

        # Here starts the masking process
        flat_copy = self.flat_img.copy()

        # Removing NaNs in copy
        flat_copy[np.isnan(flat_copy)] = 0.0

        # Just using global threshold for now
        self.mask = flat_copy > glob_thresh

        return

    def filament_trimmer(self, filament, branches):
        """
        Runs through the branches of the filament and trims based on
        certain inputted criteria.

        Parameters
        ----------
        filament : networkx.Graph
            Associated with the longest path filament found.
        branches : list (of networkx.Graph objects)
            Associated with the branches off the longest path filament.

        Attributes
        -------
        filaments : list
            Will include all the filaments that make it through the trimming
            process of this function. Each index in the list is a Filament3D
            instance.

        """

        # Creating filaments attribute which will hold a list of Filament3D
        # objects that are not trimmed from the criteria
        self.filaments = []

        main_filament = FilamentPPV(filament)
        inspect_branches = [FilamentPPV(x) for x in branches]

        # TODO Code here for testing branches
        # use del in this code to delete the branches from the list as the code runs

        # Add leftover branches and main_filament to the self.filaments attribute
        self.filaments.append(main_filament)
        # Loop through leftover branches here
        for i in inspect_branches:
            self.filaments.append(i)

    @staticmethod
    def network_plot_3D(G, angle=40, filename='plot.pdf', save=False):
        '''
        Credit: Dewanshu
        Gives a 3D plot for networkX using coordinates information of the nodes

        Parameters
        ----------
        G : networkx.Graph
        angle : int
            Angle to view the graph plot
        filename : str
            Filename to save the plot
        save : bool
            boolen value when true saves the plot

        '''
        # Get node positions
        pos = nx.get_node_attributes(G, 'pos')

        # Get the maximum number of edges adjacent to a single node
        edge_max = max([G.degree(i) for i in G.nodes])

        # Define color range proportional to number of edges adjacent to a single node
        colors = [plt.cm.plasma(G.degree(i) / edge_max) for i in G.nodes]

        # 3D network plot
        with plt.style.context(('ggplot')):

            fig = plt.figure(figsize=(10, 7))
            ax = Axes3D(fig)

            # Loop on the pos dictionary to extract the x,y,z coordinates of each node
            for i, (key, value) in enumerate(pos.items()):
                xi = value[0]
                yi = value[1]
                zi = value[2]

                # Scatter plot
                ax.scatter(xi, yi, zi, c=colors[i],
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

    def plot_data_mask_slice(self, slice_number):
        """
        Plots slice of mask, alongside image.

        Parameters
        ----------
        slice_number : int
            Array indice in major axis to slice 3D set.

        """
        fig, axs = plt.subplots(2, 1)

        axs[0].imshow(self.image[slice_number])
        axs[1].imshow(self.mask[slice_number])
