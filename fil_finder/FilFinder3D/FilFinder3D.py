#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 22:07:25 2020

@author: samuelfielder
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skan import csr
import numpy as np
import skimage.morphology as mo
import networkx as nx
import warnings

class FilFinder3D():
    
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
        
        #TODO add image checking here
        self._image = image
        
        self.save_name = save_name
        
        # Mask Initialization
        self.mask = None
        if mask is not None:
            if self.image.shape != mask.shape:
                raise ValueError("The given pre-existing mask must"
                                 " have the same shape as input image." )
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
            #TODO Add in here
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
            #TODO Check if glob_thresh is proper
            self.glob_thresh = glob_thresh

        # Here starts the masking process
        flat_copy = self.flat_img.copy()
        
        # Removing NaNs in copy
        flat_copy[np.isnan(flat_copy)] = 0.0
        
        # Just using global threshold for now
        self.mask = flat_copy > glob_thresh
        
        if verbose or save_png:
            plt.clf()
            plt.imshow(self.flat_img.value)
            plt.contour(self.mask)
            plt.title("Mask on Flattened Image.")
            
            if save_png:
                plt.savefig(self.save_name + '_mask.png')
            if verbose:
                plt.show()
            
            #TODO What is this?
            #if in_ipynb():
                #plt.clf()
        
        return
    
    def create_skeleton(self, ball_radius=3):
        """
        
        Parameters
        ----------
        ball_radius : int, optional
            Amount of pixels that ball object is used
            for dilation of mask. The default is 3.
            
        Attributes
        ----------
        skeleton : numpy.ndarray
            Thinned skeleton image
        pixel_graph : sparse.csr_matrix
            The value graph[i,j] is the distance between adjacent pixels 
            i and j.
        coodinates : numpy.ndarray
            Mapping indices in pixel_graph to pixel coordinates
            
        """
        #TODO should we use other shape here?
        # Create slider object
        selem = mo.ball(ball_radius)
        
        # Dilate the image
        dilate = mo.dilation(self.mask, selem)
        
        # Removing dark spots and small bright cracks in image
        close = mo.closing(dilate)
        
        # Creating Skeleton
        self.skeleton = mo.skeletonize_3d(close)
        
        # Converting skeleton to graph
        self.pixel_graph, self.coordinates, self.degrees = \
            csr.skeleton_to_csgraph(self.skeleton,
                                    unique_junctions=False)
        
        # Re-casting Coordinates into int
        self.coordinates = self.coordinates.astype(int)
        return
    
    def create_network(self):
        """
        Creates the initial network and subgraph_list objects.

        Attributes
        -------
        network : networkx.Graph()
            object that contains the graph representation of pixel_graph
        subgraph_list : list
            Contains graph objects of found connected component graphs
            
        """
        
        self.network = nx.from_scipy_sparse_matrix(self.pixel_graph)
        
        # Appending 3D pixel positions as node attributes
        # Appending 3D Pixel intensity value as node attribute
        for node in self.network.nodes:
            self.network.nodes[node]['pos'] = self.coordinates[node]
            self.network.nodes[node]['data'] = self._image[self.coordinates[node][0],
                                                           self.coordinates[node][1],
                                                           self.coordinates[node][2]]
            
        
        
        
        # Creating Subgraphlist
        self.subgraph_list = [self.network.subgraph(c) for c in \
                              nx.connected_components(self.network)]
            
        #TODO Check for skeleton_threshold for subgraphs -> pre-prune
            

    def longest_path(self, graph):
        """
        Finds the shortest path for every combination of endnode to endnode.
        Returns the longest path from the collection of shortest paths.
        
        Parameters
        -------
        graph : Networkx.graph()

        Returns
        -------
        paths : list
            List of smallest possible paths found.
        longest_path : list
            Longest path of paths found.
        internodes : list
            List of nodes with degree greater than 2
            Represents intersection points

        """
        
        # Checking for Single Isolated Node with 0 degree
        #TODO Won't be needed after pre-pruning is dealt with
        if len(graph) < 2:
            return None, None
        
        #Initialize lists
        paths = []
        endnodes = []
        internodes = []
        
        # Filing Endnodes and internode lists
        for node in graph:
            if graph.degree(node) == 1:
                endnodes.append(node)
            elif graph.degree(node) > 2:
                internodes.append(node)
        
        # Looping all endnodes to endnodes possible paths
        for k in endnodes:
            for i in endnodes:
                if i != k:
                    
                    # Grabbing the shortest(s) path(s) from k to i
                    path = list(nx.all_shortest_paths(graph, k, i))
                    
                    # Add the list of path into paths
                    for i in path:
                        paths.append(i)

                else:
                    # If k and i are the same, no path, continue.
                    continue
        
        # Trimming duplicate paths out of the list
        for i in paths:
            for ind, j in enumerate(paths):
                if i != j:
                    if i == j[::-1]: # Checking if the opposite is same
                        del paths[ind] # Delete if true
        
        #TODO What happens when the longest paths are tied -> tie-breaker
        longest_path = max(paths, key=len)
        
        return paths, longest_path, internodes
    
    def subgraph_analyzer(self, graph):
        '''
        Takes graph object, finds longest shortest path between two endnodes,
        isolates this path as a graph by removing edges on intersections that
        are not part of the longest shortest path. Then creates a list of
        subgraphs that are the branches to be inspected and pruned.

        Parameters
        ----------
        graph : networkx.Graph

        Returns
        -------
        filament : networkx.Graph
        branches : list
            list of networkx.Graph objects which represent branches off
            the longest shortest path

        '''
        
        # Grabbing Longest path alongs with nodes and internodes
        paths, longest_path, internodes = self.longest_path(graph)
        
        # Building a list of edges to check against the main filament
        longest_path_edges = self.edge_builder(longest_path)
        # Building list of nodes in main filament that are internodes
        intersections = [i for i in longest_path if i in internodes]
        
        #Creating new graph to edit
        H = nx.Graph(graph)
        # Looping through each intersection along main filament branch
        # and removing edges that are not part of the main filament branch
        for i in intersections:
            for j in graph.neighbors(i):
                if (i,j) in longest_path_edges or (j,i) in longest_path_edges:
                    pass
                else:
                    H.remove_edge(i, j)
        
        # This leave H being a main filament, along with branches
        # Split the data into different subgraphs
        filaments = [H.subgraph(c) for c in nx.connected_components(H)]
        filaments_lengths = [len(i.nodes) for i in filaments]
        
        #Compute the main filament, and branches
        filament = filaments[filaments_lengths.index(max(filaments_lengths))]
        branches = [i for i in filaments if i is not filament]
        
        return filament, branches

    def edge_builder(self, node_list):
        """
        Builds tuple edges for nodes in given list.
        i.e. Input: [1,2,3] -> Output: [(1,2), (2,3)]

        Parameters
        ----------
        node_list : list
            List of single node numbers.

        Returns
        -------
        edges : list
             Returns a list of tuples that are the edges linking
             input node_list together.

        """
        edges = []
        for ind, val in enumerate(node_list):
            # Looping to the second last entry of node_list
            if ind < len(node_list) -1:
                edge = (node_list[ind], node_list[ind+1])
                edges.append(edge)
            
        return edges
    
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
        colors = [plt.cm.plasma(G.degree(i)/edge_max) for i in G.nodes]
    
        # 3D network plot
        with plt.style.context(('ggplot')):
    
            fig = plt.figure(figsize=(10,7))
            ax = Axes3D(fig)
    
            # Loop on the pos dictionary to extract the x,y,z coordinates of each node
            for i, (key, value) in enumerate(pos.items()):
                xi = value[0]
                yi = value[1]
                zi = value[2]
    
                # Scatter plot
                ax.scatter(xi, yi, zi, c=colors[i], s=20+20*G.degree(key), edgecolors='k', alpha=0.7)
    
            # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
            # Those two points are the extrema of the line to be plotted
            for i,j in enumerate(G.edges()):
    
                x = np.array((pos[j[0]][0], pos[j[1]][0]))
                y = np.array((pos[j[0]][1], pos[j[1]][1]))
                z = np.array((pos[j[0]][2], pos[j[1]][2]))
    
            # Plot the connecting lines
                ax.plot(x, y, z, c='black', alpha=0.5)
    
        # Set the initial view
        ax.view_init(30, angle)
    
        # Hide the axes
        #ax.set_axis_off()
    
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


