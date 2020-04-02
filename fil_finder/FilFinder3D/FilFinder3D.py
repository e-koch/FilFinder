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
        
        #Removing NaNs in copy
        flat_copy[np.isnan(flat_copy)] = 0.0
        
        #Just using global threshold for now
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
        
        #Dilate the image
        dilate = mo.dilation(self.mask, selem)
        
        # Removing dark spots and small bright cracks in image
        close = mo.closing(dilate)
        
        # Creating Skeleton
        self.skeleton = mo.skeletonize_3d(close)
        
        # Converting skeleton to graph
        self.pixel_graph, self.coordinates, self.degrees = csr.skeleton_to_csgraph(self.skeleton)
        
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
        for node in self.network.nodes:
            self.network.nodes[node]['pos'] = self.coordinates[node]
        
        # Creating Subgraphlist
        self.subgraph_list = self.subgraph_list_maker()
        
        return
    
    def subgraph_list_maker(self):
        """
        Generates list of connected component subgraphs of network.
        """
        return [self.network.subgraph(c) for c in \
                nx.connected_components(self.network)]
    

    def longest_path(self, graph):
        """
        Finds the longest path from endnode to endnode in the input
        graph given.
        
        Parameters
        -------
        graph : Networkx.graph()

        Returns
        -------
        paths : list
            List of trimmed possible paths found.
        longest_path : list
            Longest path of paths found.

        """
        
        # Checking for Single Isolated Node with 0 degree
        for node in graph:
            if graph.degree(node) == 0:
                #TODO May need to change this output for later
                return None, None
            
        paths = []
        endpoints = []
        # Finding endpoints in subgraph
        for node in graph:
            if graph.degree(node) == 1:
                endpoints.append(node)
        
        # Looping all endpoint to endpoint possible paths
        for k in endpoints:
            for i in endpoints:
                if i != k:
                    
                    # Grabbing paths from k to i
                    path = list(nx.all_simple_paths(graph, k, i))
                    
                    #TODO Simpler way to do this?
                    # If there is more than one path found
                    if len(path) > 1:
                        for i in path:
                            paths.append(i)
                    else:
                        paths.append(path[0]) # Only one path found
                else:
                    # If k and i are the same, no path, continue.
                    continue
        
        # Trimming duplicate paths out of the list
        for i in paths:
            for ind, j in enumerate(paths):
                if i != j:
                    if i == j[::-1]: # Checking if the opposite is same
                        del paths[ind] # Delete if true
        
        #TODO Is this the right way to do this? Discussion.
        longest_path = max(paths, key=len)
        
        return paths, longest_path

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
            #Looping to the second last entry of node_list
            if ind < len(node_list) -1:
                edge = (node_list[ind], node_list[ind+1])
                edges.append(edge)
            
        return edges
    
    
    def edge_pruner(self, graph, edges):
        """
        Takes the inpuyt graph object and prunes any edges that
        are not in the edges input list.

        Parameters
        ----------
        graph : Networkx.Graph()
        edges : list

        Returns
        -------
        graph : Networkx.Graph()

        """
        for edge in graph.edges:
            # Checking both forward and backward edge.
            if (edge in edges) or (tuple(reversed(edge)) in edges):
                continue
            else:
                # If not found, remove.
                graph.remove_edge(edge[0], edge[1])
        
        return graph
    
    def node_pruner(self, graph, path):
        """
        Takes the input graph object and prunes any nodes not found 
        in path input list.
    
        Parameters
        ----------
        graph : Networkx.Graph()
        path : list
    
        Returns
        -------
        graph : Networkx.Graph()
        
        """
        
        # TODO
        # Switch to cleaner list comprehension
        #[exp(i) for i in list if filter(i)]
        
        cutnodes = []
        for node in graph.nodes:
            if node in path:
                continue
            else:
                cutnodes.append(node)
        
        graph.remove_nodes_from(cutnodes)
        
        return graph
    
    def pruning_wrapper(self):
        """
        Takes the attribute subgraph_list, and returns a new subset
        of graph objects that only contain the longets path found
        for each individual graph in subgraph_list.

        Returns
        -------
        graphs_list : list
            List of Networkx.Graph() objects.

        """
        graphs_list = []
        for i in self.subgraph_list:
            
            # Creating New graph to modify
            G = nx.Graph(i)

            # Gathering longest path and edges to keep
            paths, long_path = self.longest_path(G)
            
            # Check if NoneType was returned, continue if so.
            if paths == None and long_path == None:
                continue
            
            edges = self.edge_builder(long_path)
            
            # Pruning off edges and nodes
            G = self.edge_pruner(G, edges)
            G = self.node_pruner(G, long_path)
            
            # Appending modified graph to main list
            graphs_list.append(G)
        
        #TODO Add attribute to class, find a name for it.
        
        return graphs_list
    
    
    def plot_3D(self):
        """
        Plots skeleton in 3D space.
        """
        out = np.where(self.skeleton)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(out[0],out[1],out[2], 'b.', alpha=0.3)
        
        return
    
    def plot_data_mask_slice(self, slice_number):
        """
        Plots slice of mask, alongside image.

        Parameters
        ----------
        slice_number : int
            Array indice in major axis to slice 3D set.

        """
        fig, axs = plt.subplots(2,1)
        
        axs[0].imshow(self.image[slice_number])
        axs[1].imshow(self.mask[slice_number])


