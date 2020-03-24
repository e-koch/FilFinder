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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Delete old handlers
if (logger.hasHandlers()): 
    logger.handlers.clear()
    
logger.propagate = False
#Creating File Output
                        # Will go here eventually
# Creat console handler
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# Create formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%d-%m-%Y %I:%M:%S')
# Add to handlers
                        # Add to File output 
ch.setFormatter(formatter)
# Add handlers to the logger
logger.addHandler(ch)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

class FilFinder3D():
    
    def __init__(self, cube, mask=None):
        
        logger.info('Setting input cube to self.cube')
        self.cube = cube
        
        if mask is not None:
            logger.info('Input mask detected. Setting Mask here.')
            self.mask = mask
    
    def create_mask(self):
        
        # Global Threshold
        self.mask = self.cube > 0.4
        logger.info('Setting Global Threshold to %.1f', 0.4)
        logger.info('Mask Created')
        
        return
    
    def create_skeleton(self):
        '''
        Creates skeleton from 3D filament mask.
        '''
        
        selem = mo.ball(3)
        dilate = mo.dilation(self.mask, selem)
        close = mo.closing(dilate)
        self.d_skel = mo.skeletonize_3d(close)
        
        # Converting skeleton to graph
        
        self.pixel_graph, self.coordinates, self.degrees = csr.skeleton_to_csgraph(self.d_skel)
        
        return
    
    def create_network(self):
        
        G = nx.from_scipy_sparse_matrix(self.pixel_graph)

        for node in G.nodes:
            G.nodes[node]['pos'] = self.coordinates[node]
            
        logging.debug("Number of nodes before Pruning: %f",
                      nx.number_of_nodes(G))
        
        self.network = G
        
        logging.info('Network Created')
        
        return G
    
    def subgraph(self):
        
        # Gathering Subgraph
        self.subgraph = [self.network.subgraph(c) for c \
                         in nx.connected_components(self.network)]
        logger.debug('Subgraph set as list of %1.0f graphs',
                     len(self.subgraph))
        
        # Debug Info Here
        for i, sg in enumerate(self.subgraph):
            logger.debug("Subgraph %1.0f has %1.0f nodes.",
                         i,
                         sg.number_of_nodes())
        
        # Creating a list of nodes from subgraph
        self.subgraphlist = [sg.nodes() for sg in self.subgraph]
        logger.debug('Subgraphlist set as list of %1.0f graphs',
                     len(self.subgraphlist))
        
        #Creating a new graph composed of all subgraph nodes
        self.connected_graph = nx.Graph()
        logger.info('connected_graph made')
        for j in self.subgraphlist:
            self.connected_graph.add_nodes_from(j)
                
        logger.debug('Number of connected nodes: %1.0f', 
                      len(self.connected_graph.nodes))
            
        return
    
    def plot_3D(self):
        
        out = np.where(self.d_skel)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(out[0],out[1],out[2], 'b.', alpha=0.3)
        
        return
    
    def plot_data_mask_slice(self, slice_number):
        
        fig, axs = plt.subplots(2,1)
        
        axs[0].imshow(self.cube[slice_number])
        axs[1].imshow(self.mask[slice_number])
        
"""
    DATA_DIR = 'ngc4321_subset.fits'

    with fits.open(DATA_DIR) as hdul:
        data = hdul[0].data

    # Creating Class object
    C = FilFinder3D(data)
    
    #Creating Mask and Skeleton
    C.create_mask()
    C.create_skeleton()
    
    # Creating New Networkx object
    C.create_network()
"""  