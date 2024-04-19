#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 11:37:32 2020

@author: samuelfielder
"""

import networkx as nx
from astropy.io import fits
from FilFinder3D import FilFinder3D


def test_graph():
    """
    Returns a nx.Graph() object with testing nodes and edges
    
    Returns
    -------
    tg : nx.Graph
        includes nodes and edges for testing cases

    """
    tg = nx.Graph()
    
    tg.add_nodes_from(range(0,12))
    
    tg.add_edges_from([
        (0,1), (1,2), (3,4), (4,5), (3,6), (4,6), (6,7),
        (7,8), (8,10), (8,9)
        ])
    
    return tg

if __name__ == "__main__":
    
    DATA_DIR = '/home/samuelfielder/Documents/Astro/ngc4321_subset.fits'
    
    with fits.open(DATA_DIR) as hdul:
        data = hdul[0].data
        
    Data = FilFinder3D(data)
    
    #Preprocess -> Masking -> Skeleton -> Network
    Data.preprocess_image(skip_flatten=True)
    Data.create_mask(glob_thresh=0.4)
    Data.create_skeleton()
    Data.create_network()
    
    G = Data.subgraph_list[1]
    
    filament, branches = Data.subgraph_analyzer(G)
    
    Data.network_plot_3D(G)
    Data.network_plot_3D(filament)

    
    # network_plot_3D(G,
    #                 50,
    #                 longest_path_edges,
    #                 '/home/samuelfielder/Desktop/Figure_Test.pdf',
    #                 save=True)


    # # Returning subgraphs with longest filament found
    # # Also includes coordinate information on each node
    # graphs = Data.pruning_wrapper()
