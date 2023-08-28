#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 21:06:15 2020

@author: samuelfielder
"""

from astropy.io import fits
from skan import csr
import numpy as np
import skimage.morphology as mo
import networkx as nx

from test_graph import test_graph


def create_skeleton(mask):
    """
    Creates a a CSR matrix, along with coordinates for mapping and degree for 
    nodes.

    Parameters
    ----------
    mask : array

    Returns
    -------
    pixel_graph : sparse.csr_matrix
    coordinates : array
        Maps indices of mask to pixel coordinate in pixel_graph
    degrees : array
        degree of corresponding node in graph

    """
    
    # Choosing the object to pass through input masking image
    selem = mo.ball(3)
    dilate = mo.dilation(mask, selem)
    close = mo.closing(dilate)
    d_skel = mo.skeletonize_3d(close)
    
    # Converting skeleton to graph
    pixel_graph, coordinates, degrees = csr.skeleton_to_csgraph(d_skel)
    
    return pixel_graph, coordinates, degrees
    
def create_network(pixel_graph, coordinates):
    """
    Creating Networkx.Graph() object from pixel graph

    Parameters
    ----------
    pixel_graph : sparse.csr_matrix
    coordinates : array

    Returns
    -------
    G : Networkx.Graph()
    
    """
    
    G = nx.from_scipy_sparse_matrix(pixel_graph)
    
    # Attaches coordinate info to nodes attribute 'pos'
    for node in G.nodes:
        G.nodes[node]['pos'] = coordinates[node]
    
    return G
    

def subgraph_setup(graph):
    """
    Creates a list of individual subgraphs from input graph.

    Parameters
    ----------
    graph : Networkx.Graph()

    Returns
    -------
    subgraph_list : list
        list of individual Networkx.Graph() objects
        
    """
    
    G = nx.Graph(graph)
    
    subgraph_list = [G.subgraph(c) for c in nx.connected_components(G)]
    
    return subgraph_list       

def longest_path(graph):
    """
    Finds the longest path from endnode to endnode in given graph object.
    
    0. Test if graph is one node of degree 0.
    1. Finds all endnodes of graph (degree=1)
    2. Loops for combinations of endnodes, finds all paths
    3. Trims duplicate paths:
        i.e. Path (1,2,3) equivalent to Path (3,2,1), trims the latter.
    4. Finds the longest path out of all possible paths found:
        NOTE: This uses a max(paths) argument, such that if two or more paths
                are found with equivalent distances, the function will pick the
                first found. This may need to be fixed.

    Parameters
    ----------
    graph : Networkx.Graph()

    Returns
    -------
    paths : list
        list of lists of paths from all endnodes to endnodes
    longest_path : list
        longest path fround from endnode to endnode in graph

    """
    
    G = nx.Graph(graph)
    
    # Testing for Single Node graph
    for node in G.nodes:
        if G.degree(node) == 0:
            return None, None
    
    paths = []
    endpoints = []
    # Finding Endnodes
    for node in G.nodes:
        if G.degree(node) == 1:
            endpoints.append(node)
    
    # Gathering all endpoint to endpoint paths
    for k in endpoints:
        for i in endpoints:
            # Filter by differing endpoints
            if i != k:
                # Grabbing all paths from k to i
                path = list(nx.all_simple_paths(G, k, i))
                
                # If there is more than one path found from k to i
                if len(path) > 1:
                    for i in path:
                        paths.append(i)
                else:
                    paths.append(path[0]) # Only one sublist in list
            else:
                continue
    
    # Trimming Duplicate Paths (see Docstring for info)
    for i in paths:
        for index, j in enumerate(paths):
            if i != j:
                if i == j[::-1]:
                    del paths[index]
                    
    # Grab longest path:
    # Will return first longest list if multiple are found with same length
    longest_path = max(paths, key=len)
        
    return paths, longest_path
        
def keep_edges(node_list):
    """
    Generates a list of tuples whos entries are the edges that will connect
    the nodes given by the list object path.

    Parameters
    ----------
    node_list : list

    Returns
    -------
    edges : list

    """
    edges = []
    for i, val in enumerate(node_list):
        if i < len(node_list) -1:
            edge = (node_list[i], node_list[i+1])
            edges.append(edge)
            
    return edges

def edge_prune(graph, edges):
    """
    Takes the input graph object and prunes any edge objects that are not in
    the edges input list.

    Parameters
    ----------
    graph : Networkx.Graph()
    edges : list

    Returns
    -------
    G : Networkx.Graph()
    
    """
    
    G = nx.Graph(graph)
    
    for edge in G.edges:
        # Must also force tuple for reversed edge
        if (edge in edges) or (tuple(reversed(edge)) in edges):
            continue
        else:
            print(edge)
            G.remove_edge(edge[0], edge[1])
        
    
    return G

def node_prune(graph, path):
    """
    Takes the input graph object and prunes any nodes that are not in the
    path input list.

    Parameters
    ----------
    graph : Networkx.Graph()
    path : list

    Returns
    -------
    G : Networkx.Graph()
    
    """
    
    # TODO
    # Switch to cleaner list comprehension [exp(i) for i in list if filter(i)]
    G = nx.Graph(graph)
    
    cutnodes = []
    for node in G.nodes:
        if node in path:
            continue
        else:
            cutnodes.append(node)
    
    G.remove_nodes_from(cutnodes)
    
    return G
    
def pruning(subgraph_list):
    """
    Takes input list of graphs, finds longest path, and trims accordingly.

    Parameters
    ----------
    subgraph_list : list

    Returns
    -------
    graphs_list : list

    """
    
    graphs_list = []
    for i in subgraph_list:
        
        # Creating New graph to modify
        G = nx.Graph(i)
        
        # Gathering longest path and edges to keep
        paths, long_path = longest_path(G)
        
        # Check if NoneType was returned, pass if so.
        if paths == None and long_path == None:
            continue
        
        edges = keep_edges(long_path)
        
        # Pruning off edges and nodes
        G2 = edge_prune(G, edges)
        G3 = node_prune(G2, long_path)
        
        # Appending newly modified graph to main list
        graphs_list.append(G3)
    
    return graphs_list


if __name__ == "__main__":
    
    # Test Graphing Here
    tg = test_graph()

    tg_subgraph_list = subgraph_setup(tg)
    
    new_graphs = pruning(tg_subgraph_list)
    
    # Test Data here
    DATA_DIR = 'ngc4321_subset.fits'

    with fits.open(DATA_DIR) as hdul:
        data = hdul[0].data
        
    # Setting Global Threshold here
    mask = data > 0.4
    
    p_g, coords, degrees = create_skeleton(mask)
    
    tg2 = create_network(p_g, coords)
    
    tg2_sub_list = subgraph_setup(tg2)
    
    new_graphs2 = pruning(tg2_sub_list)