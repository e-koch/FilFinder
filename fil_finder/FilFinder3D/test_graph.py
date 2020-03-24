#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 11:37:32 2020

@author: samuelfielder
"""

import networkx as nx


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