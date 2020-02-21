# Licensed under an MIT open source license - see LICENSE

import numpy as np
import skimage.morphology as mo
from skan import csr
import networkx as nx

from .filament import Filament3D
from .base_conversions import (BaseInfoMixin, UnitConverter,
                               find_beam_properties, data_unit_check)


class FilFinder3D(BaseInfoMixin):
    """
    Filament detection and characterization for PPV cubes.

    Parameters
    ----------
    cube :

    """
    def __init__(self, cube, mask=None):
        super(FilFinder3D, self).__init__()

        self.cube = cube

        if mask is not None:
            self.mask = mask

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, inp):

        assert inp.shape == self.cube.shape

        # Consistency checks
        self._mask = inp

    def preprocess_cube(self):
        pass

    def create_mask(self, use_existing_mask=False):
        '''
        Create a boolean mask of filamentary structure.
        '''

        if use_existing_mask:
            if not hasattr(self, "_mask"):
                raise ValueError("No pre-created mask given. "
                                 "Pass a mask to the kwarg `mask` in "
                                 "`FilFinder3D` or use "
                                 "`use_existing_mask=False`")
            return

        # Implement the 3D adaptive thresholding here.
        raise NotImplementedError()

        selem = mo.ball(3)

        self.mask = mo.dilation(self._mask, selem)
        self.mask = mo.closing(self.mask, selem)

    def create_skeleton(self):
        '''
        Create a skeleton from the 3D filament mask and convert to the
        sparse scipy graph format.
        '''

        self._skeleton = mo.skeletonize_3d(self.mask)

        # Make a sparse encoding of the skeleton structure

        self._skeleton_encoded = csr.skeleton_to_csgraph(self.skeleton)
        # pixel_graph0, coordinates0, degrees0
        # Parameters :
        #      pixel_graph0: is a SciPy CSR matrix in which entry (i,j) is 0 if pixels
        # i and j are not connected, and otherwise is equal to the distance between
        # pixels i and j in the skeleton.

        # coordinates0 : Coordinates (in pixel units) of the points in the pixel graph.

    @property
    def skeleton(self):
        '''
        Boolean array of filament skeletons.
        '''
        return self._skeleton

    @skeleton.setter
    def skeleton(self, inp):

        assert inp.shape == self.cube.shape

        self._skeleton = inp

    def analyze_skeletons(self, min_length=5, verbose=False):
        '''
        Prune and calculate the length of skeletons and branches.

        Converts the skeleton structure into a networkx graph and
        prune branches shorter than the given `min_length`.

        Parameters
        -----------
        min_length : `astropy.units.Quantity`, optional
            Minimum branch length to keep in the skeleton.

        '''
        pixel_graph0, coordinates0, parameter = self._skeleton_encoded

        cutnodes = []
        G = nx.from_scipy_sparse_matrix(pixel_graph0)

        for node in G.node:
            G.node[node]['pos'] = coordinates0[node]

        if verbose:
            print("Number of nodes before Pruning:")
            print(nx.number_of_nodes(G))

        # extract subgraphs
        for H in nx.connected_component_subgraphs(G):

            # Find end points and intersections
            endPoints = []
            junction = []
            branchPoints = []
            for n in H.nodes:
                if H.degree(n) == 1:
                    endPoints.append(n)
                elif H.degree(n) == 1:
                    branchPoints.append(n)
                elif H.degree(n) > 2:
                    junction.append(n)

            # print(argh)

            # This loops through all junctions to get the endpoints
            # It would be faster loop through the end points only.

            # for all the junction nodes removing the branches
            for k in junction:

                    for i in endPoints:
                        if nx.has_path(H, k, i) and H.degree(k) > 2:

                            p = nx.shortest_path(H, source=k, target=i)
                            print(argh)
                            length = len(p)
                            if length < min_length:
                            # if length < parameter:
                                cutnode = p[1]

                                if H.has_edge(k, cutnode):
                                    H.remove_edge(k, cutnode)
                                    cutnodes.append(cutnode)

                    # Removing Subgraph less then parameter length(here 5nodes)
                    # Reminder: # nodes == # pixels.
                    for n in cutnodes:
                        if G.has_edge(k, n):
                            G.remove_edge(k, n)
                            print("removed : {0}---{1}".format(k, n))

            # Removing nodes, where nodes == pixels.
            # So this isn't quite a length, but it's not a
            # bad approx if
            if nx.number_of_nodes(H) <= parameter:
                if verbose:
                    print("Removing nodes : {}".format(H.nodes()))
                G.remove_nodes_from(H.nodes())

        if verbose:
            print("Number of nodes After removing Subgraphs < parameter "
                  "length: {}".format(nx.number_of_nodes(G)))

        # removing Subgraphs afer removing edges

        sub_graphs2 = nx.connected_component_subgraphs(G)

        subgraphlist2 = []

        for i, sg in enumerate(sub_graphs2):
            subgraphlist2.append(sg.nodes())

        for j in subgraphlist2:
            H2 = nx.Graph(G.subgraph(j))
            if nx.number_of_nodes(H2) <= parameter:
                if verbose:
                    print("removing nodes 2: {}".format(H.nodes()))
                G.remove_nodes_from(H2.nodes())

        if verbose:
            print("Number of nodes After Pruning: {}"
                  .format(nx.number_of_nodes(G)))

        return G
