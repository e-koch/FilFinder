
'''
Skeleton routines common to 3D data.
'''

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.ndimage as nd
import skimage.morphology as mo
import networkx as nx
import warnings


class Skeleton3D(object):
    """
    docstring for Skeleton3D
    """

    def _has_skan(self):
        try:
            import skan
        except ImportError:
            raise ImportError("3D filaments requires the skan package to be installed.")

    def create_skeleton(self, min_pixel=1):
        """

        Parameters
        ----------
        min_pixel: int
            Minimum number of pixels in the skeleton. Must be >1 to form a valid skeleton.

        """

        if min_pixel < 1:
            raise ValueError("Minimum number of pixels must be > 1.")

        self._min_pixel = min_pixel

        # Creating Skeleton
        # Force output type to be a bool
        skeleton_init = mo.skeletonize(self.mask, method='lee') > 0

        skel_labels, num = nd.label(skeleton_init, np.ones((3, 3, 3)))

        self._skel_pix_nums = nd.sum(skeleton_init, skel_labels, range(1, num + 1))

        # If fewer pixels than min_pixel, remove from the skeleton array.
        # Also create Filament objects
        for i in range(1, num + 1):

            coords = np.where(skel_labels == i)

            if coords[0].size > min_pixel:
                continue

            skeleton_init[coords] = False

        self._skel_labels = nd.label(skeleton_init, np.ones((3, 3, 3)))[0]

        self.skeleton = skeleton_init

    @property
    def skeleton_labels(self):
        return self._skel_labels

    def skeleton_labels_to_pixel_coords(self, label_nums=None):

        if label_nums is None:
            nmax = self.skeleton_labels.max()
            label_nums = range(1, nmax+1)

        # Convert labels to lists of pixel coords
        skel_label_pixels = []

        for this_label in label_nums:
            coords = np.where(self.skeleton_labels == this_label)

            # Create a new column with the label number
            new_column = np.ones(coords[0].shape) * this_label

            # Stack the new column with the coordinates
            new_coords = np.vstack((coords[0], coords[1], new_column)).T

            # Append the new coordinates to the list of arrays
            skel_label_pixels.append(new_coords)

        # Convert the list of arrays to a single array
        skel_label_pixels = np.vstack(skel_label_pixels)

        return skel_label_pixels

    def network_plot_3D(self, angle=40, filename='plot.pdf', save=False):
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

        self._has_skan()

        G = self.graph

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


#     def create_network(self):
#         """
#         Creates the initial network and subgraph_list objects.

#         Attributes
#         -------
#         network : networkx.Graph()
#             object that contains the graph representation of pixel_graph
#         subgraph_list : list
#             Contains graph objects of found connected component graphs

#         """
#         pass

#         # # Converting skeleton to graph
#         # from skan import csr

#         # out = csr.skeleton_to_csgraph(self.skeleton,
#         #                               unique_junctions=False)

#         # self.pixel_graph, self.coordinates, self.degrees = out

#         # # Re-casting Coordinates into int
#         # self.coordinates = self.coordinates.astype(int)

#         # self.network = nx.from_scipy_sparse_matrix(self.pixel_graph)

#         # # Appending 3D pixel positions as node attributes
#         # # Appending 3D Pixel data value as node attribute
#         # for node in self.network.nodes:
#         #     self.network.nodes[node]['pos'] = self.coordinates[node]
#         #     self.network.nodes[node]['data'] = self._image[self.coordinates[node][0],
#         #                                                    self.coordinates[node][1],
#         #                                                    self.coordinates[node][2]]

#         # self.subgraph_list = []

#         # for sub_id in nx.connected_components(self.network):

#         #     subgraph = self.network.subgraph(sub_id)

#         #     self.subgraph_list.append(subgraph)

#             # Example of the min_pixel cut applied to graphs.
#             # This might be a lot more efficient, so I'm keeping
#             # it here for now.

#             # # Check the number of pixels in the skeleton.
#             # if subgraph.number_of_nodes() >= min_pixels:

#             #     self.subgraph_list.append(subgraph)

#             # else:
#             #     # Remove from the whole network.

#             #     self.network.remove_nodes_from(subgraph.nodes)


# def subgraph_analyzer(graph):
#     """
#     Takes graph object, finds longest shortest path between two endnodes,
#     isolates this path as a graph by removing edges on intersections that
#     are not part of the longest shortest path. Then creates a list of
#     subgraphs that are the branches to be inspected and pruned.

#     Parameters
#     ----------
#     graph : networkx.Graph

#     Returns
#     -------
#     filament : networkx.Graph
#     branches : list
#         list of networkx.Graph objects which represent branches off
#         the longest shortest path

#     """

#     # Grabbing Longest path along with nodes and internodes
#     paths, long_path, internodes = longest_path(graph)

#     # Building a list of edges to check against the main filament
#     longest_path_edges = edge_builder(long_path)

#     # Building list of nodes in main filament that are internodes
#     intersections = [i for i in long_path if i in internodes]

#     # Creating new graph to edit
#     H = nx.Graph(graph)

#     # Looping through each intersection along main filament branch
#     # and removing edges that are not part of the main filament branch
#     for i in intersections:
#         for j in graph.neighbors(i):
#             if (i, j) in longest_path_edges or (j, i) in longest_path_edges:
#                 pass
#             else:
#                 H.remove_edge(i, j)

#     # This leave H being a main filament, along with branches
#     # Split the data into different subgraphs
#     filaments = [H.subgraph(c) for c in nx.connected_components(H)]
#     filaments_lengths = [len(i.nodes) for i in filaments]

#     # Compute the main filament, and branches
#     filament = filaments[filaments_lengths.index(max(filaments_lengths))]
#     branches = [i for i in filaments if i is not filament]

#     return filament, branches


# def longest_path(graph):
#     """
#     Finds the shortest path for every combination of endnode to endnode.
#     Returns the longest path from the collection of shortest paths.

#     Parameters
#     -------
#     graph : Networkx.graph()

#     Returns
#     -------
#     paths : list
#         List of smallest possible paths found.
#     longest_path : list
#         Longest path of paths found.
#     internodes : list
#         List of nodes with degree greater than 2
#         Represents intersection points

#     """

#     # Checking for Single Isolated Node with 0 degree
#     # TODO Won't be needed after pre-pruning is dealt with
#     if len(graph) < 2:
#         return None, None

#     # Initialize lists
#     paths = []
#     endnodes = []
#     internodes = []

#     # Filing Endnodes and internode lists
#     for node in graph:
#         if graph.degree(node) == 1:
#             endnodes.append(node)
#         elif graph.degree(node) > 2:
#             internodes.append(node)

#     # Looping all endnodes to endnodes possible paths
#     for k in endnodes:
#         for i in endnodes:
#             if i != k:

#                 # Grabbing the shortest(s) path(s) from k to i
#                 path = list(nx.all_shortest_paths(graph, k, i))

#                 # Add the list of path into paths
#                 for i in path:
#                     paths.append(i)

#             else:
#                 # If k and i are the same, no path, continue.
#                 continue

#     # Trimming duplicate paths out of the list
#     for i in paths:
#         for ind, j in enumerate(paths):
#             if i != j:
#                 # Checking if the opposite is same
#                 if i == j[::-1]:
#                     # Delete if true
#                     del paths[ind]

#     # TODO What happens when the longest paths are tied -> tie-breaker
#     longest_path = max(paths, key=len)

#     return paths, longest_path, internodes


# def edge_builder(node_list):
#     """
#     Builds tuple edges for nodes in given list.
#     i.e. Input: [1,2,3] -> Output: [(1,2), (2,3)]

#     Parameters
#     ----------
#     node_list : list
#         List of single node numbers.

#     Returns
#     -------
#     edges : list
#          Returns a list of tuples that are the edges linking
#          input node_list together.

#     """
#     edges = []
#     for ind, val in enumerate(node_list):
#         # Looping to the second last entry of node_list
#         if ind < len(node_list) - 1:
#             edge = (node_list[ind], node_list[ind + 1])
#             edges.append(edge)

#     return edges
