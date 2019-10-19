# Licensed under an MIT open source license - see LICENSE

from .utilities import *
from .pixel_ident import *


import numpy as np
import scipy.ndimage as nd
import networkx as nx
import operator
import string
import copy

# Create 4 to 8-connected elements to use with binary hit-or-miss
struct1 = np.array([[1, 0, 0],
                    [0, 1, 1],
                    [0, 0, 0]])

struct2 = np.array([[0, 0, 1],
                    [1, 1, 0],
                    [0, 0, 0]])

# Next check the three elements which will be double counted
check1 = np.array([[1, 1, 0, 0],
                   [0, 0, 1, 1]])

check2 = np.array([[0, 0, 1, 1],
                   [1, 1, 0, 0]])

check3 = np.array([[1, 1, 0],
                   [0, 0, 1],
                   [0, 0, 1]])


def skeleton_length(skeleton):
    '''
    Length finding via morphological operators. We use the differences in
    connectivity between 4 and 8-connected to split regions. Connections
    between 4 and 8-connected regions are found using a series of hit-miss
    operators.

    The inputted skeleton MUST have no intersections otherwise the returned
    length will not be correct!

    Parameters
    ----------
    skeleton : numpy.ndarray
        Array containing the skeleton.

    Returns
    -------
    length : float
        Length of the skeleton.

    '''

    # 4-connected labels
    four_labels = nd.label(skeleton)[0]

    four_sizes = nd.sum(skeleton, four_labels, range(np.max(four_labels) + 1))

    # Lengths is the number of pixels minus number of objects with more
    # than 1 pixel.
    four_length = np.sum(
        four_sizes[four_sizes > 1]) - len(four_sizes[four_sizes > 1])

    # Find pixels which a 4-connected and subtract them off the skeleton

    four_objects = np.where(four_sizes > 1)[0]

    skel_copy = copy.copy(skeleton)
    for val in four_objects:
        skel_copy[np.where(four_labels == val)] = 0

    # Remaining pixels are only 8-connected
    # Lengths is same as before, multiplied by sqrt(2)

    eight_labels = nd.label(skel_copy, eight_con())[0]

    eight_sizes = nd.sum(
        skel_copy, eight_labels, range(np.max(eight_labels) + 1))

    eight_length = (
        np.sum(eight_sizes) - np.max(eight_labels)) * np.sqrt(2)

    # If there are no 4-connected pixels, we don't need the hit-miss portion.
    if four_length == 0.0:
        conn_length = 0.0

    else:

        store = np.zeros(skeleton.shape)

        # Loop through the 4 rotations of the structuring elements
        for k in range(0, 4):
            hm1 = nd.binary_hit_or_miss(
                skeleton, structure1=np.rot90(struct1, k=k))
            store += hm1

            hm2 = nd.binary_hit_or_miss(
                skeleton, structure1=np.rot90(struct2, k=k))
            store += hm2

            hm_check3 = nd.binary_hit_or_miss(
                skeleton, structure1=np.rot90(check3, k=k))
            store -= hm_check3

            if k <= 1:
                hm_check1 = nd.binary_hit_or_miss(
                    skeleton, structure1=np.rot90(check1, k=k))
                store -= hm_check1

                hm_check2 = nd.binary_hit_or_miss(
                    skeleton, structure1=np.rot90(check2, k=k))
                store -= hm_check2

        conn_length = np.sqrt(2) * \
            np.sum(np.sum(store, axis=1), axis=0)  # hits

    return conn_length + eight_length + four_length

########################################################
# Composite Functions
########################################################


def init_lengths(labelisofil, filbranches, array_offsets, img):
    '''

    This is a wrapper on fil_length for running on the branches of the
    skeletons.

    Parameters
    ----------
    labelisofil : list
        Contains individual arrays for each skeleton where the
        branches are labeled and the intersections have been removed.
    filbranches : list
        Contains the number of branches in each skeleton.
    array_offsets : List
        The indices of where each filament array fits in the
        original image.
    img : numpy.ndarray
        Original image.

    Returns
    -------
    branch_properties: dict
        Contains the lengths and intensities of the branches.
        Keys are *length* and *intensity*.

    '''
    num = len(labelisofil)

    # Initialize Lists
    lengths = []
    av_branch_intensity = []
    all_branch_pts = []

    for n in range(num):
        leng = []
        av_intensity = []
        branch_pix = []

        label_copy = copy.copy(labelisofil[n])
        objects = nd.find_objects(label_copy)
        for i, obj in enumerate(objects):
            # Scale the branch array to the branch size
            branch_array = np.zeros_like(label_copy[obj])

            # Find the skeleton points and set those to 1
            branch_pts = np.where(label_copy[obj] == i + 1)
            branch_array[branch_pts] = 1

            # Now find the length on the branch
            if branch_array.sum() == 1:
                # Single pixel. No need to find length
                # For use in longest path algorithm, will be set to zero for
                # final analysis
                branch_length = 0.5
            else:
                branch_length = skeleton_length(branch_array)

            leng.append(branch_length)

            # Now let's find the average intensity along each branch
            # Get the offsets from the original array and
            # add on the offset the branch array introduces.
            x_offset = obj[0].start + array_offsets[n][0][0]
            y_offset = obj[1].start + array_offsets[n][0][1]

            # Starting w/ astropy v4.0 and numpy 1.17, the unit is retained
            # on the array. We're going to strip the unit off when needed.
            if hasattr(img, 'unit'):
                img = img.value.copy()

            intensities = []
            for x, y in zip(*branch_pts):
                is_finite = np.isfinite(img[x + x_offset, y + y_offset])
                is_not_negative = img[x + x_offset, y + y_offset] >= 0.0

                if is_finite and is_not_negative:
                    intensities.append(img[x + x_offset, y + y_offset])
            av_intensity.append(np.nanmean(intensities))

            branch_pix.append(np.array([(x + x_offset, y + y_offset)
                                        for x, y in zip(*branch_pts)]))

        lengths.append(leng)
        av_branch_intensity.append(av_intensity)
        all_branch_pts.append(branch_pix)

        branch_properties = {"length": lengths,
                             "intensity": av_branch_intensity,
                             "pixels": all_branch_pts,
                             "number": [len(length) for length in lengths]}

    return branch_properties


def pre_graph(labelisofil, branch_properties, interpts, ends):
    '''

    This function converts the skeletons into a graph object compatible with
    networkx. The graphs have nodes corresponding to end and
    intersection points and edges defining the connectivity as the branches
    with the weights set to the branch length.

    Parameters
    ----------

    labelisofil : list
        Contains individual arrays for each skeleton where the
        branches are labeled and the intersections have been removed.

    branch_properties : dict
        Contains the lengths and intensities of all branches.

    interpts : list
        Contains the pixels which belong to each intersection.

    ends : list
        Contains the end pixels for each skeleton.

    Returns
    -------

    end_nodes : list
        Contains the nodes corresponding to end points.

    inter_nodes : list
        Contains the nodes corresponding to intersection points.

    edge_list : list
        Contains the connectivity information for the graphs.

    nodes : list
        A complete list of all of the nodes. The other nodes lists have
        been separated as they are labeled differently.

    '''

    num = len(labelisofil)

    end_nodes = []
    inter_nodes = []
    nodes = []
    edge_list = []
    loop_edges = []

    def path_weighting(idx, length, intensity, w=0.5):
        '''

        Relative weighting for the shortest path algorithm using the branch
        lengths and the average intensity along the branch.

        '''
        if w > 1.0 or w < 0.0:
            raise ValueError(
                "Relative weighting w must be between 0.0 and 1.0.")
        return (1 - w) * (length[idx] / np.sum(length)) + \
            w * (intensity[idx] / np.sum(intensity))

    lengths = branch_properties["length"]
    branch_intensity = branch_properties["intensity"]

    for n in range(num):
        inter_nodes_temp = []
        # Create end_nodes, which contains lengths, and nodes, which we will
        # later add in the intersections
        end_nodes.append([(labelisofil[n][i[0], i[1]],
                           path_weighting(int(labelisofil[n][i[0], i[1]] - 1),
                                          lengths[n],
                                          branch_intensity[n]),
                           lengths[n][int(labelisofil[n][i[0], i[1]] - 1)],
                           branch_intensity[n][int(labelisofil[n][i[0], i[1]] - 1)])
                          for i in ends[n]])
        nodes.append([labelisofil[n][i[0], i[1]] for i in ends[n]])

    # Intersection nodes are given by the intersections points of the filament.
    # They are labeled alphabetically (if len(interpts[n])>26,
    # subsequent labels are AA,AB,...).
    # The branch labels attached to each intersection are included for future
    # use.
        for intersec in interpts[n]:
            uniqs = []
            for i in intersec:  # Intersections can contain multiple pixels
                int_arr = np.array([[labelisofil[n][i[0] - 1, i[1] + 1],
                                     labelisofil[n][i[0], i[1] + 1],
                                     labelisofil[n][i[0] + 1, i[1] + 1]],
                                    [labelisofil[n][i[0] - 1, i[1]], 0,
                                     labelisofil[n][i[0] + 1, i[1]]],
                                    [labelisofil[n][i[0] - 1, i[1] - 1],
                                     labelisofil[n][i[0], i[1] - 1],
                                     labelisofil[n][i[0] + 1, i[1] - 1]]]).astype(int)
                for x in np.unique(int_arr[np.nonzero(int_arr)]):
                    uniqs.append((x,
                                  path_weighting(x - 1, lengths[n],
                                                 branch_intensity[n]),
                                  lengths[n][x - 1],
                                  branch_intensity[n][x - 1]))
            # Intersections with multiple pixels can give the same branches.
            # Get rid of duplicates
            uniqs = list(set(uniqs))
            inter_nodes_temp.append(uniqs)

        # Add the intersection labels. Also append those to nodes
        inter_nodes.append(list(zip(product_gen(string.ascii_uppercase),
                                    inter_nodes_temp)))
        for alpha, node in zip(product_gen(string.ascii_uppercase),
                               inter_nodes_temp):
            nodes[n].append(alpha)
        # Edges are created from the information contained in the nodes.
        edge_list_temp = []
        loops_temp = []
        for i, inters in enumerate(inter_nodes[n]):
            end_match = list(set(inters[1]) & set(end_nodes[n]))
            for k in end_match:
                edge_list_temp.append((inters[0], k[0], k))

            for j, inters_2 in enumerate(inter_nodes[n]):
                if i != j:
                    match = list(set(inters[1]) & set(inters_2[1]))
                    new_edge = None
                    if len(match) == 1:
                        new_edge = (inters[0], inters_2[0], match[0])
                    elif len(match) > 1:
                        # Multiple connections (a loop)
                        multi = [match[l][1] for l in range(len(match))]
                        keep = multi.index(min(multi))
                        new_edge = (inters[0], inters_2[0], match[keep])

                        # Keep the other edges information in another list
                        for jj in range(len(multi)):
                            if jj == keep:
                                continue
                            loop_edge = (inters[0], inters_2[0], match[jj])
                            dup_check = loop_edge not in loops_temp and \
                                (loop_edge[1], loop_edge[0], loop_edge[2]) \
                                not in loops_temp
                            if dup_check:
                                loops_temp.append(loop_edge)

                    if new_edge is not None:
                        dup_check = (new_edge[1], new_edge[0], new_edge[2]) \
                            not in edge_list_temp \
                            and new_edge not in edge_list_temp
                        if dup_check:
                            edge_list_temp.append(new_edge)

        # Remove duplicated edges between intersections

        edge_list.append(edge_list_temp)
        loop_edges.append(loops_temp)

    return edge_list, nodes, loop_edges


def longest_path(edge_list, nodes, verbose=False,
                 skeleton_arrays=None, save_png=False, save_name=None):
    '''
    Takes the output of pre_graph and runs the shortest path algorithm.

    Parameters
    ----------

    edge_list : list
        Contains the connectivity information for the graphs.

    nodes : list
        A complete list of all of the nodes. The other nodes lists have
        been separated as they are labeled differently.

    verbose : bool, optional
        If True, enables the plotting of the graph.

    skeleton_arrays : list, optional
        List of the skeleton arrays. Required when verbose=True.

    save_png : bool, optional
        Saves the plot made in verbose mode. Disabled by default.

    save_name : str, optional
        For use when ``save_png`` is enabled.
        **MUST be specified when ``save_png`` is enabled.**

    Returns
    -------

    max_path : list
        Contains the paths corresponding to the longest lengths for
        each skeleton.

    extremum : list
        Contains the starting and ending points of max_path

    '''
    num = len(nodes)

    # Initialize lists
    max_path = []
    extremum = []
    graphs = []

    for n in range(num):
        G = nx.Graph()
        G.add_nodes_from(nodes[n])
        for i in edge_list[n]:
            G.add_edge(i[0], i[1], weight=i[2][1])
        # networkx 2.0 returns a two-element tuple. Convert to a dict first
        paths = dict(nx.shortest_path_length(G, weight='weight'))
        values = []
        node_extrema = []

        for i in paths.keys():
            j = max(paths[i].items(), key=operator.itemgetter(1))
            node_extrema.append((j[0], i))
            values.append(j[1])
        max_path_length = max(values)
        start, finish = node_extrema[values.index(max_path_length)]
        extremum.append([start, finish])

        # def get_weight(pat):
        #     return sum([G.edge[x][y]['weight'] for x, y in
        #                 zip(pat[:-1], pat[1:])])

        # for pat in nx.shortest_simple_paths(G, start, finish):
        #     if np.isclose(get_weight(pat), max_path_length) or get_weight(pat) > max_path_length:
        #         long_path = pat
        #         break

        long_path = \
            list(nx.shortest_simple_paths(G, start, finish, 'weight'))[-1]

        max_path.append(long_path)
        graphs.append(G)

        if verbose or save_png:
            if not skeleton_arrays:
                Warning("Must input skeleton arrays if verbose or save_png is"
                        " enabled. No plots will be created.")
            elif save_png and save_name is None:
                Warning("Must give a save_name when save_png is enabled. No"
                        " plots will be created.")
            else:
                # Check if skeleton_arrays is a list
                assert isinstance(skeleton_arrays, list)
                import matplotlib.pyplot as p
                # if verbose:
                #     print("Filament: %s / %s" % (n + 1, num))
                p.subplot(1, 2, 1)
                p.imshow(skeleton_arrays[n], interpolation="nearest",
                         origin="lower")

                p.subplot(1, 2, 2)
                elist = [(u, v) for (u, v, d) in G.edges(data=True)]
                pos = nx.spring_layout(G)
                nx.draw_networkx_nodes(G, pos, node_size=200)
                nx.draw_networkx_edges(G, pos, edgelist=elist, width=2)
                nx.draw_networkx_labels(
                    G, pos, font_size=10, font_family='sans-serif')
                p.axis('off')

                if save_png:
                    p.savefig(save_name)
                    p.close()
                if verbose:
                    p.show()
                    p.clf()

    return max_path, extremum, graphs


def prune_graph(G, nodes, edge_list, max_path, labelisofil, branch_properties,
                loop_edges, prune_criteria='all', length_thresh=0,
                relintens_thresh=0.2, max_iter=1):
    '''
    Function to remove unnecessary branches, while maintaining connectivity
    in the graph. Also updates edge_list, nodes, branch_lengths and
    filbranches.

    Parameters
    ----------
    G : list
        Contains the networkx Graph objects.
    nodes : list
        A complete list of all of the nodes. The other nodes lists have
        been separated as they are labeled differently.
    edge_list : list
        Contains the connectivity information for the graphs.
    max_path : list
        Contains the paths corresponding to the longest lengths for
        each skeleton.
    labelisofil : list
        Contains individual arrays for each skeleton where the
        branches are labeled and the intersections have been removed.
    branch_properties : dict
        Contains the lengths and intensities of all branches.
    loop_edges : list
        List of edges that create loops in the graph. These are not included
        in `edge_list` or the graph to avoid making self-loops.
    prune_criteria : {'all', 'intensity', 'length'}, optional
        Choose the property to base pruning on. 'all' requires that the branch
        fails to satisfy the length and relative intensity checks.
    length_thresh : int or float
        Minimum length a branch must be to be kept. Can be overridden if the
        branch is bright relative to the entire skeleton.
    relintens_thresh : float between 0 and 1, optional.
        Threshold for how bright the branch must be relative to the entire
        skeleton. Can be overridden by length.

    Returns
    -------
    labelisofil : list
        Updated from input.
    edge_list : list
        Updated from input.
    nodes : list
        Updated from input.
    branch_properties : dict
        Updated from input.
    '''

    num = len(labelisofil)

    if prune_criteria not in ['all', 'length', 'intensity']:
        raise ValueError("prune_criteria must be 'all', 'length' or "
                         "'intensity'. Given {}".format(prune_criteria))

    for n in range(num):
        # Fix for networkx 2.0
        iterat = 0
        while True:
            degree = dict(G[n].degree())

            # Look for unconnected nodes and remove from the graph
            unconn = [key for key in degree.keys() if degree[key] == 0]
            if len(unconn) > 0:
                for node in unconn:
                    G[n].remove_node(node)

            single_connect = [key for key in degree.keys() if degree[key] == 1]

            delete_candidate = list((set(nodes[n]) - set(max_path[n])) &
                                    set(single_connect))

            # Nothing to delete!
            if not delete_candidate and len(loop_edges[n]) == 0:
                break

            edge_candidates = [(edge[2][0], edge) for idx, edge in
                               enumerate(edge_list[n])
                               if edge[0] in delete_candidate or
                               edge[1] in delete_candidate]
            intensities = [edge[2][3] for edge in edge_list[n]]

            # Add in loop edges for candidates to delete
            edge_candidates += [(edge[2][0], edge) for edge in loop_edges[n]]
            intensities += [edge[2][3] for edge in loop_edges[n]]

            del_idx = []
            for idx, edge in edge_candidates:
                # In the odd case where a loop meets at the same intersection,
                # ensure that edge is kept.
                # if isinstance(edge[0], str) & isinstance(edge[1], str):
                #     continue

                length = edge[2][2]
                av_intensity = edge[2][3]

                if prune_criteria == 'all':
                    criterion1 = length < length_thresh
                    criterion2 = (av_intensity / np.sum(intensities)) < \
                        relintens_thresh
                    criteria = criterion1 & criterion2
                elif prune_criteria == 'intensity':
                    criteria = \
                        (av_intensity / np.sum(intensities)) < relintens_thresh
                else:  # Length only
                    criteria = length < length_thresh

                if criteria:
                    edge_pts = np.where(labelisofil[n] == edge[2][0])
                    assert len(edge_pts[0]) == len(branch_properties['pixels'][n][idx-1])
                    labelisofil[n][edge_pts] = 0
                    try:
                        edge_list[n].remove(edge)
                        nodes[n].remove(edge[1])
                        G[n].remove_edge(edge[0], edge[1])
                    except ValueError:
                        loop_edges[n].remove(edge)
                    branch_properties["number"][n] -= 1
                    del_idx.append(idx)

            if len(del_idx) > 0:
                del_idx.sort()
                for idx in del_idx[::-1]:
                    branch_properties['pixels'][n].pop(idx - 1)
                    branch_properties['length'][n].pop(idx - 1)
                    branch_properties['intensity'][n].pop(idx - 1)

            # Now check to see if we need to merge any nodes in the graph
            # after deletion. These will be intersection nodes with 2
            # connections
            while True:
                degree = dict(G[n].degree())
                doub_connect = [key for key in degree.keys()
                                if degree[key] == 2]

                if len(doub_connect) == 0:
                    break

                for node in doub_connect:
                    G[n] = merge_nodes(node, G[n])

            iterat += 1

            if iterat == max_iter:
                # warnings.warn("Graph pruning reached max iterations.")
                break

    return labelisofil, edge_list, nodes, branch_properties


def main_length(max_path, edge_list, labelisofil, interpts, branch_lengths,
                img_scale, verbose=False, save_png=False, save_name=None):
    '''
    Wraps previous functionality together for all of the skeletons in the
    image. To find the overall length for each skeleton, intersections are
    added back in, and any extraneous pixels they bring with them are deleted.

    Parameters
    ----------
    max_path : list
        Contains the paths corresponding to the longest lengths for
        each skeleton.
    edge_list : list
        Contains the connectivity information for the graphs.
    labelisofil : list
        Contains individual arrays for each skeleton where the
        branches are labeled and the intersections have been removed.
    interpts : list
        Contains the pixels which belong to each intersection.
    branch_lengths : list
        Lengths of individual branches in each skeleton.
    img_scale : float
        Conversion from pixel to physical units.
    verbose : bool, optional
        Returns plots of the longest path skeletons.
    save_png : bool, optional
        Saves the plot made in verbose mode. Disabled by default.
    save_name : str, optional
        For use when ``save_png`` is enabled.
        **MUST be specified when ``save_png`` is enabled.**

    Returns
    -------
    main_lengths : list
        Lengths of the skeletons.
    longpath_arrays : list
        Arrays of the longest paths in the skeletons.
    '''

    main_lengths = []
    longpath_arrays = []

    for num, (path, edges, inters, skel_arr, lengths) in \
        enumerate(zip(max_path, edge_list, interpts, labelisofil,
                      branch_lengths)):

        if len(path) == 1:
            main_lengths.append(lengths[0] * img_scale)
            skeleton = skel_arr  # for viewing purposes when verbose
        else:
            skeleton = np.zeros(skel_arr.shape)

            # Add edges along longest path
            good_edge_list = [(path[i], path[i + 1])
                              for i in range(len(path) - 1)]
            # Find the branches along the longest path.
            for i in good_edge_list:
                for j in edges:
                    if (i[0] == j[0] and i[1] == j[1]) or \
                       (i[0] == j[1] and i[1] == j[0]):
                        label = j[2][0]
                        skeleton[np.where(skel_arr == label)] = 1

            # Add intersections along longest path
            intersec_pts = []
            for label in path:
                try:
                    label = int(label)
                except ValueError:
                    pass
                if not isinstance(label, int):
                    k = 1
                    while list(zip(product_gen(string.ascii_uppercase),
                                   [1] * k))[-1][0] != label:
                        k += 1
                    intersec_pts.extend(inters[k - 1])
                    skeleton[tuple(zip(*inters[k - 1]))] = 2
            # Remove unnecessary pixels
            count = 0
            while True:
                for pt in intersec_pts:
                    # If we have already eliminated the point, continue
                    if skeleton[pt] == 0:
                        continue
                    skeleton[pt] = 0
                    lab_try, n = nd.label(skeleton, eight_con())
                    if n > 1:
                        skeleton[pt] = 1
                    else:
                        count += 1
                if count == 0:
                    break
                count = 0

            main_lengths.append(skeleton_length(skeleton) * img_scale)

        longpath_arrays.append(skeleton.astype(int))

        if verbose or save_png:
            if save_png and save_name is None:
                ValueError("Must give a save_name when save_png is enabled. No"
                           " plots will be created.")
            import matplotlib.pyplot as p
            # if verbose:
            #     print("Filament: %s / %s" % (num + 1, len(labelisofil)))

            p.subplot(121)
            p.imshow(skeleton, origin='lower', interpolation="nearest")
            p.subplot(122)
            p.imshow(labelisofil[num], origin='lower',
                     interpolation="nearest")

            if save_png:
                p.savefig(save_name)
                p.close()
            if verbose:
                p.show()
                p.clf()

    return main_lengths, longpath_arrays
