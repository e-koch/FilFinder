# Licensed under an MIT open source license - see LICENSE

from .length import *
from .utilities import distance

import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as p
import copy


def isolateregions(binary_array, size_threshold=0, pad_size=0,
                   fill_hole=False, rel_size=0.1, morph_smooth=False):
    '''

    Labels regions in a boolean array and returns individual arrays for each
    region. Regions below a threshold can optionally be removed. Small holes
    may also be filled in.

    Parameters
    ----------
    binary_array : numpy.ndarray
        A binary array of regions.
    size_threshold : int, optional
        Sets the pixel size on the size of regions.
    pad_size : int, optional
        Padding to be added to the individual arrays.
    fill_hole : int, optional
        Enables hole filling.
    rel_size : float or int, optional
        If < 1.0, sets the minimum size a hole must be relative to the area
        of the mask. Otherwise, this is the maximum number of pixels the hole
        must have to be deleted.
    morph_smooth : bool, optional
        Morphologically smooth the image using a binar opening and closing.

    Returns
    -------
    output_arrays : list
        Regions separated into individual arrays.
    num : int
        Number of filaments
    corners : list
        Contains the indices where each skeleton array was taken from
        the original.

    '''

    output_arrays = []
    corners = []

    # Label skeletons
    labels, num = nd.label(binary_array, eight_con())

    # Remove skeletons which have fewer pixels than the threshold.
    if size_threshold != 0:
        sums = nd.sum(binary_array, labels, range(1, num + 1))
        remove_fils = np.where(sums <= size_threshold)[0]
        for lab in remove_fils:
            binary_array[np.where(labels == lab + 1)] = 0

        # Relabel after deleting short skeletons.
        labels, num = nd.label(binary_array, eight_con())

    # Split each skeleton into its own array.
    for n in range(1, num + 1):
        x, y = np.where(labels == n)
        # Make an array shaped to the skeletons size and padded on each edge
        # the +1 is because, e.g., range(0, 5) only has 5 elements, but the
        # indices we're using are range(0, 6)
        shapes = (x.max() - x.min() + 2 * pad_size + 1,
                  y.max() - y.min() + 2 * pad_size + 1)
        eachfil = np.zeros(shapes)
        eachfil[x - x.min() + pad_size, y - y.min() + pad_size] = 1
        # Fill in small holes
        if fill_hole:
            eachfil = _fix_small_holes(eachfil, rel_size=rel_size)
        if morph_smooth:
            eachfil = nd.binary_opening(eachfil, np.ones((3, 3)))
            eachfil = nd.binary_closing(eachfil, np.ones((3, 3)))
        output_arrays.append(eachfil)
        # Keep the coordinates from the original image
        lower = (max(0, x.min() - pad_size), max(0, y.min() - pad_size))
        upper = (x.max() + pad_size + 1, y.max() + pad_size + 1)
        corners.append([lower, upper])

    return output_arrays, num, corners


def find_filpix(branches, labelfil, final=True, debug=False):
    '''

    Identifies the types of pixels in the given skeletons. Identification is
    based on the connectivity of the pixel.

    Parameters
    ----------
    branches : list
        Contains the number of branches in each skeleton.
    labelfil : list
        Contains the arrays of each skeleton.
    final : bool, optional
        If true, corner points, intersections, and body points are all
        labeled as a body point for use when the skeletons have already
        been cleaned.
    debug : bool, optional
        Enable to print out (a lot) of extra info on pixel classification.

    Returns
    -------
    fila_pts : list
        All points on the body of each skeleton.
    inters : list
        All points associated with an intersection in each skeleton.
    labelfil : list
       Contains the arrays of each skeleton where all intersections
       have been removed.
    endpts_return : list
        The end points of each branch of each skeleton.
  '''

    initslices = []
    initlist = []
    shiftlist = []
    sublist = []
    endpts = []
    blockpts = []
    bodypts = []
    slices = []
    vallist = []
    shiftvallist = []
    cornerpts = []
    subvallist = []
    subslist = []
    pix = []
    filpix = []
    intertemps = []
    fila_pts = []
    inters = []
    repeat = []
    temp_group = []
    all_pts = []
    pairs = []
    endpts_return = []

    for k in range(1, branches + 1):
        x, y = np.where(labelfil == k)
        for i in range(len(x)):
            if x[i] < labelfil.shape[0] - 1 and y[i] < labelfil.shape[1] - 1:
                pix.append((x[i], y[i]))
                initslices.append(np.array([[labelfil[x[i] - 1, y[i] + 1],
                                             labelfil[x[i], y[i] + 1],
                                             labelfil[x[i] + 1, y[i] + 1]],
                                            [labelfil[x[i] - 1, y[i]], 0,
                                                labelfil[x[i] + 1, y[i]]],
                                            [labelfil[x[i] - 1, y[i] - 1],
                                             labelfil[x[i], y[i] - 1],
                                             labelfil[x[i] + 1, y[i] - 1]]]))

        filpix.append(pix)
        slices.append(initslices)
        initslices = []
        pix = []

    for i in range(len(slices)):
        for k in range(len(slices[i])):
            initlist.append([slices[i][k][0, 0],
                             slices[i][k][0, 1],
                             slices[i][k][0, 2],
                             slices[i][k][1, 2],
                             slices[i][k][2, 2],
                             slices[i][k][2, 1],
                             slices[i][k][2, 0],
                             slices[i][k][1, 0]])
        vallist.append(initlist)
        initlist = []

    for i in range(len(slices)):
        for k in range(len(slices[i])):
            shiftlist.append(shifter(vallist[i][k], 1))
        shiftvallist.append(shiftlist)
        shiftlist = []

    for k in range(len(slices)):
        for i in range(len(vallist[k])):
            for j in range(8):
                sublist.append(
                    int(vallist[k][i][j]) - int(shiftvallist[k][i][j]))
            subslist.append(sublist)
            sublist = []
        subvallist.append(subslist)
        subslist = []

    # x represents the subtracted list (step-ups) and y is the values of the
    # surrounding pixels. The categories of pixels are ENDPTS (x<=1),
    # BODYPTS (x=2,y=2),CORNERPTS (x=2,y=3),BLOCKPTS (x=3,y>=4), and
    # INTERPTS (x>=3).
    # A cornerpt is [*,0,0] (*s) associated with an intersection,
    # but their exclusion from
    #   [1,*,0] the intersection keeps eight-connectivity, they are included
    #   [0,1,0] intersections for this reason.
    # A blockpt is  [1,0,1] They are typically found in a group of four,
    # where all four
    #   [0,*,*] constitute a single intersection.
    #   [1,*,*]
    # A T-pt has the same connectivity as a block point, but with two 8-conns
    # [*, *, *]
    # [0, 1, 0]
    # The "final" designation is used when finding the final branch lengths.
    # At this point, blockpts and cornerpts should be eliminated.
    for k in range(branches):
        for l in range(len(filpix[k])):
            x = [j for j, y in enumerate(subvallist[k][l]) if y == k + 1]
            y = [j for j, z in enumerate(vallist[k][l]) if z == k + 1]

            if len(x) <= 1:
                if debug:
                    print("End pt. {}".format(filpix[k][l]))
                endpts.append(filpix[k][l])
                endpts_return.append(filpix[k][l])
            elif len(x) == 2:
                if final:
                    bodypts.append(filpix[k][l])
                else:
                    if len(y) == 2:
                        if debug:
                            print("Body pt. {}".format(filpix[k][l]))
                        bodypts.append(filpix[k][l])

                    elif is_tpoint(vallist[k][l]):
                        # If there are only 3 connections to the t-point, it
                        # is an end point
                        if len(y) == 3:
                            if debug:
                                print("T-point end {}".format(filpix[k][l]))
                            endpts.append(filpix[k][l])
                            endpts_return.append(filpix[k][l])
                        # If there are 4, it is a body point
                        elif len(y) == 4:
                            if debug:
                                print("T-point body {}".format(filpix[k][l]))

                            bodypts.append(filpix[k][l])
                        # Otherwise it is a part of an intersection
                        else:
                            if debug:
                                print("T-point inter {}".format(filpix[k][l]))
                            intertemps.append(filpix[k][l])
                    elif is_blockpoint(vallist[k][l]):
                        if debug:
                            print("Block pt. {}".format(filpix[k][l]))
                        blockpts.append(filpix[k][l])
                    else:
                        if debug:
                            print("Corner pt. {}".format(filpix[k][l]))
                        cornerpts.append(filpix[k][l])
            elif len(x) >= 3:
                if debug:
                    print("Inter pt. {}".format(filpix[k][l]))
                intertemps.append(filpix[k][l])
        endpts = list(set(endpts))
        bodypts = list(set(bodypts))
        dups = set(endpts) & set(bodypts)
        if len(dups) > 0:
            for i in dups:
                bodypts.remove(i)
        # Cornerpts without a partner diagonally attached can be included as a
        # bodypt.
        if debug:
            print("Cornerpts: {}".format(cornerpts))

        if len(cornerpts) > 0:
            deleted_cornerpts = []

            for i, j in zip(cornerpts, cornerpts):
                if i != j:
                    if distance(i[0], j[0], i[1], j[1]) == np.sqrt(2.0):
                        proximity = [(i[0], i[1] - 1),
                                     (i[0], i[1] + 1),
                                     (i[0] - 1, i[1]),
                                     (i[0] + 1, i[1]),
                                     (i[0] - 1, i[1] + 1),
                                     (i[0] + 1, i[1] + 1),
                                     (i[0] - 1, i[1] - 1),
                                     (i[0] + 1, i[1] - 1)]
                        match = set(intertemps) & set(proximity)
                        if len(match) == 1:
                            print("MATCH")
                            bodypts.extend([i, j])
                            # pairs.append([i, j])
                            deleted_cornerpts.append(i)
                            deleted_cornerpts.append(j)
            cornerpts = list(set(cornerpts).difference(set(deleted_cornerpts)))

        if len(cornerpts) > 0:
            for l in cornerpts:
                proximity = [(l[0], l[1] - 1),
                             (l[0], l[1] + 1),
                             (l[0] - 1, l[1]),
                             (l[0] + 1, l[1]),
                             (l[0] - 1, l[1] + 1),
                             (l[0] + 1, l[1] + 1),
                             (l[0] - 1, l[1] - 1),
                             (l[0] + 1, l[1] - 1)]

                # Check if the matching corner point is an end point
                # Otherwise the pixel will be combined into a 2-pixel intersec
                match_ends = set(endpts) & set(proximity[-4:])
                if len(match_ends) == 1:
                    fila_pts.append(endpts + bodypts + [l])
                    continue

                match = set(intertemps) & set(proximity)
                if len(match) == 1:
                    intertemps.append(l)
                    fila_pts.append(endpts + bodypts)
                else:
                    fila_pts.append(endpts + bodypts + [l])
        else:
            fila_pts.append(endpts + bodypts)

        # Reset lists
        cornerpts = []
        endpts = []
        bodypts = []

        if len(pairs) > 0:
            for i in range(len(pairs)):
                for j in pairs[i]:
                    all_pts.append(j)
        if len(blockpts) > 0:
            for i in blockpts:
                all_pts.append(i)
        if len(intertemps) > 0:
            for i in intertemps:
                all_pts.append(i)
        # Pairs of cornerpts, blockpts, and interpts are combined into an
        # array. If there is eight connectivity between them, they are labelled
        # as a single intersection.
        arr = np.zeros((labelfil.shape))
        for z in all_pts:
            labelfil[z[0], z[1]] = 0
            arr[z[0], z[1]] = 1
        lab, nums = nd.label(arr, eight_con())
        for k in range(1, nums + 1):
            objs_pix = np.where(lab == k)
            for l in range(len(objs_pix[0])):
                temp_group.append((objs_pix[0][l], objs_pix[1][l]))
            inters.append(temp_group)
            temp_group = []
    for i in range(len(inters) - 1):
        if inters[i] == inters[i + 1]:
            repeat.append(inters[i])
    for i in repeat:
        inters.remove(i)

    if debug:
        print("Fila pts: {}".format(fila_pts))
        print("Intersections: {}".format(inters))
        print("End pts: {}".format(endpts_return))
        print(labelfil)

    return fila_pts, inters, labelfil, endpts_return


def find_extran(branches, labelfil, debug=False):
    '''
    Identify pixels that are not necessary to keep the connectivity of the
    skeleton. It uses the same labeling process as find_filpix. Extraneous
    pixels tend to be those from former intersections, whose attached branch
    was eliminated in the cleaning process.

    Parameters
    ----------
    branches : list
        Contains the number of branches in each skeleton.
    labelfil : list
        Contains arrays of the labeled versions of each skeleton.
    debug : bool, optional
        Enable plotting of each filament array to visualize where the deleted
        pixels are.

    Returns
    -------
    labelfil : list
       Contains the updated labeled arrays with extraneous pieces
       removed.
    '''

    initslices = []
    initlist = []
    shiftlist = []
    sublist = []
    extran = []
    slices = []
    vallist = []
    shiftvallist = []
    subvallist = []
    subslist = []
    pix = []
    filpix = []

    for k in range(1, branches + 1):
        x, y = np.where(labelfil == k)
        for i in range(len(x)):
            if x[i] < labelfil.shape[0] - 1 and y[i] < labelfil.shape[1] - 1:
                pix.append((x[i], y[i]))
                initslices.append(np.array([[labelfil[x[i] - 1, y[i] + 1],
                                             labelfil[x[i], y[i] + 1],
                                             labelfil[x[i] + 1, y[i] + 1]],
                                            [labelfil[x[i] - 1, y[i]], 0,
                                             labelfil[x[i] + 1, y[i]]],
                                            [labelfil[x[i] - 1, y[i] - 1],
                                             labelfil[x[i], y[i] - 1],
                                             labelfil[x[i] + 1, y[i] - 1]]]))

        filpix.append(pix)
        slices.append(initslices)
        initslices = []
        pix = []

    for i in range(len(slices)):
        for k in range(len(slices[i])):
            initlist.append([slices[i][k][0, 0],
                             slices[i][k][0, 1],
                             slices[i][k][0, 2],
                             slices[i][k][1, 2],
                             slices[i][k][2, 2],
                             slices[i][k][2, 1],
                             slices[i][k][2, 0],
                             slices[i][k][1, 0]])
        vallist.append(initlist)
        initlist = []

    for i in range(len(slices)):
        for k in range(len(slices[i])):
            shiftlist.append(shifter(vallist[i][k], 1))
        shiftvallist.append(shiftlist)
        shiftlist = []

    for k in range(len(slices)):
        for i in range(len(vallist[k])):
            for j in range(8):
                sublist.append(
                    int(vallist[k][i][j]) - int(shiftvallist[k][i][j]))
            subslist.append(sublist)
            sublist = []
        subvallist.append(subslist)
        subslist = []

    for k in range(len(slices)):
        for l in range(len(filpix[k])):
            x = [j for j, y in enumerate(subvallist[k][l]) if y == k + 1]
            y = [j for j, z in enumerate(vallist[k][l]) if z == k + 1]

            if len(x) == 0:
                if debug:
                    print("Extran removal unconnect: {}".format(filpix[k][l]))
                labelfil[filpix[k][l][0], filpix[k][l][1]] = 0
                extran.append(filpix[k][l])

            if len(x) == 1:
                if len(y) >= 2:
                    if debug:
                        print("Extran removal: {}".format(filpix[k][l]))
                    extran.append(filpix[k][l])
                    labelfil[filpix[k][l][0], filpix[k][l][1]] = 0
        # if len(extran) >= 2:
        #     for i in extran:
        #         for j in extran:
        #             if i != j:
        #                 if distance(i[0], j[0], i[1], j[1]) == np.sqrt(2.0):
        #                     proximity = [(i[0], i[1] - 1),
        #                                  (i[0], i[1] + 1),
        #                                  (i[0] - 1, i[1]),
        #                                  (i[0] + 1, i[1]),
        #                                  (i[0] - 1, i[1] + 1),
        #                                  (i[0] + 1, i[1] + 1),
        #                                  (i[0] - 1, i[1] - 1),
        #                                  (i[0] + 1, i[1] - 1)]
        #                     match = set(filpix[k]) & set(proximity)
        #                     if len(match) > 0:
        #                         for z in match:
        #                             labelfil[z[0], z[1]] = 0

    if debug:
        import matplotlib.pyplot as plt
        plt.imshow(labelfil, origin='lower')
        for pix in extran:
            plt.plot(pix[1], pix[0], 'bD')
        plt.draw()
        raw_input("?")
        plt.clf()

    return labelfil


######################################################################
# Wrapper Functions
######################################################################


def pix_identify(isolatefilarr, num, debug=False):
    '''
    This function is essentially a wrapper on find_filpix. It returns the
    outputs of find_filpix in the form that are used during the analysis.

    Parameters
    ----------
    isolatefilarr : list
        Contains individual arrays of each skeleton.
    num  : int
        The number of skeletons.
    debug : bool, optional
        Print out identification steps in find_filpix.

    Returns
    -------
    interpts : list
        Contains lists of all intersections points in each skeleton.
    hubs : list
        Contains the number of intersections in each filament. This is
        useful for identifying those with no intersections as their analysis
        is straight-forward.
    ends : list
        Contains the positions of all end points in each skeleton.
    filbranches : list
        Contains the number of branches in each skeleton.
    labelisofil : list
        Contains individual arrays for each skeleton where the
        branches are labeled and the intersections have been removed.
    '''

    interpts = []
    hubs = []
    ends = []
    filbranches = []
    labelisofil = []

    for n in range(num):
        funcreturn = find_filpix(1, isolatefilarr[n], final=False, debug=debug)
        interpts.append(funcreturn[1])
        hubs.append(len(funcreturn[1]))
        isolatefilarr.pop(n)
        isolatefilarr.insert(n, funcreturn[2])
        ends.append(funcreturn[3])

        # isolatefilarr contains end and body pts. We need to make sure the end
        # points don't touch, and if they do, label them independently of the
        # others. Touching endpoints can only occur for 1-pixel branches

        # First label the end points
        ends_arr = np.zeros_like(isolatefilarr[n])
        for end in ends[n]:
            ends_arr[end[0], end[1]] = True
        end_label, num_end = nd.label(ends_arr, eight_con())
        # If none are connected, label normally
        if len(ends[n]) == num_end:
            label_branch, num_branch = nd.label(isolatefilarr[n], eight_con())
        else:
            # Find the connected ends, then label and remove them
            conn_ends = np.where(nd.sum(ends_arr, end_label,
                                        range(1, num_end + 1)) > 1)[0] + 1
            if conn_ends.size == 0:
                raise ValueError("This should not be possible, since "
                                 "num_end != len(ends[n]), but no connected"
                                 " structure was found? Possible bug.")

            fil_arr = isolatefilarr[n].copy()

            end_branches = []
            for conn in conn_ends:
                end_posns = np.where(end_label == conn)
                for posn in zip(*end_posns):
                    end_branches.append(posn)
                fil_arr[end_posns] = False

            # Label the version without those ends
            label_branch, num_branch = nd.label(fil_arr, eight_con())

            # Now re-add those connected ends with a new label
            for i, posn in enumerate(end_branches):
                label_branch[posn[0], posn[1]] = num_branch + i + 1

            num_branch = label_branch.max()

        filbranches.append(num_branch)
        labelisofil.append(label_branch)

    return interpts, hubs, ends, filbranches, labelisofil


def extremum_pts(labelisofil, extremum, ends):
    '''
    This function returns the the farthest extents of each filament. This
    is useful for determining how well the shortest path algorithm has worked.

    Parameters
    ----------
    labelisofil : list
        Contains individual arrays for each skeleton.
    extremum : list
       Contains the extents as determined by the shortest
       path algorithm.
    ends : list
        Contains the positions of each end point in eahch filament.

    Returns
    -------
    extrem_pts : list
        Contains the indices of the extremum points.
    '''

    num = len(labelisofil)
    extrem_pts = []

    for n in range(num):
        per_fil = []
        for i, j in ends[n]:
            if labelisofil[n][i, j] == extremum[n][0] or labelisofil[n][i, j] == extremum[n][1]:
                per_fil.append([i, j])
        extrem_pts.append(per_fil)

    return extrem_pts


def make_final_skeletons(labelisofil, inters, verbose=False, save_png=False,
                         save_name=None):
    '''
    Creates the final skeletons outputted by the algorithm.

    Parameters
    ----------
    labelisofil : list
        List of labeled skeletons.
    inters : list
        Positions of the intersections in each skeleton.
    verbose : bool, optional
        Enables plotting of the final skeleton.
    save_png : bool, optional
        Saves the plot made in verbose mode. Disabled by default.
    save_name : str, optional
        For use when ``save_png`` is enabled.
        **MUST be specified when ``save_png`` is enabled.**

    Returns
    -------
    filament_arrays : list
        List of the final skeletons.
    '''

    filament_arrays = []

    for n, (skel_array, intersec) in enumerate(zip(labelisofil, inters)):
        copy_array = np.zeros(skel_array.shape, dtype=int)

        for inter in intersec:
            for pts in inter:
                x, y = pts
                copy_array[x, y] = 1

        copy_array[np.where(skel_array >= 1)] = 1

        cleaned_array = find_extran(1, copy_array)

        filament_arrays.append(cleaned_array)

        if verbose or save_png:
            if save_png and save_name is None:
                ValueError("Must give a save_name when save_png is enabled. No"
                           " plots will be created.")

            p.clf()
            p.imshow(cleaned_array, origin='lower', interpolation='nearest')

            if save_png:
                p.savefig(save_name)
                p.close()
            if verbose:
                p.show()
            if in_ipynb():
                p.clf()

    return filament_arrays


def recombine_skeletons(skeletons, offsets, orig_size, pad_size):
    '''
    Takes a list of skeleton arrays and combines them back into
    the original array.

    Parameters
    ----------
    skeletons : list
        Arrays of each skeleton.
    offsets : list
        Coordinates where the skeleton arrays have been sliced from the
        image.
    orig_size : tuple
        Size of the image.
    pad_size : int
        Size of the array padding.

    Returns
    -------
    master_array : numpy.ndarray
        Contains all skeletons placed in their original positions in the image.
    '''

    num = len(skeletons)

    master_array = np.zeros(orig_size)
    for n in range(num):
        # These are the coordinates of the bottom left in the master array.
        x_off, y_off = offsets[n][0]
        x_top, y_top = offsets[n][1]

        # Now check if padding will put the array outside of the original array
        # size
        excess_x_top = x_top - orig_size[0]

        excess_y_top = y_top - orig_size[1]

        copy_skeleton = copy.copy(skeletons[n])

        if excess_x_top > 0:
            copy_skeleton = copy_skeleton[:-excess_x_top, :]

        if excess_y_top > 0:
            copy_skeleton = copy_skeleton[:, :-excess_y_top]

        if x_off < 0:
            copy_skeleton = copy_skeleton[-x_off:, :]
            x_off = 0

        if y_off < 0:
            copy_skeleton = copy_skeleton[:, -y_off:]
            y_off = 0

        x, y = np.where(copy_skeleton >= 1)
        for i in range(len(x)):
            master_array[x[i] + x_off, y[i] + y_off] = 1

    return master_array


def _fix_small_holes(mask_array, rel_size=0.1):
    '''
    Helper function to remove only small holes within a masked region.

    Parameters
    ----------
    mask_array : numpy.ndarray
        Array containing the masked region.

    rel_size : float, optional
        If < 1.0, sets the minimum size a hole must be relative to the area
        of the mask. Otherwise, this is the maximum number of pixels the hole
        must have to be deleted.

    Returns
    -------
    mask_array : numpy.ndarray
        Altered array.
    '''

    if rel_size <= 0.0:
        raise ValueError("rel_size must be positive.")
    elif rel_size > 1.0:
        pixel_flag = True
    else:
        pixel_flag = False

    # Find the region area
    reg_area = len(np.where(mask_array == 1)[0])

    # Label the holes
    holes = np.logical_not(mask_array).astype(float)
    lab_holes, n_holes = nd.label(holes, eight_con())

    # If no holes, return
    if n_holes == 1:
        return mask_array

    # Ignore area outside of the region.
    out_label = lab_holes[0, 0]
    # Set size to be just larger than the region. Thus it can never be
    # deleted.
    holes[np.where(lab_holes == out_label)] = reg_area + 1.

    # Sum up the regions and find holes smaller than the threshold.
    sums = nd.sum(holes, lab_holes, range(1, n_holes + 1))
    if pixel_flag:  # Use number of pixels
        delete_holes = np.where(sums < rel_size)[0]
    else:  # Use relative size of holes.
        delete_holes = np.where(sums / reg_area < rel_size)[0]

    # Return if there is nothing to delete.
    if len(delete_holes) == 0:
        return mask_array

    # Add one to take into account 0 in list if object label 1.
    delete_holes += 1
    for label in delete_holes:
        mask_array[np.where(lab_holes == label)] = 1

    return mask_array


def is_blockpoint(vallist):
    '''
    Determine if point is part of a block:
    [X X]
    [X X]

    Will have 3 connected sides, with one as an 8-connection.
    '''

    vals = np.array(vallist)

    if vals.sum() < 3:
        return False

    arrangements = [np.array([0, 1, 7]), np.array([1, 2, 3]),
                    np.array([3, 4, 5]), np.array([5, 6, 7])]

    posns = np.where(vals)[0]

    # Check if all 3 in an arrangement are within the vallist
    for arrange in arrangements:
        if np.in1d(posns, arrange).sum() == 3:
            return True

    return False


def is_tpoint(vallist):
    '''
    Determine if point is part of a block:
    [X X X]
    [0 X 0]
    And all 90 deg rotation of this shape

    If there are only 3 connections, this is an end point. If there are 4,
    it is a body point, and if there are 5, it remains an intersection

    '''

    vals = np.array(vallist)

    if vals.sum() < 3:
        return False

    arrangements = [np.array([0, 6, 7]), np.array([0, 1, 2]),
                    np.array([2, 3, 4]), np.array([4, 5, 6])]

    posns = np.where(vals)[0]

    # Check if all 3 in an arrangement are within the vallist
    for arrange in arrangements:
        if np.in1d(posns, arrange).sum() == 3:
            return True

    return False


def merge_nodes(node, G):
    '''
    Combine a node into its neighbors.
    '''

    neigb = list(G[node])

    if len(neigb) != 2:
        return G

    new_weight = G[node][neigb[0]]['weight'] + \
        G[node][neigb[1]]['weight']

    G.remove_node(node)
    G.add_edge(neigb[0], neigb[1], weight=new_weight)

    return G
