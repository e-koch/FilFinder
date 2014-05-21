#!/usr/bin/python


'''
Skeleton Length Routines for fil-finder package


Contains:
        fil_length
        init_lengths
        pre_graph
        longest_path
        final_lengths



Requires:
        numpy
        networkx


'''




import numpy as np
from scipy.stats import nanmean
import networkx as nx
from utilities import *
from pixel_ident import *
#from curvature import *
import operator,string, copy




def skeleton_length(skeleton):
    '''
    Length finding via morphological operators.
    '''

    # 4-connected labels
    four_labels = label(skeleton, 4, background=0)

    four_sizes = nd.sum(skeleton, four_labels, range(np.max(four_labels)+1))

    # Lengths is the number of pixels minus number of objects with more
    # than 1 pixel.
    four_length = np.sum(four_sizes[four_sizes>1]) - len(four_sizes[four_sizes>1])

    # Find pixels which a 4-connected and subtract them off the skeleton

    four_objects = np.where(four_sizes>1)[0]

    skel_copy = copy.copy(skeleton)
    for val in four_objects:
        skel_copy[np.where(four_labels==val)] = 0

    # Remaining pixels are only 8-connected
    # Lengths is same as before, multiplied by sqrt(2)

    eight_labels = label(skel_copy, 8, background=0)

    eight_sizes = nd.sum(skel_copy, eight_labels, range(np.max(eight_labels)+1))

    eight_length = ((np.sum(eight_sizes)-1) - np.max(eight_labels)) * np.sqrt(2)

    # If there are no 4-connected pixels, we don't need the hit-miss portion.
    if four_length==0.0:
        conn_length = 0.0

    else:

        # Check 4 to 8-connected elements
        struct1 = np.array([[1, 0, 0],
                            [0, 1 ,1],
                            [0, 0, 0]])

        struct2 = np.array([[0, 0, 1],
                            [1, 1 ,0],
                            [0, 0, 0]])

        # Next check the three elements which will be double counted
        check1 = np.array([[1, 1, 0, 0],
                           [0, 0, 1, 1]])

        check2 = np.array([[0, 0, 1, 1],
                           [1, 1, 0, 0]])

        check3 = np.array([[1, 1, 0],
                           [0, 0, 1],
                           [0, 0, 1]])

        store = np.zeros(skeleton.shape)

        # Loop through the 4 rotations of the structuring elements
        for k in range(0,4):
            hm1 = nd.binary_hit_or_miss(skeleton, structure1=np.rot90(struct1, k=k))
            store += hm1

            hm2 = nd.binary_hit_or_miss(skeleton, structure1=np.rot90(struct2, k=k))
            store += hm2

            hm_check3 = nd.binary_hit_or_miss(skeleton, structure1=np.rot90(check3, k=k))
            store -=hm_check3

            if k <= 1:
                hm_check1 = nd.binary_hit_or_miss(skeleton, structure1=np.rot90(check1, k=k))
                store -= hm_check1

                hm_check2 = nd.binary_hit_or_miss(skeleton, structure1=np.rot90(check2, k=k))
                store -= hm_check2

        conn_length = np.sqrt(2) * np.sum(np.sum(store, axis=1), axis=0)#hits


    return conn_length + eight_length + four_length

########################################################
###       Composite Functions
########################################################


def init_lengths(labelisofil,filbranches, array_offsets, img):
  '''

  This is a wrapper on fil_length for running on the branches of the skeletons.

  Parameters
  ----------

  labelisofil : list
                Contains individual arrays for each skeleton where the
                branches are labeled and the intersections have been removed.

  filbranches : list
                Contains the number of branches in each skeleton.

  array_offsets : List
                  The indices of where each filament array fits in the original image.

  img : numpy.ndarray
        Original image.

  Returns
  -------

  lengths : list
            Contains the lengths of each branch on each skeleton.

  av_branch_intensity : list
                        Average Intensity along each branch.

  '''
  num = len(labelisofil)

  #Initialize Lists
  lengths = []
  filpts = []
  av_branch_intensity = []

  for n in range(num):
    leng = []
    for branch in range(1, filbranches[n]+1):
      branch_array = np.zeros(labelisofil[n].shape)
      branch_pts = np.where(labelisofil[n]==branch)
      branch_array[branch_pts] = 1
      branch_length = skeleton_length(branch_array)
      if branch_length==0.0:
        leng.append(0.5) # For use in longest path algorithm, will be set to zero for final analysis
      else:
        leng.append(branch_length)

      # Now let's find the average intensity along each branch
      av_intensity = []
      x_offset, y_offset = array_offsets[n][0]
      av_intensity.append(nanmean([img[x+x_offset,y+y_offset] for x,y in zip(*branch_pts)]))

    lengths.append(leng)
    av_branch_intensity.append(av_intensity)

  return lengths, av_branch_intensity




def pre_graph(labelisofil, lengths, branch_intensity, interpts, ends):
  '''

  This function converts the skeletons into a graph object compatible with
  networkx. The graphs have nodes corresponding to end and intersection points
  and edges defining the connectivity as the branches with the weights set
  to the branch length.

  Parameters
  ----------

  labelisofil : list
                Contains individual arrays for each skeleton where the
                branches are labeled and the intersections have been removed.

  lengths : list
            Contains the lengths of all of the branches.

  branch_intensity : list
                     The mean intensities along each branch. Used along
                     with length as a weight.

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

  def path_weighting(idx, length, intensity, w=0.5):
    '''

    Relative weighting for the shortest path algorithm using the branch
    lengths and the average intensity along the branch.

    '''
    if w>1.0 or w<0.0:
      raise ValueError("Relative weighting w must be between 0.0 and 1.0.")
    return  (1-w) * (length[idx]/ np.sum(length)) + w * (intensity[idx]/np.sum(intensity))


  for n in range(num):
    inter_nodes_temp = []
    ## Create end_nodes, which contains lengths, and nodes, which we will later add in the intersections
    end_nodes.append([(labelisofil[n][i[0],i[1]], path_weighting(int(labelisofil[n][i[0],i[1]]-1), lengths[n], branch_intensity[n]),\
                     lengths[n][int(labelisofil[n][i[0],i[1]]-1)], branch_intensity[n][int(labelisofil[n][i[0],i[1]]-1)]) for i in ends[n]])
    nodes.append([labelisofil[n][i[0],i[1]] for i in ends[n]])

  # Intersection nodes are given by the intersections points of the filament.
  # They are labeled alphabetically (if len(interpts[n])>26, subsequent labels are AA,AB,...).
  # The branch labels attached to each intersection are included for future use.
    for intersec in interpts[n]:
        uniqs = []
        for i in intersec: ## Intersections can contain multiple pixels
          int_arr = np.array([[labelisofil[n][i[0]-1,i[1]+1], labelisofil[n][i[0],i[1]+1], labelisofil[n][i[0]+1,i[1]+1]],\
                              [labelisofil[n][i[0]-1,i[1]]  , 0, labelisofil[n][i[0]+1,i[1]]],\
                              [labelisofil[n][i[0]-1,i[1]-1], labelisofil[n][i[0],i[1]-1], labelisofil[n][i[0]+1,i[1]-1]]]).astype(int)
          for x in np.unique(int_arr[np.nonzero(int_arr)]):
            uniqs.append((x,path_weighting(x-1, lengths[n], branch_intensity[n]), lengths[n][x-1], branch_intensity[n][x-1]))
        # Intersections with multiple pixels can give the same branches. Get rid of duplicates
        uniqs = list(set(uniqs))
        inter_nodes_temp.append(uniqs)

    # Add the intersection labels. Also append those to nodes
    inter_nodes.append(zip(product_gen(string.ascii_uppercase),inter_nodes_temp))
    for alpha, node in zip(product_gen(string.ascii_uppercase),inter_nodes_temp):
      nodes[n].append(alpha)

    #Edges are created from the information contained in the nodes.
    edge_list_temp = []
    for i, inters in enumerate(inter_nodes[n]):
      end_match = list(set(inters[1]) & set(end_nodes[n]))
      for k in end_match:
        edge_list_temp.append((inters[0],k[0],k))

      for j, inters_2 in enumerate(inter_nodes[n]):
        if i != j:
          match = list(set(inters[1]) & set(inters_2[1]))
          new_edge = None
          if len(match)==1:
            new_edge = (inters[0],inters_2[0],match[0])
          elif len(match)>1:
            multi = [match[l][1] for l in range(len(match))]
            keep = multi.index(min(multi))
            new_edge = (inters[0],inters_2[0],match[keep])
          if new_edge is not None:
            if not (new_edge[1], new_edge[0], new_edge[2]) in edge_list_temp \
            and new_edge not in edge_list_temp:
              edge_list_temp.append(new_edge)

    # Remove duplicated edges between intersections

    edge_list.append(edge_list_temp)

  return edge_list, nodes



def longest_path(edge_list,nodes,lengths,verbose=False, skeleton_arrays=None):
  '''
  Takes the output of pre_graph and runs the shortest path algorithm.

  Parameters
  ----------

  edge_list : list
              Contains the connectivity information for the graphs.

  nodes : list
          A complete list of all of the nodes. The other nodes lists have
          been separated as they are labeled differently.

  lengths : list
            Contains the lengths of each branch.

  verbose : bool, optional
            If True, enables the plotting of the graph. *Recommend pygraphviz
            be installed for best results.*

  skeleton_arrays : list, optional
                    List of the skeleton arrays. Required when verbose=True.

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
      G.add_edge(i[0],i[1],weight=i[2][1])
    paths = nx.shortest_path_length(G,weight='weight')
    values = [];node_extrema = []
    for i in paths.iterkeys():
      j = max(paths[i].iteritems(),key=operator.itemgetter(1))
      node_extrema.append((j[0],i))
      values.append(j[1])
    start,finish = node_extrema[values.index(max(values))]
    extremum.append([start,finish])
    max_path.append(nx.shortest_path(G,start,finish))
    graphs.append(G)
    if verbose:
      if not skeleton_arrays:
        print "Must input skeleton arrays if verbose=True."
      else:
        # Check if skeleton_arrays is a list
        assert isinstance(skeleton_arrays, list)
        import matplotlib.pyplot as p

        p.subplot(1,2,1)
        p.imshow(skeleton_arrays[n], origin="lower", interpolation=None)

        p.subplot(1,2,2)
        elist = [(u,v) for (u,v,d) in G.edges(data=True)]
        try:
          import pygraphviz
          pos = nx.graphviz_layout(G, arg=str(lengths[n]))
        except ImportError:
          pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos, node_size=200)
        nx.draw_networkx_edges(G, pos, edgelist=elist, width=2)
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
        p.axis('off')
        p.show()

  return max_path, extremum, graphs


def prune_graph(G, nodes, edge_list, max_path, labelisofil, length_thresh, relintens_thresh=0.2):
  '''
  Function to remove unnecessary branches.

  '''

  num = len(labelisofil)

  for n in range(num):
    degree = G[n].degree()
    single_connect = [key for key in degree.keys() if degree[key]==1]

    delete_candidate = list((set(nodes[n]) - set(max_path[n])) & set(single_connect))

    if not delete_candidate: # Nothing to delete!
      return labelisofil, edge_list, nodes

    else:
      edge_candidates = [edge for edge in edge_list[n] if edge[0] in delete_candidate or edge[1] in delete_candidate]
      intensities = [edge[2][3] for edge in edge_list[n]]
      for edge in edge_candidates:
        ## If its too short and relatively not as intense, delete it
        if edge[2][2]<length_thresh and (edge[2][3]/np.sum(intensities))<relintens_thresh:
          x,y = np.where(labelisofil[n]==edge[2][0])
          for i in range(len(x)):
            labelisofil[n][x[i], y[i]] = 0
          edge_list[n].remove(edge)
          nodes[n].remove(edge[1])

  return labelisofil, edge_list, nodes


def final_lengths(img,max_path,edge_list,labelisofil,filpts,interpts,filbranches,lengths,img_scale,length_thresh):
  '''
  The function finds the overall length of the filament from the longest_path
  and pre_graph outputs. For intersections of more than one pixel, it calculates
  a weighted average (based on intensity) and uses that position for calculating
  the length. The curvature routines are also wrapped in this function.

  This function also performs the "pruning". Branches that are less than
  length_thresh are deleted.

  Parameters
  ----------

  img : numpy.ndarray
        The image being analyzed.

  max_path : list
             Contains the longest paths through the skeleton.

  edge_list : list
              Contains the connectivity of the graphs

  labelisofil : list
                Contains individual arrays for each skeleton where the
                branches are labeled and the intersections have been removed.

  filpts : list
           Contains the pixels belonging to each skeleton.

  interpts : list
             Contains the pixels which belong to the intersections.

  filbranches : list
                Contains the number of branches in each filament.

  lengths : list
            Contains the lengths of the branches.

  img_scale : float
              The conversion to physical units (pc).

  length_thresh : float
                  The minimum length a branch must be.

  Returns
  -------

  main_lengths : list
                 Contains the overall skeleton lengths.

  lengths : list
            The updated list of the lengths of the skeleton branches.

  labelisofil : list
                The updated versions of the skeleton arrays. The intersection
                points have been re-added and the short branches pruned off.

  '''
  num = len(max_path)

  # Initialize lists
  main_lengths = []

  for n in range(num):

    if len(max_path[n])==1: #Catch filaments with no intersections
      main_lengths.append(lengths[n][0] * img_scale)
    else:
      good_edge_list = [(max_path[n][i],max_path[n][i+1]) for i in range(len(max_path[n])-1)]
      # Find the branches along the longest path.
      keep_branches = []
      for i in good_edge_list:
        for j in edge_list[n]:
          if (i[0]==j[0] and i[1]==j[1]) or (i[0]==j[1] and i[1]==j[0]):
            keep_branches.append(j[2][0])
      # Each branch label is duplicated, get rid of extras
      keep_branches = list(set(keep_branches))
      fils = [filpts[n][int(i-1)] for i in keep_branches]

      branches = np.unique(labelisofil[n][np.nonzero(labelisofil[n])])
      # Find the branches which are not in keep_branches, then delete them from labelisofil array
      delete_branches = list(set(branches) ^ set(keep_branches))
      for branch in delete_branches:
        x,y = np.where(labelisofil[n]==branch)
        for i in range(len(x)):
          labelisofil[n][x[i],y[i]]=0
      # A "big_inter" is any intersection which contains multiple pixels
      big_inters = []
      for intersec in interpts[n]:
        if len(intersec)>1:
          big_inters.append(intersec)
        for pix in intersec:
          labelisofil[n][pix]=filbranches[n]+1
      # find_pilpix is used again to find the end-points of the remaining branches. The
      # branch labels are used to check which intersections are included in the longest
      #path. For intersections containing multiple points, an average of the
      #positions, weighted by their value in the image, is used in the length
      #calculation.
      relabel, numero = nd.label(labelisofil[n],eight_con())
      endpts = find_filpix(numero,relabel,final=False)[3]
      for intersec in interpts[n]:
        match = list(set(endpts) & set(intersec)) # Remove duplicates in endpts
        if len(match)>0:
          for h in match:
            endpts.remove(h)
      for i in big_inters:
        weight = [];xs = [];ys = []
        for x,y in i:
          weight.append(img[x,y])
          xs.append(x);ys.append(y)
        av_x = weighted_av(xs,weight)
        av_y = weighted_av(ys,weight)
        interpts[n].insert(interpts[n].index(i),[(av_x,av_y)])
        interpts[n].remove(i)
      # The pixels of the longest path are combined with the intersection pixels. This gives overall length of the filament.
      good_pts = [];[[good_pts.append(i[j]) for j in range(len(i))]for i in fils]
      match = list(set(endpts) & set(good_pts))
      if len(match)>0:
        for i in match:
          good_pts.remove(i)
      for i in endpts:
        good_pts.insert(0,i)

      intersec_labels = [zip(product_gen(string.ascii_uppercase),range(len(interpts[n])))[0]]
      inter_find = list(set(max_path[n]) & set(intersec_labels))

      good_inter = []
      if len(inter_find) != 0:
        for i in inter_find:
          good_inter.append(interpts[n][intersec_labels.index(i)])
      interpts[n] = [];[[interpts[n].append(i[j]) for j in range(len(i))] for i in good_inter]
      finalpix = [good_pts + interpts[n]]
      lengthh,order = fil_length(n,finalpix,initial=False)
      main_lengths.append(lengthh[0] * img_scale)

      # Re-adding all deleted branches
      for i in delete_branches:
          for x,y in filpts[n][i-1]:
            labelisofil[n][x,y]=i
            good_pts.insert(i,filpts[n][i-1])

  return main_lengths, lengths, labelisofil


def final_analysis(labelisofil):
  '''

  In the case that a skeleton has been split into parts during the pruning
  process in final_lengths, this function re-analyzes and corrects any such
  problems.

  Parameters
  ----------

  labelisofil : list
                The versions of the skeleton arrays outputted from final_lengths.

  Returns
  -------

  labelisofil : list
                The updated version of the skeleton arrays.

  filbranches : list
                The updated version of the number of branches in each skeleton.

  hubs : list
         The updated version of the number of intersections in each skeleton.

  lengths : list
            Updated version of the lengths of the branches.


  '''
  num = len(labelisofil)

  # Initialize lists
  filbranches = []
  hubs = []
  lengths = []
  filament_arrays = []

  for n in range(num):
    x,y = np.where(labelisofil[n]>0)
    for i in range(len(x)):
      labelisofil[n][x[i],y[i]]=1
    deletion = find_extran(1,labelisofil[n])
    filament_arrays.append(copy.copy(deletion)) # A cleaned, final skeleton is returned here.
    funcreturn = find_filpix(1,deletion,final=False)
    relabel,num_branches = nd.label(funcreturn[2],eight_con())
    for i in range(num_branches):
      x,y = np.where(relabel==num_branches+1)
      if len(x)==1:
        labelisofil[x[0],y[0]]=0
    labelisofil.pop(n)
    labelisofil.insert(n,relabel)
    filbranches.insert(n,num_branches)
    hubs.append(len(funcreturn[1]))
    funcsreturn = find_filpix(num_branches,relabel,final=True)
    lenn,ordd = fil_length(n,funcsreturn[0],initial=True)
    lengths.append(lenn)

  return labelisofil, filbranches, hubs, lengths, filament_arrays
