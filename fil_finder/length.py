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




def fil_length(n,pixels,initial=True):
  '''
  This function calculates either the length of the branches, or the entire
  filament. It does this by creating an array of the distances between each
  pixel. It then searches each column and identifies the minimum of that
  row. The column containing the minimum is the next row to be searched.
  After a row is searched, the corresponding row and column are set to
  zero and ignored. When initial is True, the maximum distance between
  connected pixels is sqrt(2). When initial is False, the function accounts
  for the average position of intersections when finding the overall length
  of the filament. Due to the somewhat unpredictable size of the larger
  intersections, the minimum distances are allowed to be much larger than
  sqrt(2). The threshold is set at 5 as an approximation for the largest
  gap that should be created. At this point, the intersections are the
  only places where a distance of sqrt(2) can be returned.

  Parameters
  ----------

  n : int
      The number of the skeleton being analyzed.

  pixels : list
           Contains the positions of the pixels in the skeleton or branch.

  initial : bool, optional
            If True, the initial branches are inputted. If False, the
            entire cleaned skeleton is inputted.

  Returns
  -------

  distances : list
              Contains the length of the inputted structure.

  orders : list
           Contains the order of the pixels.

  '''
  dists = [];distances = [];orders = [];order = []
  for i in range(len(pixels)):
    if pixels[i]==[[]]: pass
    else:
        eucarr = np.zeros((len(pixels[i]),len(pixels[i])))
        for j in range(len(pixels[i])):
          for k in range(len(pixels[i])):
            eucarr[j,k]=np.linalg.norm(map(operator.sub,pixels[i][k],pixels[i][j]))
        for _ in range(len(pixels[i])-1):
          if _== 0:
            j=0
            last=0
          else:
            j=last
            last = []
          try:
            min_dist = np.min(eucarr[:,j][np.nonzero(eucarr[:,j])])
          except ValueError:
            min_dist = 0 # In some cases, a duplicate of the first pixel
                         # is added to the end, causing issues. This takes
                         # corrects the length while I try to find the
                         # issue.
          if initial:
            if min_dist>np.sqrt(2.0):
              print "PROBLEM : Dist %s, Obj# %s,Branch# %s, # In List %s" % (min_dist,n,i,pixels[i][j])
          else:
            if min_dist>5.0:
              if j==1:
                min_dist = 0
          dists.append(min_dist)
          x,y = np.where(eucarr==min_dist)
          order.append(j)
          for z in range(len(y)):
            if y[z]==j:
              last = x[z]
              eucarr[:,j]=0
              eucarr[j,:]=0
    distances.append(sum(dists))
    orders.append(order)
    dists = [];order = []

  return distances, orders


def curve(n,pts):
  '''
  The average curvature of the filament is found using the Menger curvature.
  The formula relates the area of the triangle created by the three points
  and the distance between the points. The formula is given as 4*area/|x-y||y-z||z-x|=curvature.
  The curvature is weighted by the Euclidean length of the three pixels.

  *Note:* The normalization is still an issue with this method. Its results
  should **NOT** be used.

  Parameters
  ----------

  n : int
      The number of the skeleton being analyzed.

  pts : list
        Contains the pixels contained in the inputted structure.

  Returns
  -------

  numer/denom : float
                The value of the Menger Curvature.

  References
  ----------

  '''
  lenn = len(pts)
  kappa = [];seg_len = []
  for i in range(lenn-2):
    x1 = pts[i][0];y1 = pts[i][1]
    x2 = pts[i+1][0];y2 = pts[i+1][1]
    x3 = pts[i+2][0];y3 = pts[i+2][1]
    num = abs(2*((x2-x1)*(y2-y1)+(y3-y2)*(x3-x2)))
    den = np.sqrt( (pow((x2-x1),2) + pow((y2-y1),2)) * (pow((x3-x2),2)+pow((y3-y2),2 ))* (pow((x1-x3),2)+pow((y1-y3),2) ) )
    if ( den == 0 ):
      kappa.append(0)
    else:
      kappa.append(num/den)
    seg_len.append(fil_length(n,[[pts[i],pts[i+1],pts[i+2]]],initial=False)[0])
  numer = sum(kappa[i] * seg_len[i][0] for i in range(len(kappa)))
  denom = sum(seg_len[i][0] for i in range(len(seg_len)))
  if denom!= 0:
    return numer/denom
  else:
    print n
    print pts
    raise ValueError('Sum of length segments is zero.')

def av_curvature(n,finalpix,ra_picks=100,seed=500):
  '''
  This function acts as a wrapper on curve. It calculates the average curvature
  by choosing 3 random points on the filament and calculating the curvature.
  The average of many iterations of this method is reported as the curvature
  for that skeleton.

  Parameters
  ----------

  n : int
      The number of the skeleton being analyzed.

  finalpix : list
             Contains the pixels contained in the inputted structure.

  ra_picks : int
             The number of iterations to run.

  seed : int
         Sets the seed.
  '''
  import numpy.random as ra
  seed = int(seed)
  ra.seed(seed=int(seed))
  ra_picks = int(ra_picks)

  curvature = []

  for i in range(len(finalpix)):
    if len(finalpix[i])>3:
      trials = []
      for _ in range(ra_picks):
        picks = ra.choice(len(finalpix[i]),3,replace=False) ### REQUIRE NUMPY 1.7!!!
        points = [finalpix[i][j] for j in picks]
        trials.append(curve(n,points))
      curvature.append(np.mean(trials))
    else:
      curvature.append("Fail")
  return curvature

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
            Contains the lengths of eahc branch on each skeleton.

  filpts : list
           Contains the pixels in each branch.

  av_branch_intensity : list
                        Average Intensity along each branch.

  '''
  num = len(labelisofil)

  #Initialize Lists
  lengths = []
  filpts = []
  av_branch_intensity = []

  for n in range(num):
    funcreturn = find_filpix(filbranches[n],labelisofil[n],final=False)
    leng = fil_length(n,funcreturn[0],initial=True)[0]
    for i in range(len(leng)):
      if leng[i]==0.0:
        leng.pop(i)
        leng.insert(i,0.5) # For use in longest path algorithm, will be set to zero for final analysis
    lengths.append(leng)
    filpts.append(funcreturn[0])
    # Now let's find the average intensity along each branch
    av_intensity = []
    x_offset, y_offset = array_offsets[n][0]
    for pts in funcreturn[0]:
      av_intensity.append(nanmean([img[pt[0]+x_offset,pt[1]+y_offset] for pt in pts]))
    av_branch_intensity.append(av_intensity)

  return lengths, filpts, av_branch_intensity




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
    if w>1.0:
      raise ValueError("Relative weighting w must be between 0.0 and 1.0.")
    return  (1-w) * (length[idx]/ np.sum(length)) + w * (intensity[idx]/np.sum(intensity))


  for n in range(num):
    inter_nodes_temp = []
    ## Create end_nodes, which contains lengths, and nodes, which we will later add in the intersections
    # end_nodes.append([(labelisofil[n][i[0],i[1]],lengths[n][int(labelisofil[n][i[0],i[1]]-1)]) for i in ends[n]])
    end_nodes.append([(labelisofil[n][i[0],i[1]], path_weighting(int(labelisofil[n][i[0],i[1]]-1), lengths[n], branch_intensity[n])) for i in ends[n]])
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
            uniqs.append((x,lengths[n][x-1]))
        # Intersections with multiple pixels can give the same branches. Get rid of duplicates
        uniqs = list(set(uniqs))
        inter_nodes_temp.append(uniqs)

    # Add the intersection labels. Also append those to nodes
    inter_nodes.append(zip(product_gen(string.ascii_uppercase),inter_nodes_temp))
    nodes[n].append(zip(product_gen(string.ascii_uppercase),inter_nodes_temp)[0])

    #Edges are created from the information contained in the nodes.
    edge_list_temp = []
    for i, inters in enumerate(inter_nodes[n]):
      end_match = list(set(inters[1]) & set(end_nodes[n]))
      for k in end_match:
        edge_list_temp.append((inters[0],k[0],k))

      for j, inters_2 in enumerate(inter_nodes[n]):
        if i != j:
          match = list(set(inters[1]) & set(inters_2[1]))
          if len(match)==1:
            edge_list_temp.append((inters[0],inters_2[0],match[0]))
          elif len(match)>1:
            multi = [match[l][1] for l in range(len(match))]
            keep = multi.index(min(multi))
            edge_list_temp.append((inters[0],inters_2[0],match[keep]))
    edge_list.append(edge_list_temp)


  return end_nodes, inter_nodes, edge_list, nodes



def longest_path(edge_list,nodes,lengths,verbose=False):
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
            If True, enables the plotting of the graph. *Requires pygraphviz
            be installed.*

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
    if verbose:
      import matplotlib.pyplot as p
      clean_graph = p.figure(1.,facecolor='1.0')
      graph = clean_graph.add_subplot(1,2,2)
      elist = [(u,v) for (u,v,d) in G.edges(data=True)]
      pos = nx.graphviz_layout(G)#,arg=str(lengths[n])) # The argument throws an error. I have yet to understand why...
      nx.draw_networkx_nodes(G,pos,node_size=200)
      nx.draw_networkx_edges(G,pos,edgelist=elist,width=2)
      nx.draw_networkx_labels(G,pos,font_size=10,font_family='sans-serif')
      p.axis('off')
      p.show()


  return max_path,extremum


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

  curvature : list
              The Menger Curvature values for the main extents of each skeleton.

  '''
  num = len(max_path)

  # Initialize lists
  curvature = []
  main_lengths = []

  for n in range(num):

    if len(max_path[n])==1: #Catch filaments with no intersections
      main_lengths.append(lengths[n][0] * img_scale)
      curvature.append(av_curvature(n,filpts[n])[0])
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

      branches = range(1,filbranches[n]+1)
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

      curvature.append(av_curvature(n,finalpix)[0]) ### SEE CURVE FOR EXPLANATION

      # Re-adding long branches, "long" greater than the length threshold
      del_length = []
      for i in delete_branches:
        if lengths[n][i-1]> length_thresh:
          for x,y in filpts[n][i-1]:
            labelisofil[n][x,y]=i
            good_pts.insert(i,filpts[n][i-1])
        else:
          del_length.append(lengths[n][i-1])
      lengths[n] = list(set(lengths[n]) - set(del_length))
      filpts[n] = [];[filpts[n].append(i) for i in good_pts]

  return main_lengths, lengths, labelisofil, curvature # Returns the main lengths, the updated branch lengths, the final skeleton arrays, and curvature


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
