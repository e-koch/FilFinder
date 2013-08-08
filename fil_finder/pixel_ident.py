#!/usr/bin/python


'''
Pixel Identification Routines for fil-finder package

The pixels considered are those on the skeletons only. For use only with a skeletonized image.

Contains:
		makefilamentsappear
		isolatefila
		find_filpix
		find_extran

		pix_identify


Requires:
		numpy
		skimage
		ndimage

'''






import numpy as np
#from skimage.filter import threshold_adaptive
import scipy.ndimage as nd
from length import *
import matplotlib.pyplot as p
from skimage.morphology import medial_axis
import skimage.filter as skfilter

#From skimage, which cannot be imported on the server for various reasons.
def threshold_adaptive(image, block_size, method='gaussian', offset=0,mode='reflect', param=None):

    return skfilter.threshold_adaptive(image, block_size, method='gaussian', offset=0,mode='reflect', param=None)


def makefilamentsappear(thearray,size,abs_thresh):
  #Adaptive thresholding is used to segregate filaments. The thresholded image is passed to a median filter to eliminate extraneous spurs when the skeleton is taken.
  size = float(size)
  abs_thresh = float(abs_thresh)

  from scipy.stats import scoreatpercentile
  abs_filter = nd.median_filter(thearray>scoreatpercentile(np.ravel(thearray[~np.isnan(thearray)]),abs_thresh),size=32,mode='mirror')
  # For particularly noisy images where the signal is only discernible near the sources, adaptive thresholding fails, so only abs thresh is used
  adapt_filter = threshold_adaptive(thearray,size,'median')

  if np.sum(adapt_filter)/float(len(np.ravel(adapt_filter)))<=0.02:
    print "Adaptive Threshold Fail"
    filter_full = abs_filter
  else:
    medfilter = nd.median_filter(adapt_filter,size=32,mode='mirror') ## 32 rids extraneous spurs, while preserving shape of region
    filter_full = abs_filter * medfilter #*

  return filter_full


def isolatefilaments(skel_img,mask,size_threshold):
  '''
  Separates each filament, over a threshold of number of pixels, into its own array with the same dimensions as the inputed image.
  skel_img is the result of the Medial Axis Transform
  mask is the image used to make skel_img
  sep_arr allows for just the number of and overall labels filaments to be returned as nd.label will not ignore small regions
  Size_threshold sets the pixel size on the size of objects
  '''

  filarrays = []; pix_val = []; corners = []
  labels,num = nd.label(skel_img,eight_con())
  labels_mask,num_mask = nd.label(mask,eight_con())
  if num_mask!=num: raise ValueError('The number of objects must match the number of skeletons.')
  sums = nd.sum(skel_img,labels,range(num))
  for n in range(num):
    if sums[n]<size_threshold: ## Less than 10 pixels only?? Add a parameter
      x,y = np.where(labels==n)
      for i in range(len(x)):
        if labels_mask[x[i],y[i]]==skel_img[x[i],y[i]]: #Make sure each label array has the same label
          mask_n = n
        else: mask_n = labels_mask[x[i],y[i]]
        skel_img[x[i],y[i]]=0
      x,y = np.where(labels_mask==mask_n)
      for i in range(len(x)):
        mask[x[i],y[i]]=0

  labels,num = nd.label(skel_img,eight_con())
  # eachfil = np.zeros((skel_img.shape)) ## this needs to be scaled to the size of the filament due to memory concerns
  for n in range(1,num+1):
    x,y = np.where(labels==n)
    lower = (x.min()-10,y.min()-10)
    upper = (x.max()+10,y.max()+10)
    shapes = (upper[0]-lower[0],upper[1]-lower[1])
    eachfil = np.zeros(shapes)
    for i in range(len(x)):
      eachfil[x[i]-lower[0],y[i]-lower[1]] = 1
    filarrays.append(eachfil)
    corners.append([lower,upper])
    eachfil = np.zeros((skel_img.shape))
  return filarrays,mask,num,corners



def find_filpix(branches,labelfil,final=True):
  # find_filpix takes identifies the types of pixels contained in the skeleton. This is done by creating lists of the pixel values surrounding the pixel to be determined.
# Eg. a 3x3 array about a pixel is [1,0,1]
# 				   [0,1,0] creating a list of [0,0,1,0,1,0,0,1]
#				   [0,1,0]
# by taking the pixel values that surround the pixel. The list is then shifted once to the right giving [1,0,0,1,0,1,0,0]. The shifted list is subtracted from the original giving [-1,0,1,-1,1,-1,0,1].
#The number of 1s (or -1s) give the amount of step-ups around the pixel. By comparing the step-ups and the number of non-zero elements in the original list, the pixel can be identified into a category.
  initslices = [];initlist = []; shiftlist = [];sublist = [];endpts = [];blockpts = []
  bodypts = [];slices = []; vallist = [];shiftvallist=[];cornerpts = [];delete = []
  subvallist = []
  subslist = [];pix = [];filpix = [];intertemps = [];fila_pts = [];blocks = [];corners = []
  filpts = [];group = [];endpts_return = [];nodes = [];inters = [];repeat = []
  temp_group = [];replace = [];all_pts = [];pairs = []

  for k in range(1,branches+1):
    x,y = np.where(labelfil==k)
    for i in range(len(x)):
      if x[i]<labelfil.shape[0]-1 and y[i]<labelfil.shape[1]-1:
	  pix.append((x[i],y[i]))
          initslices.append(np.array([[labelfil[x[i]-1,y[i]+1],labelfil[x[i],y[i]+1],labelfil[x[i]+1,y[i]+1]],[labelfil[x[i]-1,y[i]],0,labelfil[x[i]+1,y[i]]],[labelfil[x[i]-1,y[i]-1],labelfil[x[i],y[i]-1],labelfil[x[i]+1,y[i]-1]]]))

    filpix.append(pix)
    slices.append(initslices)
    initslices = [];pix= []

  for i in range(len(slices)):
    for k in range(len(slices[i])):
      initlist.append([slices[i][k][0,0],slices[i][k][0,1],slices[i][k][0,2],slices[i][k][1,2],slices[i][k][2,2],slices[i][k][2,1],slices[i][k][2,0],slices[i][k][1,0]])
    vallist.append(initlist)
    initlist = []

  for i in range(len(slices)):
    for k in range(len(slices[i])):
      shiftlist.append(shifter(vallist[i][k],1))
    shiftvallist.append(shiftlist)
    shiftlist = []

  for k in range(len(slices)):
    for i in range(len(vallist[k])):
      for j in range(8):
        sublist.append(int(vallist[k][i][j])-int(shiftvallist[k][i][j]))
      subslist.append(sublist)
      sublist = []
    subvallist.append(subslist)
    subslist = []
# x represents the subtracted list (step-ups) and y is the values of the surrounding pixels. The categories of pixels are ENDPTS (x<=1), BODYPTS (x=2,y=2),CORNERPTS (x=2,y=3),BLOCKPTS (x=3,y>=4), and INTERPTS (x>=3).
# A cornerpt is [*,0,0] (*s) associated with an intersection, but their exclusion from
#		[1,*,0] the intersection keeps eight-connectivity, they are included
#		[0,1,0] intersections for this reason.
# A blockpt is  [1,0,1] They are typically found in a group of four, where all four
#		[0,*,*] constitute a single intersection.
#		[1,*,*]
# The "final" designation is used when finding the final branch lengths. At this point, blockpts and cornerpts should be eliminated.
  for k in range(len(slices)):
    for l in range(len(filpix[k])):
      x = [j for j,y in enumerate(subvallist[k][l]) if y==k+1]
      y = [j for j,z in enumerate(vallist[k][l]) if z==k+1]

      if len(x)<=1:
          endpts.append(filpix[k][l])
	  endpts_return.append(filpix[k][l])
      elif len(x)==2:
	if final:
	  bodypts.append(filpix[k][l])
	else:
          if len(y)==2:
            bodypts.append(filpix[k][l])
	  elif len(y)==3:
	    cornerpts.append(filpix[k][l])
	  elif len(y)>=4:
	    blockpts.append(filpix[k][l])
      elif len(x)>=3:
          intertemps.append(filpix[k][l])
    endpts = list(set(endpts))
    bodypts = list(set(bodypts))
    dups = set(endpts) & set(bodypts)
    if len(dups)>0:
          for i in dups:
            bodypts.remove(i)
#Cornerpts without a partner diagonally attached can be included as a bodypt.
    if len(cornerpts)>0:
      for i in cornerpts:
	for j in cornerpts:
          if i !=j:
	    if distance(i[0],j[0],i[1],j[1])==np.sqrt(2.0):
	      proximity = [(i[0],i[1]-1),(i[0],i[1]+1),(i[0]-1,i[1]),(i[0]+1,i[1]),(i[0]-1,i[1]+1),(i[0]+1,i[1]+1),(i[0]-1,i[1]-1),(i[0]+1,i[1]-1)]
	      match = set(intertemps) & set(proximity)
	      if len(match)==1:
	        pairs.append([i,j])
	        cornerpts.remove(i);cornerpts.remove(j)
    if len(cornerpts)>0:
      for l in cornerpts:
	proximity = [(l[0],l[1]-1),(l[0],l[1]+1),(l[0]-1,l[1]),(l[0]+1,l[1]),(l[0]-1,l[1]+1),(l[0]+1,l[1]+1),(l[0]-1,l[1]-1),(l[0]+1,l[1]-1)]
	match = set(intertemps) & set(proximity)
	if len(match)==1:
	  intertemps.append(l)
	else:
          fila_pts.append(endpts+bodypts+[l]);endpts = [];bodypts = []
          cornerpts.remove(l)
    else: fila_pts.append(endpts+bodypts);endpts = [];bodypts = []
    cornerpts = []

    if len(pairs)>0:
        for i in range(len(pairs)):
          for j in pairs[i]:
            all_pts.append(j)
    if len(blockpts)>0:
        for i in blockpts:
          all_pts.append(i)
    if len(intertemps)>0:
        for i in intertemps:
          all_pts.append(i)
# Pairs of cornerpts, blockpts, and interpts are combined into an array. If there is eight connectivity between them, they are labelled as a single intersection.
    arr = np.zeros((labelfil.shape))
    for z in all_pts:
      labelfil[z[0],z[1]]=0
      arr[z[0],z[1]]=1
    lab,nums = nd.label(arr,eight_con())
    for k in range(1,nums+1):
	  objs_pix = np.where(lab==k)
	  for l in range(len(objs_pix[0])):
	      temp_group.append((objs_pix[0][l],objs_pix[1][l]))
          inters.append(temp_group);temp_group = []
  for i in range(len(inters)-1):
    if inters[i]==inters[i+1]: repeat.append(inters[i])
  for i in repeat:
    inters.remove(i)

  return fila_pts,inters,labelfil,endpts_return

def find_extran(branches,labelfil):
  # The purpose of find_extran is to indentify pixels that are not neceassary to keep the connectivity of the skeleton.
  #It uses the same process as find_filpix. Extraneous pixels tend to be those from former intersections whose attached branch was eliminated in the cleaning process.
  initslices = [];initlist = []; shiftlist = [];sublist = [];extran= []
  slices = []; vallist = [];shiftvallist=[]
  subvallist = []
  subslist = [];pix = [];filpix = [];filpts = []

  for k in range(1,branches+1):
    x,y = np.where(labelfil==k)
    for i in range(len(x)):
      if x[i]<labelfil.shape[0]-1 and y[i]<labelfil.shape[1]-1:
	  pix.append((x[i],y[i]))
          initslices.append(np.array([[labelfil[x[i]-1,y[i]+1],labelfil[x[i],y[i]+1],labelfil[x[i]+1,y[i]+1]],[labelfil[x[i]-1,y[i]],0,labelfil[x[i]+1,y[i]]],\
            [labelfil[x[i]-1,y[i]-1],labelfil[x[i],y[i]-1],labelfil[x[i]+1,y[i]-1]]]))

    filpix.append(pix)
    slices.append(initslices)
    initslices = [];pix= []

  for i in range(len(slices)):
    for k in range(len(slices[i])):
      initlist.append([slices[i][k][0,0],slices[i][k][0,1],slices[i][k][0,2],slices[i][k][1,2],slices[i][k][2,2],slices[i][k][2,1],slices[i][k][2,0],slices[i][k][1,0]])
    vallist.append(initlist)
    initlist = []

  for i in range(len(slices)):
    for k in range(len(slices[i])):
      shiftlist.append(shifter(vallist[i][k],1))
    shiftvallist.append(shiftlist)
    shiftlist = []

  for k in range(len(slices)):
    for i in range(len(vallist[k])):
      for j in range(8):
        sublist.append(int(vallist[k][i][j])-int(shiftvallist[k][i][j]))
      subslist.append(sublist)
      sublist = []
    subvallist.append(subslist)
    subslist = []

  for k in range(len(slices)):
    for l in range(len(filpix[k])):
      x = [j for j,y in enumerate(subvallist[k][l]) if y==k+1]
      y = [j for j,z in enumerate(vallist[k][l]) if z==k+1]
      if len(x)==0:
	labelfil[filpix[k][l][0],filpix[k][l][1]]=0
      if len(x)==1:
        if len(y)>=2:
	  extran.append(filpix[k][l])
	  labelfil[filpix[k][l][0],filpix[k][l][1]]=0
    if len(extran)>=2:
      for i in extran:
	for j in extran:
          if i !=j:
	    if distance(i[0],j[0],i[1],j[1])==np.sqrt(2.0):
	      proximity = [(i[0],i[1]-1),(i[0],i[1]+1),(i[0]-1,i[1]),(i[0]+1,i[1]),(i[0]-1,i[1]+1),(i[0]+1,i[1]+1),(i[0]-1,i[1]-1),(i[0]+1,i[1]-1)]
	      match = set(filpix[k]) & set(proximity)
	      if len(match)>0:
		for z in match:
		  labelfil[z[0],z[1]]=0
  return labelfil


######################################################################
###				Composite Functions
######################################################################


def pix_identify(isolatefilarr,num):
  '''
    Inputs: isolatefilarr - list of segmented filaments
            num           - number of filaments (doesn't really need to be an input since len(isolatefilarr) would work)
  '''
	#Initialize lists
  interpts = []
  hubs = []
  ends = []
  filbranches=  []
  labelisofil = []

  for n in range(num):
		funcreturn = find_filpix(1, isolatefilarr[n], final=False)
  		interpts.append(funcreturn[1])
  		hubs.append(len(funcreturn[1]))
  		isolatefilarr.pop(n)
  		isolatefilarr.insert(n,funcreturn[2])
  		ends.append(funcreturn[3])

  		label_branch,num_branch = nd.label(isolatefilarr[n],eight_con())
  		filbranches.append(num_branch)
  		labelisofil.append(label_branch)

  return interpts, hubs, ends, filbranches, labelisofil

def extremum_pts(labelisofil,extremum,ends):
  # Returns the the farthest points of the filament
  # For use in "global gradient" finding
  num = len(labelisofil)
  # Initialize List
  extrem_pts = []

  for n in range(num):
    per_fil = []
    for i,j in ends[n]:
      if labelisofil[n][i,j]==extremum[n][0] or labelisofil[n][i,j]==extremum[n][1]:
        per_fil.append([i,j])
    extrem_pts.append(per_fil)

  return extrem_pts




if __name__ == "__main__":
    import sys
    fib(int(sys.argv[1]))




























