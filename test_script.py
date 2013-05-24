#!/usr/bin/python

'''
Test Script for fil_finder

Includes all working portions.

Run from command line as: test_script.py image.fits
'''
import numpy as np
import matplotlib.pyplot as p
import sys


## Import fil_finder
from fil_finder import *

## Load the image
img,hdr = fromfits(sys.argv[1])

## Set the distance to the object to find the scale and beamwidth
dist_to_img = 150.0 # pc
bmwdth = 18.1 # "
try:
  img_freq = (3*10**14)/hdr["WAVE"] # hopefully the header has this
except KeyError:
  img_freq = 110201.3541 # Line of 13CO, which I'm testing this on, Feb.6/13
img_scale = (hdr['CDELT2']*(np.pi/180.0)*dist_to_img)
img_beam = (bmwdth/np.sqrt(8*np.log(2.))) * (2*np.pi / 206265.) * dist_to_img # FWHM beamwidth

## Segment filamentary structure
## NOTE: the inputs are the image, the size of areas used for adaptive thresholding, the percentile to
## globally threshold the image (to get rid of empty regions)
## These settings REALLY need to be played with to find a good combination for the image in use.
mask = makefilamentsappear(img,80,70)

## If only a segment of the image is of interest
slice_img = img#[1500:2300,500:1600] # for polaris-250

## Pad the array by 1 so pixels on the edge can be analyzed
slice_img = np.pad(slice_img,1,padwithzeros)
p.imshow(slice_img);p.show()

mask = mask#[1500:2300,500:1600] # for polaris-250
mask = np.pad(mask,1,padwithzeros)

## Peform a medial_axis transform to get skeleton structure.
medskel = medial_axis(mask)

## Examine how well the filaments have been segmented
mask_img = mask * slice_img
nanned = np.where(medskel==1)
for i in range(len(nanned[0])):
    mask_img[nanned[0][i],nanned[1][i]] = np.NaN
p.imshow(mask_img,interpolation=None,origin=lower);p.show()

## For use with column density. Portion not yet complete.
#thresh_array = abs_thresh(mask_img,2e22,img_scale,img_freq)

## Core subtraction to rid the detected filaments of embedded cores, so further calculations are not affected
## Currently using 2D gaussian fit and setting the area equal to the background
## Fails nearly every time, so has been omitted for the time being.
#slice_img = subtract_cores(thresh_array)

## Separate each filament into its own array
isolatefilarr,mask, num = isolatefila(medskel,mask)
print "Initial Fil # : %s" % (num)

## Here, we label each pixel in each filament based on the surrounding 8 pixels
## interpts are the intersection points, hubs are the number of intersections, ends are the pixels at the end of branches
## labelisofil is a list of arrays with each branch labelled.
interpts, hubs, ends, filbranches, labelisofil = pix_identify(isolatefilarr,num)

## For inspection of labelled skeletons
# for n in labelisofil:
# p.imshow(n,interpolation=None);p.show()

## Calculate the length of the labelled branches
## lengths is a list of lists of the branch lengths for each filament
## filpts is the order of the pixels in each branch
lengths,filpts = init_lengths(labelisofil,filbranches)

## pre_graph takes the outputs of init_lengths and prepares for finding the longest path
## end_nodes and inter_nodes will create the nodes for the graphs. These are labelled separately and thus are separated.
## edge_list are the branches connecting the nodes, using the branch lengths as weights.
end_nodes, inter_nodes, edge_list, nodes = pre_graph(labelisofil,lengths,interpts,ends)

## Finding the longest path using a shortest path algorithm (from networkx)
## Note that we avoid any loops by choosing the longest side in pre_graph
## verbose plots each graph
max_path,extremum = longest_path(edge_list,nodes,lengths,verbose=False)

## Finding the end points of the longest path through the filament
extrem_pts = extremum_pts(labelisofil,extremum,ends)
# print extrem_pts

## Using the longest path, the main lengths of the filaments are calculated
## main_lengths is the length of each filament, lengths are the lengths of the surviving branches (> length_thresh)
## labelisofil are the arrays of cleaned filaments
## curvature is our description of the filament's shape
## This is calculated by choosing 3 random points on the skeleton and calculating the Menger Curvature
## The curvature is the average value of 1000 trials
length_thresh = 3.0 ## based on observing the image, this looks like a good value
main_lengths,lengths,labelisofil,curvature = final_lengths(img,max_path,edge_list,labelisofil,filpts,interpts,filbranches,lengths,img_scale,length_thresh)
# print main_lengths,curvature

## Inaccuracies in length and skeleton structure are checked and corrected
## Final branch lengths, intersections, skeletons are returned
## The outputs are updated versions of the previous descriptions
labelisofil,filbranches,hubs,lengths = final_analysis(labelisofil)

## A distance transform is performed on each filament and the combination of all filaments (to determine which filament a pixel is
## closest to)
dist_transform_all,dist_transform_sep = dist_transform(labelisofil)


## A radial profile is created from the distance transforms
## A gaussian is fit to the profiles, where the mean is forced to be 0 (skeleton is assumed centre)
## widths_gn are the widths, fits_gn are the fit parameters, errors_gn are the errors on each parameter
widths_gn,fits_gn,fit_errors_gn = gauss_width(slice_img,dist_transform_all,dist_transform_sep,img_beam,img_scale,verbose=False)
# print widths_gn

## Add widths onto main filament length, while separating out fit failures for the width
## We assume that the skeleton of the filament is shortened by the width of the filament
overall_lengths = []
overall_widths = []
for i in range(num):
    if isinstance(widths_gn[i],float):
        if main_lengths[i]>widths_gn[i]:
            overall_lengths.append(main_lengths[i] + widths_gn[i])
            overall_widths.append(widths_gn[i])
        else:
            overall_lengths.append(main_lengths[i])
            overall_widths.append("Fit Fail")
    else:
        overall_lengths.append(main_lengths[i])
        overall_widths.append(widths_gn[i])

## The main lengths, branch lengths, average curvature, and widths of the filaments have been calculated.

print overall_lengths
print curvature
print overall_widths

########## Missing printing out table of results, adding FWHM width to length, density and column density calulations