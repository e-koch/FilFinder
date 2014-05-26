fil_finder
==========

Description
------------

fil_finder is a module for extraction and analysis of filamentary structure in molecular clouds.

fil_finder segments filamentary structure in an integrated intensity image using adaptive thresholding.
Detected regions are reduced to a skeleton using a Medial Axis Transform.
Pixels within each skeleton are classified by the number of connecting pixels.
A pixel can be a body point, end point, or intersection point.
A shortest path algorithm, weighted by the intensity and length, finds the longest path through the skeleton, which is reported as the main length.
At this point, branches less than a length threshold are removed to give a final, cleaned skeleton.
A Euclidean Distance Transform is performed to build a radial profile of each filament.
A Gaussian with a constant background is fit to the profile to find the width.
The filament width is the FWHM value after deconvolving with the FWHM beamwidth of the instrument.
The curvature of the filament is described using the Rolling Hough Transform (Clark et al., 2013) is used.
This method returns the direction of the filament on the plane (median of the RHT) and the curvature (IQR of the RHT).
Also included in the package is an alternate method to find curvature based on the Menger Curvature Formula. This method does NOT yet return a reliable value and we recommend using the RHT method.

Example Images
--------------

Data shown below is from the Herschel Gould Belt Survey (Andre et al. 2010).

![Chameleon-250 Scaled to 2150](images/chameleon-250-filcontours-2150.png "Chameleon-250 Scaled to 2200")

![Chameleon-250 Scaled to 2500](images/chameleon-250-filcontours-2500.png "Chameleon-250 Scaled to 2500")

Package Dependencies
--------------------

Requires:

 *   numpy 1.7.1
 *   matplotlib
 *   astropy
 *   scipy
 *   scikits-image 0.8.0
 *   networkx
 *   pandas

Optional:

 *  pygraphviz -- to make connectivity graphs (set verbose = True)
 *  prettyplotlib (https://github.com/olgabot/prettyplotlib)
