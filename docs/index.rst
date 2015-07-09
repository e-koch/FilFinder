.. fil_finder documentation master file, created by
   sphinx-quickstart on Sat Jan 18 16:16:54 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

FilFinder
=========

FilFinder is a module for extraction and analysis of filamentary structure in molecular clouds. In particular, the algorithm is capable of uniformly extracting structure over a large dynamical range in intensity (see images below).

FilFinder segments filamentary structure in an integrated intensity image using adaptive thresholding.
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

Contents:

.. toctree::
   :maxdepth: 3

   install.rst
   tutorial.rst
   fil_finder_2d.rst
   length.rst
   pixel_ident.rst
   width.rst
   curve_direc.rst
   analysis.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

