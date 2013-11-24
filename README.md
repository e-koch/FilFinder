Description
------------

fil_finder: Module for extraction and analysis of filamentary structure in molecular clouds.

fil_finder segments filamentary structure in an integrated intensity image using adaptive thresholding. The scale of the patch size for
this must be set manually for the time being (see test_script.py). Detected regions are reduced to a skeleton using a Medial Axis
Transform. Pixels within each skeleton are classified by the number of connecting pixels. A pixel can be a body, end, or intersection
point. A shortest path algorithm finds the longest path through the skeleton, which is reported as the main length. At this point,
branches less than a length threshold are removed to give a final skeleton. A Euclidean Distance Transform is performed to build a
radial profile of each filament. A gaussian is fit to the profile to find the width, which is consequently converted to the FWHM value,
and deconvolved with the FWHM beamwidth of the instrument. To describe the shape of the filament, we report its average curvature. This
is done by randomly choosing three pixels on the skeleton and using the Menger Curvature Formula.

Example Images
--------------

![Chameleon-250 Scaled to 2150](https://github.com/e-koch/fil_finder/blob/master/images/chameleon-250-filcontours-2150.png "Chameleon-250 Scaled to 2200")

![Chameleon-250 Scaled to 2500](https://github.com/e-koch/fil_finder/blob/master/images/chameleon-250-filcontours-2500.png "Chameleon-250 Scaled to 2500")

Package Dependencies
--------------------

Requires: numpy 1.7.1,
          matplotlib,
          pyfits (or astropy),
          scipy,
          scikit-image 0.8.0,
          networkx
	  pandas

Optional: pygraphviz -- to make connectivity graphs (set verbose = True)
