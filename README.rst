FilFinder
=========

**NOTE: FilFinder v1.6 introduces API changes and several critical bug fixes, and v1.7 fixes an error when using networkx >v2. Please update to use v1.7!**

To be notified of future package releases and updates to FilFinder, please join the `mailing list <https://groups.google.com/forum/#!forum/filfinder>`__.

If you use FilFinder in a publication, please cite `Koch & Rosolowsky (2015) <https://ui.adsabs.harvard.edu/abs/2015MNRAS.452.3435K/abstract>`__.


Build and coverage status
=========================

|Coverage Status| |DOI|

.. |Coverage Status| image:: https://codecov.io/gh/e-koch/FilFinder/branch/master/graph/badge.svg?token=MxoAaTTsjc
   :target: https://codecov.io/gh/e-koch/FilFinder
.. |DOI| image:: https://zenodo.org/badge/9172/e-koch/FilFinder.svg
   :target: http://dx.doi.org/10.5281/zenodo.18463

Brief Description
-----------------

FilFinder is a Python package for extraction and analysis of filamentary structure in molecular clouds. In particular, the algorithm is capable of uniformly extracting structure over a large dynamical range in intensity (see images below). FilFinder supports python 2 and 3.

The algorithm proceeds through multiple steps:

* FilFinder segments filamentary structure by using `adaptive thresholding <http://scikit-image.org/docs/dev/auto_examples/plot_threshold_adaptive.html>`__. This performs thresholding over local neighborhoods, allowing for the extraction of structure over a large dynamic range.
* The final filament mask is constructed by applying morphological operators to remove extraneous small regions. The order of these operations are:

    * (Optionally) Flatten using an arctan transform - this removes the effects of small bright features (ie. cores) from effecting the filament mask.
    * Smooth with a small median filter (half the size of the expected filament widths) - this decreases fragmentation of regions in the final mask
    * Apply the adaptive threshold - the patch size is set to the expected filament width (0.1 pc) by default - this sets the scale of the objects to be detected. Within a factor of a few, this size does not effect the result greatly. The widths of the masked regions are not used for deriving any physical properties.
    * Objects below a set area threshold are removed to give the final mask - For the HGBS data, we found a good threshold was 5 * (0.1 pc)^2 but this may change depending on the data the algorithm is used on.

* The final regions are reduced to skeletons via a `Medial Axis Transform <http://scikit-image.org/docs/dev/auto_examples/plot_medial_transform.html>`__ for further analysis.
* Pixels within each skeleton are classified by the number of connecting pixels. A pixel can be a body point, end point, or intersection point. The skeletons are broken up into a set of branches to determine the length.
* The length is determined by converting the set of branches into a graph. Nodes on the graph are intersections and end points. The branches make up the connections and their weighting in the graph is determined by their length and average intensity. A shortest path algorithm determines the longest path through the skeleton, which is reported as the main length.
* The skeletons are then pruned by removing branches that are: not in the main length, will not affect the connectivity of the entire graph if they are removed, and whose length and average intensity are below a set threshold.
* The width of the filament is determined by building a radial profile using the distance from the skeleton. This is accomplished by using a Euclidean Distance Transform and binning the intensity values of the pixels based on their minimum distance from a skeleton pixel. By default, a Gaussian with a constant background is fit to the profile. The reported filament width is the FWHM after deconvolving with the the FWHM of the beam.
* A measure of filament direction and curvature is found using the `Rolling Hough Transform <http://adsabs.harvard.edu/abs/2014ApJ...789...82C>`__. This method returns a distribution of angles, from which the mean and variance are  defined using circular statistics.

These are the basic steps of the algorithm, which will return the main filament properties: local amplitude and background, width, length, orientation and curvature. Additional tools are available, such as creating a filament-only image based on the properties of the radial fits.

The resulting mask and skeletons may be saved in FITS format. Property tables may be saved as a csv, fits or latex table. See the ```fil_finder_2D``` documentation for more details.


Contributing
------------

We welcome any user feedback on FilFinder's performance. If you find an issue with the code, or would like to request additional features, please raise an issue in the repository or send me an email at the address on `this page <https://github.com/e-koch>`__.

We also welcome and encourage contributions to the code base! We want this package to evolve into a tool developed for the community, by the community.
