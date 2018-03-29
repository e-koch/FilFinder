.. fil_finder documentation master file, created by
   sphinx-quickstart on Sat Jan 18 16:16:54 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

FilFinder
=========

FilFinder is a Python package for extraction and analysis of filamentary structure in molecular clouds. In particular, the algorithm is capable of uniformly extracting structure over a large dynamical range in intensity (see images below).

If you make use of FilFinder in a publication, please cite our accompanying paper::

    @ARTICLE{2015MNRAS.452.3435K,
       author = {{Koch}, E.~W. and {Rosolowsky}, E.~W.},
        title = "{Filament identification through mathematical morphology}",
      journal = {\mnras},
    archivePrefix = "arXiv",
       eprint = {1507.02289},
     keywords = {techniques: image processing, stars: formation, ISM: structure, submillimetre: ISM},
         year = 2015,
        month = oct,
       volume = 452,
        pages = {3435-3450},
          doi = {10.1093/mnras/stv1521},
       adsurl = {http://adsabs.harvard.edu/abs/2015MNRAS.452.3435K},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

Citation courtesy of `ADS <https://ui.adsabs.harvard.edu/#>`_

Please join the `FilFinder mailing list <https://groups.google.com/forum/#!forum/filfinder>`_ to receive alerts on new package releases.

FilFinder segments filamentary structure by using `adaptive thresholding <http://scikit-image.org/docs/dev/auto_examples/plot_threshold_adaptive.html>`_. This performs thresholding over local neighborhoods, allowing for the extraction of structure over a large dynamic range. Using the filament mask, the length, width, orientation and curvature are calculated. Further features include extracting radial profiles along the longest skeleton path, creating a filament-only model images, and extracting values along the skeleton.


Contributing & Reporting Issues
-------------------------------

We welcome all user feedback on FilFinder's performance. If you find an issue with the code, or would like to request additional features, please raise an issue in the repository, post a question to the `google group <https://groups.google.com/forum/#!forum/filfinder>`_ or send me an email at the address on `this page <https://github.com/e-koch>`_.

Contributions to the package are welcomed! We follow the `astropy coding guidelines <http://docs.astropy.org/en/stable/development/codeguide.html>`_ and contributions should follow these conventions.

Contents:

.. toctree::
   :maxdepth: 3

   install
   tutorial
   Filament2D_tutorial
   old_tutorial
   filfinder


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

