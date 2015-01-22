
fil\_finder Tutorial
====================

Load in the algorithm and the usual suspects.

.. code:: python

    from astropy.io.fits import getdata
    from fil_finder import fil_finder_2D
    import matplotlib.pylab as pylab
    pylab.rcParams['figure.figsize'] = 128, 96
    import matplotlib.pyplot as p
    %matplotlib inline
Load in the FITS file containing the simulated image.

.. code:: python

    img, hdr = getdata("filaments_updatedhdr.fits", header=True)
Next we initialize the fil\_finder\_2D object.

The algorithm requires a few inputs to begin (other than the image and
header): \* beamsize in arcseconds (set to 15.1 arcsec, though this is a
simulated image, a none zero value is needed as it sets the minimum size
a filament can be). \* skeleton threshold - minimum pixels a skeleton
must contain to be considered (= 30 pixels) \* branch threshold -
minimum length for a branch. This sets one of the skeleton pruning
criteria. If the intensity along it is significant to the filament, or
if its deletion will change the graph connectivity, it will still be
kept. (= 5 pixels) \* pad size - number of pixels to pad around each
filament. This ensures the adaptive thresholding can reach the edges of
the image. Must be at least 1 pixel. (= 10 pixels, about the size of the
patch used). \* distance - distance to the region in parsecs. This is
used to set the size of the adaptive thresholding patch. The input is
optional. If no distance is provided, results remain in pixel units (=
260 pc, distance set for the simulation). \* global threshold - sets the
percentile of data to ignore. This is intended to remove noisy regions
of the data. (= 20%) \* flattening threshold - sets the normalization to
use in the arctan transform (flattens bright, compact regions). This
parameter is generally set automatically, but we seem to get better
results by setting it to the 95% percentile.

.. code:: python

    fils = fil_finder_2D(img, hdr, 15.1, 30, 5, 10, distance=260, glob_thresh=20, flatten_thresh=95)
The algorithm has several steps, which will be outlined below. Using the
run() function will perform all the steps in one with the algorithm
defaults.

Masking
=======

We begin by creating the mask of the image. All of the parameters are
set by default based on physical parameters. However this simulation
doesn't quite adhere to these and so the effect of manipulating these
parameters is shown in the next few steps.

.. code:: python

    fils.create_mask(verbose=True)



.. image:: images/fil_finder%20Tutorial_9_0.png




.. parsed-literal::

    <fil_finder.filfind_class.fil_finder_2D at 0x10dd8e690>



Here is the default mask. The algorithm has largely picked out the
filamentary structure, but there are two issues. First, the mask is not
able to go to the edges of the image, due to the padding with ``Nans``.
To fix this, we invoke the ``border_masking=False`` input.

.. code:: python

    # Reset the mask
    fils.mask = None
    fils.create_mask(verbose=True, border_masking=False)


.. image:: images/fil_finder%20Tutorial_11_0.png




.. parsed-literal::

    <fil_finder.filfind_class.fil_finder_2D at 0x10dd8e690>



This is better, but some variations within the regions are being
combined together. To try to pick up on the smaller scale variations, we
can try using a smaller patch-size for the adaptive thresholding.
Typically, we attain a good mask using a patch size of

.. math:: 0.2 \textrm{pc}/ \textrm{pixel size}.

This works well for observational data, but the filaments in this small
simulation aren't quite the same. So let us try half of the normal patch
size,

.. code:: python

    fils.mask = None
    fils.create_mask(verbose=True, border_masking=False, adapt_thresh=13.)


.. image:: images/fil_finder%20Tutorial_13_0.png




.. parsed-literal::

    <fil_finder.filfind_class.fil_finder_2D at 0x10dd8e690>



This hasn't made a large difference. In general if the patch size is a
reasonable size based on physical information, the mask obtained will be
largely the same.

There are a couple of other parameters based off of physical priors. One
of these is a smoothing filter, which is generally set to be
:math:`~0.05` pc, so as to smooth the small scale variations leading to
more continuous regions. Let's try half of this size as we did before.
This corresponds to about 3 pixels.

.. code:: python

    fils.mask = None
    fils.create_mask(verbose=True, border_masking=False, adapt_thresh=13., smooth_size=3.0)


.. image:: images/fil_finder%20Tutorial_15_0.png




.. parsed-literal::

    <fil_finder.filfind_class.fil_finder_2D at 0x10dd8e690>



Again, this has not made a large difference which ensures that the
smoothing is only acting on scales smaller than we care about here.

The next parameter to try is to disable the regridding function. The
algorithm has functionality to double the image size for the purposes of
adaptive thresholding. When a small patch size is used for the
thresholding, regions become too skinny and often fragment into small
pieces. To deal with this pixelization issue, we perform the
thresholding on the super-sampled image. This negates the patch size
issue, and we obtain a better mask after regridding to the original
size.

.. code:: python

    fils.mask = None
    fils.create_mask(verbose=True, border_masking=False, adapt_thresh=13., smooth_size=3.0, regrid=False, zero_border=True, size_thresh=300.)


.. image:: images/fil_finder%20Tutorial_17_0.png




.. parsed-literal::

    <fil_finder.filfind_class.fil_finder_2D at 0x10dd8e690>



That's better! Not only are the small scale features better
characterized, but some additional faint regions have also been picked
up.

The regridding is useful only when the regions are becoming fragmented.
As a default, it is enabled when the patch size is less than 40 pixels.
This is value is based on many trials with observational data.

Note that pre-made masks can also be supplied to the algorithm during
initialization without completing this step. As a default, if a mask has
been attached to the object it will assume that that mask has been
prescribed and will skip the mask making process.

Skeletons
=========

The next step in the algorithm is to use a Medial Axis Transform to
return the skeletons of the regions. These skeletons are the actual
objects used to derive the filament properties. We make the assumption
that the skeletons run along the ridge of the filament so that they can
be defined as the centers.

.. code:: python

    fils.medskel(verbose=True)


.. image:: images/fil_finder%20Tutorial_20_0.png




.. parsed-literal::

    <fil_finder.filfind_class.fil_finder_2D at 0x10dd8e690>



Pruning and Lengths
===================

Now begins the analysis of the filaments! This begins with finding the
length. The skeletons are also pruned during this process to remove
short branches which aren't essential. This is preferable over
traditional pruning methods which shorten the entire skeleton.

A whole ton of information is printed out when verbose mode is enabled.
\* The first set show the skeletons segmented into their branches (and
intersections have beem removed). Their connectivity graphs are also
shown. Their placement is unfortunately only useful for small
structures. \* Next, the longest paths through the skeleton are shown.
This is determined by the length of the branch and the median brightness
along it relative to the rest of the structure. These lengths are
classified as the main length of the filament. \* The final set shows
the final, pruned skeletons which are recombined into the skeleton image
to be used for the rest of the analysis.

.. code:: python

    fils.analyze_skeletons(verbose=True)

.. parsed-literal::

    Filament: 1 / 19



.. image:: images/fil_finder%20Tutorial_22_1.png


.. parsed-literal::

    Filament: 2 / 19



.. image:: images/fil_finder%20Tutorial_22_3.png


.. parsed-literal::

    Filament: 3 / 19



.. image:: images/fil_finder%20Tutorial_22_5.png


.. parsed-literal::

    Filament: 4 / 19



.. image:: images/fil_finder%20Tutorial_22_7.png


.. parsed-literal::

    Filament: 5 / 19



.. image:: images/fil_finder%20Tutorial_22_9.png


.. parsed-literal::

    Filament: 6 / 19



.. image:: images/fil_finder%20Tutorial_22_11.png


.. parsed-literal::

    Filament: 7 / 19



.. image:: images/fil_finder%20Tutorial_22_13.png


.. parsed-literal::

    Filament: 8 / 19



.. image:: images/fil_finder%20Tutorial_22_15.png


.. parsed-literal::

    Filament: 9 / 19



.. image:: images/fil_finder%20Tutorial_22_17.png


.. parsed-literal::

    Filament: 10 / 19



.. image:: images/fil_finder%20Tutorial_22_19.png


.. parsed-literal::

    Filament: 11 / 19



.. image:: images/fil_finder%20Tutorial_22_21.png


.. parsed-literal::

    Filament: 12 / 19



.. image:: images/fil_finder%20Tutorial_22_23.png


.. parsed-literal::

    Filament: 13 / 19



.. image:: images/fil_finder%20Tutorial_22_25.png


.. parsed-literal::

    Filament: 14 / 19



.. image:: images/fil_finder%20Tutorial_22_27.png


.. parsed-literal::

    Filament: 15 / 19



.. image:: images/fil_finder%20Tutorial_22_29.png


.. parsed-literal::

    Filament: 16 / 19



.. image:: images/fil_finder%20Tutorial_22_31.png


.. parsed-literal::

    Filament: 17 / 19



.. image:: images/fil_finder%20Tutorial_22_33.png


.. parsed-literal::

    Filament: 18 / 19



.. image:: images/fil_finder%20Tutorial_22_35.png


.. parsed-literal::

    Filament: 19 / 19



.. image:: images/fil_finder%20Tutorial_22_37.png


.. parsed-literal::

    Filament: 1 / 19



.. image:: images/fil_finder%20Tutorial_22_39.png


.. parsed-literal::

    Filament: 2 / 19



.. image:: images/fil_finder%20Tutorial_22_41.png


.. parsed-literal::

    Filament: 3 / 19



.. image:: images/fil_finder%20Tutorial_22_43.png


.. parsed-literal::

    Filament: 4 / 19



.. image:: images/fil_finder%20Tutorial_22_45.png


.. parsed-literal::

    Filament: 5 / 19



.. image:: images/fil_finder%20Tutorial_22_47.png


.. parsed-literal::

    Filament: 6 / 19



.. image:: images/fil_finder%20Tutorial_22_49.png


.. parsed-literal::

    Filament: 7 / 19



.. image:: images/fil_finder%20Tutorial_22_51.png


.. parsed-literal::

    Filament: 8 / 19



.. image:: images/fil_finder%20Tutorial_22_53.png


.. parsed-literal::

    Filament: 9 / 19



.. image:: images/fil_finder%20Tutorial_22_55.png


.. parsed-literal::

    Filament: 10 / 19



.. image:: images/fil_finder%20Tutorial_22_57.png


.. parsed-literal::

    Filament: 11 / 19



.. image:: images/fil_finder%20Tutorial_22_59.png


.. parsed-literal::

    Filament: 12 / 19



.. image:: images/fil_finder%20Tutorial_22_61.png


.. parsed-literal::

    Filament: 13 / 19



.. image:: images/fil_finder%20Tutorial_22_63.png


.. parsed-literal::

    Filament: 14 / 19



.. image:: images/fil_finder%20Tutorial_22_65.png


.. parsed-literal::

    Filament: 15 / 19



.. image:: images/fil_finder%20Tutorial_22_67.png


.. parsed-literal::

    Filament: 16 / 19



.. image:: images/fil_finder%20Tutorial_22_69.png


.. parsed-literal::

    Filament: 17 / 19



.. image:: images/fil_finder%20Tutorial_22_71.png


.. parsed-literal::

    Filament: 18 / 19



.. image:: images/fil_finder%20Tutorial_22_73.png


.. parsed-literal::

    Filament: 19 / 19



.. image:: images/fil_finder%20Tutorial_22_75.png



.. image:: images/fil_finder%20Tutorial_22_76.png



.. image:: images/fil_finder%20Tutorial_22_77.png



.. image:: images/fil_finder%20Tutorial_22_78.png



.. image:: images/fil_finder%20Tutorial_22_79.png



.. image:: images/fil_finder%20Tutorial_22_80.png



.. image:: images/fil_finder%20Tutorial_22_81.png



.. image:: images/fil_finder%20Tutorial_22_82.png



.. image:: images/fil_finder%20Tutorial_22_83.png



.. image:: images/fil_finder%20Tutorial_22_84.png



.. image:: images/fil_finder%20Tutorial_22_85.png



.. image:: images/fil_finder%20Tutorial_22_86.png



.. image:: images/fil_finder%20Tutorial_22_87.png



.. image:: images/fil_finder%20Tutorial_22_88.png



.. image:: images/fil_finder%20Tutorial_22_89.png



.. image:: images/fil_finder%20Tutorial_22_90.png



.. image:: images/fil_finder%20Tutorial_22_91.png



.. image:: images/fil_finder%20Tutorial_22_92.png



.. image:: images/fil_finder%20Tutorial_22_93.png



.. image:: images/fil_finder%20Tutorial_22_94.png




.. parsed-literal::

    <fil_finder.filfind_class.fil_finder_2D at 0x10dd8e690>



Let's plot the final skeletons before moving on:

.. code:: python

    p.imshow(fils.flat_img, interpolation=None, origin='lower')
    p.contour(fils.skeleton, colors='k')



.. parsed-literal::

    <matplotlib.contour.QuadContourSet instance at 0x10dff2290>




.. image:: images/fil_finder%20Tutorial_24_1.png


The original skeletons didn't contain too many spurious features, so
there is relatively little change.

Curvature and Direction
=======================

Following this step, we use a version of the `Rolling Hough
Transform <http://adsabs.harvard.edu/abs/2014ApJ...789...82C>`__ to find
the orientation of the filaments (median of transform) and their
curvature (IQR of transform).

The polar plots shown plot :math:`2\theta`. The transform itself is
limited to :math:`(0, \pi)`. The first plot shows the transform
distribution for that filament. Beside it is the CDF of that
distribution. By default, the transform is applied on the longest path
of the skeleton. It can also be applied on a per-branch basis. This
destroys information of the filaments relative to each other, but gives
a better estimate for the image as a whole.

.. code:: python

    fils.exec_rht(verbose=True)


.. image:: images/fil_finder%20Tutorial_26_0.png



.. image:: images/fil_finder%20Tutorial_26_1.png



.. image:: images/fil_finder%20Tutorial_26_2.png



.. image:: images/fil_finder%20Tutorial_26_3.png



.. image:: images/fil_finder%20Tutorial_26_4.png



.. image:: images/fil_finder%20Tutorial_26_5.png



.. image:: images/fil_finder%20Tutorial_26_6.png



.. image:: images/fil_finder%20Tutorial_26_7.png



.. image:: images/fil_finder%20Tutorial_26_8.png



.. image:: images/fil_finder%20Tutorial_26_9.png



.. image:: images/fil_finder%20Tutorial_26_10.png



.. image:: images/fil_finder%20Tutorial_26_11.png



.. image:: images/fil_finder%20Tutorial_26_12.png



.. image:: images/fil_finder%20Tutorial_26_13.png



.. image:: images/fil_finder%20Tutorial_26_14.png



.. image:: images/fil_finder%20Tutorial_26_15.png



.. image:: images/fil_finder%20Tutorial_26_16.png



.. image:: images/fil_finder%20Tutorial_26_17.png



.. image:: images/fil_finder%20Tutorial_26_18.png




.. parsed-literal::

    <fil_finder.filfind_class.fil_finder_2D at 0x10dd8e690>



Widths
======

One of the final steps is to find the widths of the filaments.
``fil_finder`` supports three different models to fit to the radial
profiles. By default, a Gaussian with a background and mean zero is
used. Using the ``fit_model`` parameter, a Lorentzian model or radial
cylindrical model can also be specified (imported from
``fil_finder.widths``). With observational data, we found that many
profiles are not well fit by these idealized cases. So there is also a
non-parameteric method we have developed which simply estimates a peak
and background and interpolates between them to estimate the width. This
is enabled, by default, using ``try_nonparam``. If a fit returns a lousy
:math:`\chi^2` value, we attempt to use the non-parameteric method.

Fits are rejected based on a set of criteria: \* Background is above the
peak \* Errors are larger than the respective parameters \* The width is
too small to be deconvolved from the beamwidth \* The width is not
appreciably smaller than the length \* The non-parametric method cannot
find a reasonable estimate

*Note:* Each profile is plotted before invoking the rejection criteria.
This is why some of the plots below look particularly suspect. Also, the
fitted lines are based on the model given (gaussian for this case) and
since the non-parameteric method is not quite this profile, the fits
appear to be overestimated.

.. code:: python

    fils.find_widths(verbose=True)

.. parsed-literal::

    0 in 19
    Fit Parameters: [ 0.07826921  0.08033422 -0.00112114  0.18222796]
    Fit Errors: [ 0.00691331  0.04211509  0.03820988  0.04363059]
    Fit Type: gaussian


.. parsed-literal::

    /Users/eric/anaconda/lib/python2.7/site-packages/numpy/core/_methods.py:59: RuntimeWarning: Mean of empty slice.
      warnings.warn("Mean of empty slice.", RuntimeWarning)
    /Users/eric/anaconda/lib/python2.7/site-packages/numpy/core/_methods.py:71: RuntimeWarning: invalid value encountered in true_divide
      ret = ret.dtype.type(ret / rcount)



.. image:: images/fil_finder%20Tutorial_28_2.png


.. parsed-literal::

    1 in 19
    Fit Parameters: [ 0.02902631  0.06588213  0.01916882  0.14659243]
    Fit Errors: [ 0.00014506  0.01061423  0.00149817  0.01121018]
    Fit Type: gaussian



.. image:: images/fil_finder%20Tutorial_28_4.png


.. parsed-literal::

    2 in 19
    Fit Parameters: [ 1.20522334  0.0219877   0.0189823   0.01008039]
    Fit Errors: [ 0.00984049  0.00051245  0.00057427  0.00262677]
    Fit Type: gaussian



.. image:: images/fil_finder%20Tutorial_28_6.png


.. parsed-literal::

    3 in 19
    Fit Parameters: [ 2.3075243   0.14985602  0.13712579  0.3492103 ]
    Fit Errors: [ 1.55684352  0.15573363  0.36581519  0.13365941]
    Fit Type: nonparam



.. image:: images/fil_finder%20Tutorial_28_8.png


.. parsed-literal::

    4 in 19
    Fit Parameters: [ 0.83000271  0.01555012  0.02147785  0.        ]
    Fit Errors: [ 0.02972446  0.00200598  0.00662622  0.        ]
    Fit Type: gaussian


.. parsed-literal::

    /Users/eric/anaconda/lib/python2.7/site-packages/fil_finder-1.0-py2.7.egg/fil_finder/filfind_class.py:896: RuntimeWarning: invalid value encountered in less_equal



.. image:: images/fil_finder%20Tutorial_28_11.png


.. parsed-literal::

    5 in 19
    Fit Parameters: [ 0.4307426   0.02101916  0.00674746  0.        ]
    Fit Errors: [ 0.00333481  0.00061792  0.0014207   0.        ]
    Fit Type: gaussian



.. image:: images/fil_finder%20Tutorial_28_13.png


.. parsed-literal::

    6 in 19
    Fit Parameters: [ 1.10187851  0.04946428  0.03033507  0.10482469]
    Fit Errors: [ 0.03499969  0.00359438  0.0162534   0.00398585]
    Fit Type: gaussian



.. image:: images/fil_finder%20Tutorial_28_15.png


.. parsed-literal::

    7 in 19
    Fit Parameters: [ 0.19037385  0.09057394  0.02852461  0.20715061]
    Fit Errors: [ 0.04603811  0.09476529  0.07063716  0.0828698 ]
    Fit Type: nonparam



.. image:: images/fil_finder%20Tutorial_28_17.png


.. parsed-literal::

    8 in 19
    Fit Parameters: [ 0.77565624  0.02372127  0.0387381   0.02325965]
    Fit Errors: [ 0.0045763   0.00061823  0.00323371  0.00148167]
    Fit Type: gaussian



.. image:: images/fil_finder%20Tutorial_28_19.png


.. parsed-literal::

    9 in 19
    Fit Parameters: [  1.54644855e-01   5.82543676e+00  -5.40882024e+02   1.37177612e+01]
    Fit Errors: [  8.82951623e-03   5.13787488e+03   9.54312772e+05   5.12739338e+03]
    Fit Type: gaussian



.. image:: images/fil_finder%20Tutorial_28_21.png


.. parsed-literal::

    10 in 19
    Fit Parameters: [ 0.06034145  0.03863372  0.03557391  0.07548035]
    Fit Errors: [ 0.00033573  0.00265216  0.00025108  0.00319008]
    Fit Type: gaussian



.. image:: images/fil_finder%20Tutorial_28_23.png


.. parsed-literal::

    11 in 19
    Fit Parameters: [ 0.27323391  0.12239918  0.04373292  0.28371845]
    Fit Errors: [ 0.03727465  0.12587496  0.09212256  0.10860762]
    Fit Type: nonparam



.. image:: images/fil_finder%20Tutorial_28_25.png


.. parsed-literal::

    12 in 19
    Fit Parameters: [  1.73530290e-01   6.73499469e+00  -7.87543652e+02   1.58596192e+01]
    Fit Errors: [  6.98964005e-03   5.85109038e+03   1.36859049e+06   5.83914381e+03]
    Fit Type: gaussian



.. image:: images/fil_finder%20Tutorial_28_27.png


.. parsed-literal::

    13 in 19
    Fit Parameters: [ 1.73875602  0.01200995  0.03349105  0.        ]
    Fit Errors: [ 0.00662544  0.00019407  0.00544095  0.        ]
    Fit Type: gaussian



.. image:: images/fil_finder%20Tutorial_28_29.png


.. parsed-literal::

    14 in 19
    Fit Parameters: [ 0.35794214  0.0486717   0.02433705  0.10274682]
    Fit Errors: [ 0.00234102  0.00196255  0.00246432  0.00218472]
    Fit Type: gaussian



.. image:: images/fil_finder%20Tutorial_28_31.png


.. parsed-literal::

    15 in 19
    Fit Parameters: [ 2.02660581  0.01160084  0.45819778  0.        ]
    Fit Errors: [ 0.06886086  0.00224936  0.01280631  0.        ]
    Fit Type: gaussian



.. image:: images/fil_finder%20Tutorial_28_33.png


.. parsed-literal::

    16 in 19
    Fit Parameters: [ 0.50019826  0.12083968  0.02364759  0.27998697]
    Fit Errors: [ 0.07848576  0.05176816  0.21708085  0.04468528]
    Fit Type: nonparam



.. image:: images/fil_finder%20Tutorial_28_35.png


.. parsed-literal::

    17 in 19
    Fit Parameters: [ 0.31453035  0.01606313  0.09530031  0.        ]
    Fit Errors: [ 0.00364746  0.0010019   0.0009622   0.        ]
    Fit Type: gaussian



.. image:: images/fil_finder%20Tutorial_28_37.png


.. parsed-literal::

    18 in 19
    Fit Parameters: [ 2.23622518  0.02492762  0.1032614   0.02943544]
    Fit Errors: [ 0.031038    0.00147043  0.01080677  0.00292633]
    Fit Type: gaussian



.. image:: images/fil_finder%20Tutorial_28_39.png




.. parsed-literal::

    <fil_finder.filfind_class.fil_finder_2D at 0x10dd8e690>



Further Methods and Properties
==============================

While the above represent the major filamentary properties, some others
can also be computed.

As part of the width finding function, the sum of the intensity within
the filament's width is found. It requires information from the radial
profiles, which are not returned, and is therefore lopped into that
process. They can be accessed by ``fils.total_intensity``.

The median intensity of each filament can also be found using the
function ``fils.compute_filament_brightness``. This estimate is along
the ridge of the filament, unlike ``fils.total_intensity`` which is
within the fitted width.

Finally, we can model the filamentary network found in the image using
``fils.filament_model``. Using the fitted profile information, filaments
whose fits did not fail can be estimated. For this image, the model is
shown below.

.. code:: python

    p.imshow(fils.filament_model(), interpolation=None, origin='lower')



.. parsed-literal::

    <matplotlib.image.AxesImage at 0x110d5a4d0>




.. image:: images/fil_finder%20Tutorial_30_1.png


Though not a perfect representation, it gives an esimate of the network
and the relation of the intensity in the network versus the entire
image. This fraction is computed by the function
``fils.find_covering_fraction``:

.. code:: python

    fils.find_covering_fraction()
    print fils.covering_fraction

.. parsed-literal::

    0.529317467425


Approximately 52% of the total intensity in the image is coming from the
filamentary network. This seems reasonable, as the algorithm inherently
ignores compact features, whose intensities generally greatly exceed
that of the filaments.

Saving Outputs
==============

Saving of outputs created by the algorithm are split into 2 functions.

Numerical data is dealt with using ``fils.save_table``. This combines
the results derived for each of the portions into a final table. We use
the `astropy.table <http://astropy.readthedocs.org/en/latest/table/>`__
package to save the results. Currently, the type of output is specified
through ``table_type`` and accepts 'csv', 'fits', and 'latex' as valid
output types. If the output is saved as a fits file, branch information
is not saved as BIN tables do not accept lists as an entry. The data
table created can be accessed after through ``fils.dataframe``, which is
accepted by the ``Analysis`` object.

Image products are saved using ``fils.save_fits``. By default, the mask,
skeleton, and model images are all saved. Saving of the model can be
disabled through ``model_save=False``. The output skeleton FITS file has
one extension of the final, cleaned skeletons, and a second containing
only the longest path skeletons. Optionally, stamp images of each
individual filament can be created. These contain a portion of the
image, the final skeleton, and the longest path in the outputted FITS
file. The files are automatically saved in a 'stamps\_(save\_name)'
folder.

