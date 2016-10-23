
1.3 (2016-10-22)
----------------
- [#30] - Fixes minor branch deletion when using branches with RHT, +1 index for slicing out individual skeletons, re-worked `radial_profile` to clarify usage of `pad_to_distance` and `max_distance`, tests for `radial_profile`, and re-wrote `_smooth_and_cut` to smooth with a moving average instead of a gaussian kernel. The unit test values were updated to reflect minor changes in the treatment of the radial profiles.
- [#27] - Use WCS objects, instead of the header, to get the pixel scales.
- [#24] - Allow for `fits.PrimaryHDUs` to be given. Headers are now optional; units default to pixel units in this case. `beamwidth` and `distance` are expected to be `astropy.units.Quantity` objects with appropriate units!
- [#23] - Fix error in length when using new version of skimage.
- [#22] - Fix FWHM error calculation.

- [#21] - Enforce odd patch sizes for adaptive thresholding. New unit test data to reflect the slight change from using an odd patch size in the test cases.

- Added preliminary functionality to return radial filament profiles for each
position in the longest path skeleton.

- [#19] - Generalize radial profile cutting.

- [#17] - Allow longest path skeletons to be used for width fitting. Also fixed a fitting bug where bad guesses at the initial width led to terrible fits. Fixed a bug in the non-parametric width where the target width was not using e^-0.5 of the maximum.

- [#16] - Remove graphviz drawing of graphs


1.2.1 (2015-10-30): Patch for Numpy 1.10
----------------------------------------

- [#9] - Fix figure clearing in terminal

- [#10] - Separate skeleton and image padding. Default to no image padding.
          Clean-up with "no_mask" properties from [#9]. Removed
          "return_distance" option for the medial axis transform; it is always
          used.
- [#11] - Re-wrote radial profiling to work with the changes in numpy 1.10