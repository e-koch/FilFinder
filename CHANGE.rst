
1.3 (unreleased)
----------------

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