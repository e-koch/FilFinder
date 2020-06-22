2.0 (unreleased)
----------------
- #60 : Change longest path algorithm to catch unending loops from `networkx.shortest_simple_paths`. Note that this is not yet a general solution.

1.7 (2020-04-01)
----------------
- #53 : Fix for netwroksx v2.4. Changed `Graph.node` to `Graph.nodes`. The former was removed.
- #52 : Fixes for astropy v4.0. Start dropping py27 tests from travis.
- #44 : Return end and intersection points from final skeletons.
- #42 : Fix edge slicing issue raise in Issue 42 (https://github.com/e-koch/FilFinder/issues/42). Switched all slicing to use astropy.nddata.utils.extract_array, which correctly handles edges cases.


1.6 (2018-03-29)
----------------
- #41 : Major overhaul of the code. Implements the FilFinder2D and Filament2D classes as a replacement for fil_finder_2D. fil_finder_2D is deprecated and will be removed in the next release. See the full list of changes at: https://github.com/e-koch/FilFinder/pull/41
- #40 : Add python 3 support and enable py3 testing on travis. Fixed crucial bug for finding the longest skeleton path. Enable coveralls. Use fourier shifting when calculating the circular confidence intervals. Updated testing data after length finding fix.

1.5 (2017-11-28)
----------------
- #36 : Updates to work with networkx v2.0. Fix getting the angular pixel scale from the WCS object. Fix bug in Gaussian width failure flags.

1.4 (2017-09-12)
----------------
- #34 : Fixed cases where a float was used as an index, which now fails with numpy 1.12. Added testing for numpy 1.12 and astropy 1.3, while removing tests for numpy 1.9.
- #32 : Fixed last point being dropped in `walk_through_skeleton`. Added test for skeleton walk and for end finding.
