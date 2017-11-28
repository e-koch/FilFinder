1.5 (2017-11-28)
----------------
- #36 : Updates to work with networkx v2.0. Fix getting the angular pixel scale from the WCS object. Fix bug in Gaussian width failure flags.

1.4 (2017-09-12)
----------------
- #34 : Fixed cases where a float was used as an index, which now fails with numpy 1.12. Added testing for numpy 1.12 and astropy 1.3, while removing tests for numpy 1.9.
- #32 : Fixed last point being dropped in `walk_through_skeleton`. Added test for skeleton walk and for end finding.
