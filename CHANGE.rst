
1.4 (unreleased)
----------------
- #34 : Fixed cases where a float was used as an index, which now fails with numpy 1.12. Added testing for numpy 1.12 and astropy 1.3, while removing tests for numpy 1.9.
- #32 : Fixed last point being dropped in `walk_through_skeleton`. Added test for skeleton walk and for end finding.
