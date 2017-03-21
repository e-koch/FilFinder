
'''
Tests for functions in fil_finder.width_profiles.
'''

import pytest

import numpy as np
import numpy.testing as npt


from ..width_profiles.profile_line_width import walk_through_skeleton, return_ends


def make_test_skeleton(shape=(10, 10)):
    skel = np.zeros(shape)
    crds = np.array([[3, 3, 4], [4, 5, 6]])
    crds = crds.T
    walk_idx = (crds[:, 0], crds[:, 1])
    skel[walk_idx] = 1

    return skel, crds


def test_walk_through_skeleton():

    skel, crds = make_test_skeleton()
    out_crds = walk_through_skeleton(skel)

    assert out_crds == zip(crds[:, 0], crds[:, 1])


def test_return_ends():

    skel, crds = make_test_skeleton()

    ends = return_ends(skel)

    # Check both the first and last point, since the direction isn't
    # really important.
    if (crds[0] == ends[0]).all() or (crds[0] == ends[-1]).all():
        first_end = True
    else:
        first_end = False

    if (crds[-1] == ends[0]).all() or (crds[-1] == ends[-1]).all():
        second_end = True
    else:
        second_end = False

    if not first_end and not second_end:
        raise Exception("At least one end point was not found.")

