
'''
Tests for FilFinderPPP and FilamentPPP
'''

import pytest

import numpy as np
import astropy.units as u
import warnings

from ..filament import FilamentPPP, FilamentNDBase


def test_FilamentPPP():

    pixels = (np.array([0, 1, 2]), np.array([0, 0, 0]), np.array([0, 1, 2]))

    image = np.zeros((3, 1, 3))
    image[0, :] = 2.

    fil = FilamentPPP(pixels)

    assert all([(out == inp).all() for out, inp in
                zip(fil.pixel_coords, pixels)])
    assert fil.pixel_extents == [(0, 0, 0), (2, 0, 2)]

    # assert fil.position() == [0 * u.pix, 1 * u.pix]

    # # Should return pixels again because no WCS info is given
    # with warnings.catch_warnings(record=True) as w:
    #     assert fil.position(world_coord=True) == [0 * u.pix, 1 * u.pix]

    # assert len(w) == 1
    # assert w[0].category == UserWarning
    # assert str(w[0].message) == ("No WCS information given. Returning pixel"
    #                              " position.")

    mask_expect = np.zeros((3, 1, 3), dtype=bool)
    mask_expect[pixels] = True

    mask = fil.skeleton()

    assert (mask == mask_expect).all()

    # Test out the padding
    pad = 1
    mask_expect = np.zeros((3 + 2 * pad, 1 + 2 * pad, 3 + 2 * pad), dtype=bool)
    mask_expect[pixels[0] + pad, pixels[1] + pad, pixels[2] + pad] = True

    mask = fil.skeleton(pad_size=pad)

    assert (mask == mask_expect).all()

    # Long path mask should fail without skeleton_analysis
    with pytest.raises(AttributeError):
        fil.skeleton(out_type='longpath')

    # Now run the skeleton analysis
    # Only one branch. Nothing should be removed.
    fil.skeleton_analysis(image)

    mask = fil.skeleton(out_type='longpath', pad_size=pad)
    assert (mask == mask_expect).all()

    # Check the length
    assert fil.length().value == 2 * np.sqrt(2)
    assert fil.length().unit == u.pix

    # Check the intersection and end points
    assert len(fil.intersec_pts) == 0
    assert fil.intersec_pts == []
    assert len(fil.end_pts) == 2
    assert fil.end_pts[0] == (0, 0, 0)
    assert fil.end_pts[1] == (2, 0, 2)

    # Angular and physical conversion should fail b/c no WCS or distance is
    # given
    with pytest.raises(AttributeError):
        fil.length(unit=u.deg)

    with pytest.raises(AttributeError):
        fil.length(unit=u.pc)

    # Test pickling
    fil.to_pickle("pickled_fil.pkl")

    loaded_fil = FilamentPPP.from_pickle("pickled_fil.pkl")

    # Compare a few properties
    assert hasattr(loaded_fil, "_skan_skeleton")
    assert hasattr(loaded_fil, "_graph")
    assert (loaded_fil.length() == fil.length()).all()
    assert (loaded_fil.skeleton(out_type='longpath', pad_size=pad) == mask_expect).all()

    import os
    os.remove('pickled_fil.pkl')


def test_FilamentPPP_onebranch():

    pixels = (np.array([0, 1, 2, 3, 2, 2, 2]),
              np.array([0, 0, 0, 0, 1, 2, 3]),
              np.array([0, 1, 2, 3, 1, 1, 1]))

    image = np.zeros((4, 4, 4))
    image[0, :] = 2.

    fil = FilamentPPP(pixels)

    assert all([(out == inp).all() for out, inp in
                zip(fil.pixel_coords, pixels)])
    assert fil.pixel_extents == [(0, 0, 0), (3, 3, 3)]

    # assert fil.position() == [0 * u.pix, 1 * u.pix]

    # # Should return pixels again because no WCS info is given
    # with warnings.catch_warnings(record=True) as w:
    #     assert fil.position(world_coord=True) == [0 * u.pix, 1 * u.pix]

    # assert len(w) == 1
    # assert w[0].category == UserWarning
    # assert str(w[0].message) == ("No WCS information given. Returning pixel"
    #                              " position.")

    mask_expect = np.zeros((4, 4, 4), dtype=bool)
    mask_expect[pixels] = True

    mask = fil.skeleton()

    assert (mask == mask_expect).all()

    # Test out the padding
    pad = 1
    mask_expect = np.zeros((4 + 2 * pad, 4 + 2 * pad, 4 + 2 * pad), dtype=bool)
    mask_expect[pixels[0] + pad, pixels[1] + pad, pixels[2] + pad] = True

    mask = fil.skeleton(pad_size=pad)

    assert (mask == mask_expect).all()

    # Long path mask should fail without skeleton_analysis
    with pytest.raises(AttributeError):
        fil.skeleton(out_type='longpath')

    # Now run the skeleton analysis
    # No pruning for 0 branch threshold
    fil.skeleton_analysis(image, branch_thresh=0 * u.pix)
    # fil.skeleton_analysis(image, branch_thresh=5 * u.pix)

    mask = fil.skeleton(out_type='longpath', pad_size=0)

    mask_expect = np.zeros((4, 4, 4), dtype=bool)
    mask_expect[pixels[0][np.r_[0, 1, 2, 4, 5, 6]],
                pixels[1][np.r_[0, 1, 2, 4, 5, 6]],
                pixels[2][np.r_[0, 1, 2, 4, 5, 6]]] = True

    assert (mask == mask_expect).all()

    # Check the length
    np.testing.assert_equal(fil.length().value, 3 * np.sqrt(2) + 2)
    assert fil.length().unit == u.pix

    # Check the intersection and end points
    # "intersections" due to 26-connected criteria and not centroiding
    # together to keep pixel mapping
    assert len(fil.intersec_pts) == 3
    assert fil.intersec_pts == [(1, 0, 1), (2, 0, 2), (2, 1, 1)]
    assert len(fil.end_pts) == 3
    assert fil.end_pts[0] == (0, 0, 0)
    assert fil.end_pts[1] == (2, 3, 1)
    assert fil.end_pts[2] == (3, 0, 3)

    # Angular and physical conversion should fail b/c no WCS or distance is
    # given
    with pytest.raises(AttributeError):
        fil.length(unit=u.deg)

    with pytest.raises(AttributeError):
        fil.length(unit=u.pc)

    # Now set the branch threshold to remove one branch.
    # The longest path should be unchanged
    fil.skeleton_analysis(image, branch_thresh=5 * u.pix)

    mask = fil.skeleton(out_type='longpath', pad_size=0)

    # Should have removed (3, 0, 3)
    assert not mask[3, 0, 3]

    assert np.where(mask)[0].size == 6

    mask_expect = np.zeros((4, 4, 4), dtype=bool)
    mask_expect[pixels[0][np.r_[0, 1, 2, 4, 5, 6]],
                pixels[1][np.r_[0, 1, 2, 4, 5, 6]],
                pixels[2][np.r_[0, 1, 2, 4, 5, 6]]] = True

    assert (mask == mask_expect).all()

    # Check the length
    np.testing.assert_equal(fil.length().value, 3 * np.sqrt(2) + 2)
    assert fil.length().unit == u.pix

    # Check the intersection and end points
    # "intersections" due to 26-connected criteria and not centroiding
    # together to keep pixel mapping
    assert len(fil.intersec_pts) == 2
    assert fil.intersec_pts == [(1, 0, 1), (2, 1, 1)]
    assert len(fil.end_pts) == 2
    assert fil.end_pts[0] == (0, 0, 0)
    assert fil.end_pts[1] == (2, 3, 1)
