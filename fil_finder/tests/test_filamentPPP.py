
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

    assert (loaded_fil.length() == fil.length()).all()
    assert (loaded_fil.skeleton(out_type='longpath', pad_size=pad) == mask_expect).all()

    import os
    os.remove('pickled_fil.pkl')
