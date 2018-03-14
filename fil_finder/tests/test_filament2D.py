
import pytest

import numpy as np
import numpy.testing as npt
import astropy.units as u
from astropy.wcs import WCS

from ..filament import Filament2D, FilamentNDBase


def test_Filament2D():

    pixels = (np.array([0, 0, 0]), np.array([0, 1, 2]))

    image = np.zeros((1, 3))
    image[0, :] = 2.

    fil = Filament2D(pixels)

    assert all([(out == inp).all() for out, inp in
                zip(fil.pixel_coords, pixels)])
    assert fil.pixel_extents == [(0, 0), (0, 2)]

    mask_expect = np.zeros((1, 3), dtype=bool)
    mask_expect[pixels] = True

    mask = fil.skeleton()

    assert (mask == mask_expect).all()

    # Test out the padding
    pad = 1
    mask_expect = np.zeros((1 + 2 * pad, 3 + 2 * pad), dtype=bool)
    mask_expect[pixels[0] + pad, pixels[1] + pad] = True

    mask = fil.skeleton(pad_size=pad)

    assert (mask == mask_expect).all()

    # Long path mask should fail without skeleton_analysis
    with pytest.raises(AttributeError):
        fil.skeleton(out_type='longpath')

    # Now run the skeleton analysis
    fil.skeleton_analysis(image)

    mask = fil.skeleton(out_type='longpath', pad_size=pad)
    assert (mask == mask_expect).all()

    # Check the length
    assert fil.length().value == 2.0
    assert fil.length().unit == u.pix

    # Angular and physical conversion should fail b/c no WCS or distance is
    # given
    with pytest.raises(AttributeError):
        fil.length(unit=u.deg)

    with pytest.raises(AttributeError):
        fil.length(unit=u.pc)

    # Test pickling
    fil.to_pickle("pickled_fil.pkl")

    loaded_fil = Filament2D.from_pickle("pickled_fil.pkl")

    # Compare a few properties

    assert (loaded_fil.length() == fil.length()).all()
    assert (loaded_fil.skeleton(out_type='longpath', pad_size=pad) ==
            mask_expect).all()

    import os
    os.remove('pickled_fil.pkl')


def test_Filament2D_with_WCS():

    pixels = (np.array([0, 0, 0]), np.array([0, 1, 2]))

    image = np.zeros((1, 3))
    image[0, :] = 2.

    mywcs = WCS()
    mywcs.wcs.ctype = ['GLON-CAR', 'GLAT-CAR']
    mywcs.wcs.cunit = ['deg', 'deg']

    fil = Filament2D(pixels, wcs=mywcs)
    fil.skeleton_analysis(image)

    # Check the length
    assert fil.length().value == 2.0
    assert fil.length().unit == u.pix

    assert fil.length(unit=u.deg) == 2.0 * u.deg

    # Physical conversion still fails
    with pytest.raises(AttributeError):
        fil.length(unit=u.pc)


def test_Filament2D_with_distance():

    pixels = (np.array([0, 0, 0]), np.array([0, 1, 2]))

    image = np.zeros((1, 3))
    image[0, :] = 2.

    mywcs = WCS()
    mywcs.wcs.ctype = ['GLON-CAR', 'GLAT-CAR']
    mywcs.wcs.cunit = ['deg', 'deg']

    fil = Filament2D(pixels, wcs=mywcs, distance=100 * u.pc)
    fil.skeleton_analysis(image)

    # Check the length
    assert fil.length().value == 2.0
    assert fil.length().unit == u.pix

    assert fil.length(unit=u.deg) == 2.0 * u.deg

    assert fil.length(unit=u.pc) == (2.0 * u.deg).to(u.rad).value * 100 * u.pc


def test_Filament2D_onebranch():
    '''
    Longest path will be in a straight line. Expect length of 8
    '''

    pixels = (np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2]),
              np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 3, 3]))

    shape = (3, 9)

    image = np.zeros(shape)
    image[pixels] = 2.

    fil = Filament2D(pixels)

    assert all([(out == inp).all() for out, inp in
                zip(fil.pixel_coords, pixels)])
    assert fil.pixel_extents == [(0, 0), (shape[0] - 1, shape[1] - 1)]

    mask_expect = np.zeros(shape, dtype=bool)
    mask_expect[pixels] = True

    mask = fil.skeleton()

    assert (mask == mask_expect).all()

    # Test out the padding
    pad = 1
    mask_expect = np.zeros((shape[0] + 2 * pad,
                            shape[1] + 2 * pad), dtype=bool)
    mask_expect[pixels[0] + pad, pixels[1] + pad] = True

    mask = fil.skeleton(pad_size=pad)

    assert (mask == mask_expect).all()

    # Long path mask should fail without skeleton_analysis
    with pytest.raises(AttributeError):
        fil.skeleton(out_type='longpath')

    # Now run the skeleton analysis
    fil.skeleton_analysis(image)

    # Remove the one branch pixel
    mask_expect = np.zeros((shape[0] + 2 * pad, shape[1] + 2 * pad),
                           dtype=bool)
    mask_expect[pixels[0][:-2] + pad, pixels[1][:-2] + pad] = True

    mask = fil.skeleton(out_type='longpath', pad_size=pad)
    assert (mask == mask_expect).all()

    # Check the length
    assert fil.length().value == 8.0
    assert fil.length().unit == u.pix

    # Angular and physical conversion should fail b/c no WCS or distance is
    # given
    with pytest.raises(AttributeError):
        fil.length(unit=u.deg)

    with pytest.raises(AttributeError):
        fil.length(unit=u.pc)

    # The branches should all have the same average intensity
    for branch_int in fil.branch_properties['intensity']:
        assert branch_int == 2.0

    # Finally, make sure the transpose shape gives the same length, masks
    fil = Filament2D((pixels[1], pixels[0]))

    assert all([(out == inp).all() for out, inp in
                zip(fil.pixel_coords, pixels[::-1])])
    assert fil.pixel_extents == [(0, 0), (shape[1] - 1, shape[0] - 1)]

    mask_expect = np.zeros(shape, dtype=bool)
    mask_expect[pixels] = True

    mask = fil.skeleton()

    assert (mask == mask_expect.T).all()

    fil.skeleton_analysis(image.T)

    assert fil.length().value == 8.0

    mask_expect = np.zeros((shape[0] + 2 * pad, shape[1] + 2 * pad),
                           dtype=bool)
    mask_expect[pixels[0][:-2] + pad, pixels[1][:-2] + pad] = True

    mask = fil.skeleton(out_type='longpath', pad_size=pad)
    assert (mask == mask_expect.T).all()


def test_Filament2D_onebranch_wpadding():
    '''
    Longest path will be in a straight line. Expect length of 8
    '''

    pixels = (np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2]) + 1,
              np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 3, 3]) + 1)

    shape = (5, 11)

    image = np.zeros(shape)
    image[pixels] = 2.

    fil = Filament2D(pixels)

    assert all([(out == inp).all() for out, inp in
                zip(fil.pixel_coords, pixels)])
    assert fil.pixel_extents == [(1, 1),
                                 (pixels[0].max(), pixels[1].max())]

    mask_expect = np.zeros(shape, dtype=bool)
    mask_expect[pixels] = True

    mask = fil.skeleton(pad_size=1)

    assert (mask == mask_expect).all()

    # Long path mask should fail without skeleton_analysis
    with pytest.raises(AttributeError):
        fil.skeleton(out_type='longpath')

    # Now run the skeleton analysis
    fil.skeleton_analysis(image)

    # Remove the one branch pixel
    mask_expect = np.zeros(shape, dtype=bool)
    mask_expect[pixels[0][:-2], pixels[1][:-2]] = True

    pad = 1
    mask = fil.skeleton(out_type='longpath', pad_size=pad)
    assert (mask == mask_expect[fil.image_slice(pad_size=pad)]).all()

    # Check the length
    assert fil.length().value == 8.0
    assert fil.length().unit == u.pix

    # Angular and physical conversion should fail b/c no WCS or distance is
    # given
    with pytest.raises(AttributeError):
        fil.length(unit=u.deg)

    with pytest.raises(AttributeError):
        fil.length(unit=u.pc)

    # The branches should all have the same average intensity
    for branch_int in fil.branch_properties['intensity']:
        assert branch_int == 2.0

    # Finally, make sure the transpose shape gives the same length, masks
    fil = Filament2D((pixels[1], pixels[0]))

    assert all([(out == inp).all() for out, inp in
                zip(fil.pixel_coords, pixels[::-1])])
    assert fil.pixel_extents == [(1, 1),
                                 (pixels[1].max(), pixels[0].max())]

    mask_expect = np.zeros(shape, dtype=bool)
    mask_expect[pixels] = True

    mask = fil.skeleton(pad_size=1)

    assert (mask == mask_expect.T).all()

    fil.skeleton_analysis(image.T)

    assert fil.length().value == 8.0

    mask_expect = np.zeros(shape, dtype=bool)
    mask_expect[pixels[0][:-2], pixels[1][:-2]] = True

    mask = fil.skeleton(out_type='longpath', pad_size=pad)
    assert (mask == mask_expect.T).all()
