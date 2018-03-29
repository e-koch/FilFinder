# Licensed under an MIT open source license - see LICENSE

import pytest

import numpy as np
import astropy.units as u
from astropy.wcs import WCS

from ..base_conversions import UnitConverter

from ._testing_data import hdr


def test_UnitConverter_nowcs():
    '''
    Only pixel to pixel conversions
    '''

    convert = UnitConverter()

    twopix = 2 * u.pix

    assert convert.to_pixel(twopix) == twopix
    assert convert.to_pixel_area(twopix**2) == twopix**2

    with pytest.raises(AttributeError):
        convert.ang_size

    with pytest.raises(AttributeError):
        convert.from_pixel(twopix, u.deg)

    with pytest.raises(AttributeError):
        convert.to_angular(twopix)


def test_UnitConverter_nodistance():

    mywcs = WCS(hdr)

    convert = UnitConverter(mywcs)

    assert convert.ang_size.value == np.abs(hdr['CDELT2'])
    assert convert.ang_size.unit == mywcs.wcs.cunit[0]

    twopix_ang = np.abs(hdr['CDELT2']) * 2 * u.deg
    assert convert.to_pixel(twopix_ang).value == 2.

    assert convert.to_pixel_area(twopix_ang**2) == 4 * u.pix**2

    assert convert.from_pixel(2 * u.pix, u.deg) == twopix_ang
    assert convert.to_angular(2 * u.pix, u.deg) == twopix_ang

    # Round trip
    assert convert.to_pixel(convert.from_pixel(2 * u.pix, u.deg)) == 2 * u.pix
    assert convert.to_pixel(convert.to_angular(2 * u.pix, u.deg)) == 2 * u.pix

    assert convert.from_pixel(convert.to_pixel(twopix_ang), u.deg) == twopix_ang
    assert convert.to_angular(convert.to_pixel(twopix_ang), u.deg) == twopix_ang

    # Physical conversions should fail
    with pytest.raises(AttributeError):
        convert.from_pixel(2 * u.pix, u.pc)
    with pytest.raises(AttributeError):
        assert convert.to_physical(2 * u.pix, u.pc)


def test_UnitConverter():

    mywcs = WCS(hdr)

    distance = 250 * u.pc

    convert = UnitConverter(mywcs, distance=distance)

    assert convert.ang_size.value == np.abs(hdr['CDELT2'])
    assert convert.ang_size.unit == mywcs.wcs.cunit[0]

    twopix_ang = np.abs(hdr['CDELT2']) * 2 * u.deg
    twopix_phys = twopix_ang.to(u.rad).value * distance
    assert convert.to_pixel(twopix_ang).value == 2.
    assert convert.to_pixel(twopix_phys).value == 2.

    assert convert.to_pixel_area(twopix_ang**2) == 4 * u.pix**2
    assert convert.to_pixel_area(twopix_phys**2) == 4 * u.pix**2

    assert convert.from_pixel(2 * u.pix, u.deg) == twopix_ang
    assert convert.to_angular(2 * u.pix, u.deg) == twopix_ang

    assert convert.from_pixel(2 * u.pix, u.pc) == twopix_phys
    assert convert.to_physical(2 * u.pix, u.pc) == twopix_phys

    # Round trip
    assert convert.to_pixel(convert.from_pixel(2 * u.pix, u.deg)) == 2 * u.pix
    assert convert.to_pixel(convert.to_angular(2 * u.pix, u.deg)) == 2 * u.pix

    assert convert.from_pixel(convert.to_pixel(twopix_ang), u.deg) == twopix_ang
    assert convert.to_angular(convert.to_pixel(twopix_ang), u.deg) == twopix_ang

    assert convert.to_pixel(convert.from_pixel(2 * u.pix, u.pc)) == 2 * u.pix
    assert convert.to_pixel(convert.to_physical(2 * u.pix, u.pc)) == 2 * u.pix

    assert convert.from_pixel(convert.to_pixel(twopix_phys), u.pc) == twopix_phys
    assert convert.to_physical(convert.to_pixel(twopix_phys), u.pc) == twopix_phys


@pytest.mark.xfail(raises=u.UnitConversionError)
def test_UnitConverter_bad_distance_unit():

    mywcs = WCS(hdr)

    distance = 250 * u.K

    UnitConverter(mywcs, distance=distance)


@pytest.mark.xfail(raises=TypeError)
def test_UnitConverter_bad_angular_unit():

    mywcs = WCS(hdr)

    convert = UnitConverter(mywcs)

    assert convert.to_pixel(np.abs(hdr['CDELT2']) * 2).value == 2.
