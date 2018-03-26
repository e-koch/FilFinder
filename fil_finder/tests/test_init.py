# Licensed under an MIT open source license - see LICENSE

import numpy as np
import numpy.testing as npt
import astropy.units as u

from .. import fil_finder_2D
from .. import FilFinder2D

from ._testing_data import *

FWHM_FACTOR = 2 * np.sqrt(2 * np.log(2.))


def test_with_distance():

    distance = 260 * u.pc
    beamwidth = 10.0 * u.arcsec
    # Divide by FWHM factor
    width = beamwidth / (2 * np.sqrt(2 * np.log(2)))

    test1 = fil_finder_2D(img, header=hdr, beamwidth=beamwidth,
                          flatten_thresh=95,
                          distance=distance, size_thresh=430,
                          glob_thresh=20, save_name="test1")

    imgscale = hdr['CDELT2'] * \
        (np.pi / 180.0) * distance.to(u.pc).value
    beamwidth = (width.to(u.arcsec).value / 206265.) * distance.value
    npt.assert_equal(test1.imgscale, imgscale)
    npt.assert_equal(test1.beamwidth, beamwidth)


def test_with_distance_FilFinder2D():
    '''
    FilFinder2D converts everything into pixel units. Conversions back out are
    done on-the-fly.
    '''

    distance = 260 * u.pc
    beamwidth = 10.0 * u.arcsec
    # Divide by FWHM factor
    # width = beamwidth / (2 * np.sqrt(2 * np.log(2)))

    test1 = FilFinder2D(img, header=hdr, beamwidth=beamwidth,
                        distance=distance,
                        save_name="test1")

    imgscale = hdr['CDELT2'] * \
        (np.pi / 180.0) * distance.to(u.pc).value
    ang_pix = hdr['CDELT2'] * u.deg

    filfind_scale = test1.converter.from_pixel(1 * u.pix, u.pc).value
    npt.assert_almost_equal(filfind_scale, imgscale)

    filfind_scale = (test1.converter.from_pixel(1 * u.pix, u.deg)**2).to(u.sr).value
    npt.assert_almost_equal(filfind_scale,
                            (ang_pix**2).to(u.sr).value)
    npt.assert_almost_equal(test1.beamwidth.value,
                            (beamwidth.to(u.deg).value / hdr['CDELT2']))


def test_without_distance():

    beamwidth = 10.0 * u.arcsec

    test1 = fil_finder_2D(img, header=hdr, beamwidth=beamwidth,
                          flatten_thresh=95,
                          distance=None, size_thresh=430,
                          glob_thresh=20, save_name="test1")

    beamwidth = (beamwidth.to(u.deg) /
                 (hdr["CDELT2"] * u.deg)) / FWHM_FACTOR
    imgscale = 1.0
    npt.assert_equal(test1.imgscale, imgscale)
    npt.assert_equal(test1.beamwidth, beamwidth.value)


def test_without_header():

    beamwidth = 10.0 * u.pix
    distance = 260 * u.pc

    test1 = fil_finder_2D(img, header=None, beamwidth=beamwidth,
                          flatten_thresh=95,
                          distance=distance, size_thresh=430,
                          glob_thresh=20, save_name="test1")

    beamwidth = beamwidth / FWHM_FACTOR
    imgscale = 1.0
    npt.assert_equal(test1.imgscale, imgscale)
    npt.assert_equal(test1.beamwidth, beamwidth.value)


def test_without_header_distance():

    beamwidth = 10.0 * u.pix

    test1 = fil_finder_2D(img, header=None, beamwidth=beamwidth,
                          flatten_thresh=95,
                          distance=None, size_thresh=430,
                          glob_thresh=20, save_name="test1")

    beamwidth = beamwidth / FWHM_FACTOR
    imgscale = 1.0
    npt.assert_equal(test1.imgscale, imgscale)
    npt.assert_equal(test1.beamwidth, beamwidth.value)


def test_header_beam():

    beam_hdr = hdr.copy()
    # It only looks for BMAJ at the moment.
    beam_hdr["BMAJ"] = 10.0

    test1 = fil_finder_2D(img, header=beam_hdr, beamwidth=None,
                          flatten_thresh=95,
                          distance=None, size_thresh=430,
                          glob_thresh=20, save_name="test1")

    beamwidth = (10.0 * u.deg / (hdr["CDELT2"] * u.deg)) / FWHM_FACTOR
    imgscale = 1.0
    npt.assert_equal(test1.imgscale, imgscale)
    npt.assert_equal(test1.beamwidth, beamwidth.value)
