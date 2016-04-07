# Licensed under an MIT open source license - see LICENSE

from unittest import TestCase

import numpy as np
import numpy.testing as npt
import astropy.units as u

from fil_finder import fil_finder_2D

from _testing_data import *

FWHM_FACTOR = 2 * np.sqrt(2 * np.log(2.))


class Test_FilFinder_Units(TestCase):

    def test_with_distance(self):

        distance = 260*u.pc
        beamwidth = 10.0*u.arcsec
        # Divide by FWHM factor
        width = beamwidth / (2*np.sqrt(2*np.log(2)))

        test1 = fil_finder_2D(img, header=hdr, beamwidth=beamwidth,
                              flatten_thresh=95,
                              distance=distance, size_thresh=430,
                              glob_thresh=20, save_name="test1")

        imgscale = hdr['CDELT2'] * \
            (np.pi / 180.0) * distance.to(u.pc).value
        beamwidth = (width.to(u.arcsec).value / 206265.) * distance.value
        npt.assert_equal(test1.imgscale, imgscale)
        npt.assert_equal(test1.beamwidth, beamwidth)

    def test_without_distance(self):

        beamwidth = 10.0*u.arcsec

        test1 = fil_finder_2D(img, header=hdr, beamwidth=beamwidth,
                              flatten_thresh=95,
                              distance=None, size_thresh=430,
                              glob_thresh=20, save_name="test1")

        beamwidth = (beamwidth.to(u.deg) /
                     (hdr["CDELT2"] * u.deg)) / FWHM_FACTOR
        imgscale = 1.0
        npt.assert_equal(test1.imgscale, imgscale)
        npt.assert_equal(test1.beamwidth, beamwidth.value)

    def test_without_header(self):

        beamwidth = 10.0*u.pix
        distance = 260*u.pc

        test1 = fil_finder_2D(img, header=None, beamwidth=beamwidth,
                              flatten_thresh=95,
                              distance=distance, size_thresh=430,
                              glob_thresh=20, save_name="test1")

        beamwidth = beamwidth / FWHM_FACTOR
        imgscale = 1.0
        npt.assert_equal(test1.imgscale, imgscale)
        npt.assert_equal(test1.beamwidth, beamwidth.value)

    def test_without_header_distance(self):

        beamwidth = 10.0*u.pix

        test1 = fil_finder_2D(img, header=None, beamwidth=beamwidth,
                              flatten_thresh=95,
                              distance=None, size_thresh=430,
                              glob_thresh=20, save_name="test1")

        beamwidth = beamwidth / FWHM_FACTOR
        imgscale = 1.0
        npt.assert_equal(test1.imgscale, imgscale)
        npt.assert_equal(test1.beamwidth, beamwidth.value)

    def test_header_beam(self):

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
