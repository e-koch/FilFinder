# Licensed under an MIT open source license - see LICENSE

from unittest import TestCase

import numpy as np
import numpy.testing as npt
import astropy.units as u
from astropy.io.fits import PrimaryHDU

from fil_finder.io_funcs import input_data

from _testing_data import *

FWHM_FACTOR = 2 * np.sqrt(2 * np.log(2.))


class Test_FilFinder_Input_Types(TestCase):

    def test_array_input(self):

        output = input_data(img)

        npt.assert_equal(img, output["data"])

    def test_HDU_input(self):

        hdu = PrimaryHDU(img, header=hdr)

        output = input_data(hdu)

        npt.assert_equal(img, output["data"])
        npt.assert_equal(hdr, output["header"])

    def test_3D_input(self):

        try:
            output = input_data(np.ones((3,) * 3))
        except Exception, e:
            assert isinstance(e, TypeError)

    def test_3D_squeezable_input(self):

        output = input_data(np.ones((3,3,1)))

        npt.assert_equal(np.ones((3,3)), output["data"])


