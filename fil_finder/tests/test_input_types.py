# Licensed under an MIT open source license - see LICENSE

import pytest

import numpy as np
import numpy.testing as npt
import astropy.units as u
from astropy.io.fits import PrimaryHDU

try:
    from spectral_cube import Projection, Slice
    SPECTRALCUBE_INSTALL = True
except ImportError:
    SPECTRALCUBE_INSTALL = False


from ..io_funcs import input_data

from ._testing_data import *


def test_array_input():

    output = input_data(img)

    npt.assert_equal(img, output["data"].value)
    assert output['data'].unit == u.dimensionless_unscaled


def test_array_input_withheader():

    new_hdr = hdr.copy()
    new_hdr['BUNIT'] = 'K'
    output = input_data(img, header=new_hdr)

    npt.assert_equal(img, output["data"].value)
    assert output['data'].unit == u.K
    npt.assert_equal(new_hdr, output["header"])


def test_quantity_input():

    quant = img * u.K

    output = input_data(quant)

    npt.assert_equal(img, output["data"].value)
    assert output['data'].unit == u.K


def test_quantity_input_withheader():

    quant = img * u.K

    # Give the header a different BUNIT. Always use the unit
    # attached to the Quantity object
    new_hdr = hdr.copy()
    new_hdr['BUNIT'] = 'Jy/beam'

    output = input_data(quant, header=new_hdr)

    npt.assert_equal(img, output["data"].value)
    assert output['data'].unit == u.K

    # The header should now have K set
    assert output['header']['BUNIT'] == 'K'


def test_HDU_input():

    hdu = PrimaryHDU(img, header=hdr)

    output = input_data(hdu)

    npt.assert_equal(img, output["data"].value)
    assert output['data'].unit == u.dimensionless_unscaled
    npt.assert_equal(hdr, output["header"])


def test_HDU_input_withbunit():

    hdr['BUNIT'] = 'K'
    hdu = PrimaryHDU(img, header=hdr)

    output = input_data(hdu)

    npt.assert_equal(img, output["data"].value)
    assert output['data'].unit == u.K
    npt.assert_equal(hdr, output["header"])


@pytest.mark.skipif("not SPECTRALCUBE_INSTALL")
def test_SC_inputs():

    hdr['BUNIT'] = 'K'
    hdu = PrimaryHDU(img, header=hdr)

    proj = Projection.from_hdu(hdu)

    output = input_data(proj)

    npt.assert_equal(img, output["data"].value)
    assert output['data'].unit == u.K
    npt.assert_equal(proj.header, output["header"])

    slic = Slice.from_hdu(hdu)

    output = input_data(slic)

    npt.assert_equal(img, output["data"].value)
    assert output['data'].unit == u.K
    npt.assert_equal(slic.header, output["header"])


def test_3D_input():

    with pytest.raises(TypeError):
        input_data(np.ones((3, ) * 3))


def test_3D_squeezable_input():

    output = input_data(np.ones((3, 3, 1)))

    npt.assert_equal(np.ones((3, 3)), output["data"].value)
    assert output['data'].unit == u.dimensionless_unscaled
