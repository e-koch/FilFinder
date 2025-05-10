# Licensed under an MIT open source license - see LICENSE

import numpy as np
import numpy.testing as npt
import astropy.units as u

from .. import FilFinder2D

from ._testing_data import *

FWHM_FACTOR = 2 * np.sqrt(2 * np.log(2.))


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
