
from fil_finder.width import nonparam_width, gauss_model

import numpy as np
import numpy.testing as npt


def test_nonparam():

    pts = np.linspace(0, 10, 100)

    profile = 2.0 * np.exp(-pts ** 2 / (2 * 3.0 ** 2)) + 0.5

    params, errors, fail = \
        nonparam_width(pts, profile, pts, profile, 1.0, 5, 99)

    # This shouldn't be failing
    assert fail is False

    # Check the amplitude
    npt.assert_allclose(params[0], 2.5, atol=0.01)
    # Width
    npt.assert_allclose(params[1], 3.0, atol=0.01)
    # Background
    npt.assert_allclose(params[2], 0.5, atol=0.02)


def test_gaussian():

    pts = np.linspace(0, 10, 100)

    profile = 2.0 * np.exp(-pts ** 2 / (2 * 3.0 ** 2)) + 0.5

    params, errors, _, _, fail = \
        gauss_model(pts, profile, np.ones_like(pts), 1.0)

    # Check the amplitude
    npt.assert_allclose(params[0], 2.5, atol=0.01)
    # Width
    npt.assert_allclose(params[1], 3.0, atol=0.01)
    # Background
    npt.assert_allclose(params[2], 0.5, atol=0.02)
