# Licensed under an MIT open source license - see LICENSE

from ..rollinghough import rht

import numpy as np
from numpy.testing import assert_allclose


def test_rht_eye():

    test1 = np.eye(20)

    expected_output = \
        (0.56726012004973192, 0.78609389334676461, 1.004927666643797)

    expected_flipped_output = \
        (-0.9992547253703965, -0.78301561713294787, -0.56677650889549924)

    assert_allclose(rht(test1, 10)[2], expected_output, atol=1e-3)
    assert_allclose(rht(test1[::-1], 10)[2], expected_flipped_output,
                    atol=1e-3)


def test_rht_straight():

    test2 = np.zeros((20, 20))
    test2[10, :] = 1

    expected_output = \
        (1.4325334658241844, 1.5707963267948966, 1.7090591877656087)

    expected_flipped_output = \
        (-0.13975291307401344, -1.2236762866959916e-16,
         0.13975291307401316)

    assert_allclose(rht(test2, 10)[2], expected_output, atol=1e-3)
    assert_allclose(rht(test2.T, 10)[2], expected_flipped_output,
                    atol=1e-3)
