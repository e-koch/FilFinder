
from fil_finder.rollinghough import rht

import numpy as np
from numpy.testing import assert_allclose

def test_rht():
    test1 = np.eye(20)
    test2 = np.zeros((20,20))
    test2[10, :] = 1

    assert_allclose(rht(test1, 10)[2], (0.56726012004973192, 0.78609389334676461, 1.004927666643797))
    assert_allclose(rht(test1[::-1], 10)[2], (-0.9992547253703965, -0.78301561713294787, -0.56677650889549924))

    assert_allclose(rht(test2, 10)[2], (1.4325334658241844, 1.5707963267948966, 1.7090591877656087))
    assert_allclose(rht(test2.T, 10)[2], (-0.13975291307401344, -1.2236762866959916e-16, 0.13975291307401316))
