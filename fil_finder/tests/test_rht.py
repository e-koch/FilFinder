
from ..rollinghough import rht

from numpy.testing import assert_allclose

def test_rht():
    test1 = np.eye(20)
    test2 = np.zeros((20,20))
    test2[10, :] = 1

    assert_allclose(rht(test1, 10)[3], (0.61427789316001524, 0.7897858626343055, 0.94774303516116642))
    assert_allclose(rht(test1[::-1], 10)[3], (2.1938496184286262, 2.3518067909554876, 2.5273147604297774))

    assert_allclose(rht(test2, 10)[3], (1.439165349689179, 1.5620209283211821, 1.7024273039006141))
    assert_allclose(rht(test2.T, 10)[3], (3.01873707495779, 0.0, 0.12285557863200314))
