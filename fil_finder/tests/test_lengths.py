# Licensed under an MIT open source license - see LICENSE

from ..length import skeleton_length
from ..pixel_ident import is_blockpoint, is_tpoint

import numpy as np


def test_length1():

    test_skel1 = np.zeros((10, 10))

    test_skel1[1, 1:3] = 1
    test_skel1[2, 3] = 1
    test_skel1[3, 4:6] = 1

    length1 = skeleton_length(test_skel1)

    assert length1 == 2 + np.sqrt(2) * 2


def test_length2():

    test_skel2 = np.eye(10)

    length2 = skeleton_length(test_skel2)

    assert length2 == 9 * np.sqrt(2)


def test_length3():

    test_skel3 = np.zeros((10, 10))

    test_skel3[:, 5] = 1

    length3 = skeleton_length(test_skel3)

    assert length3 == 9


def test_length4():

    test_skel4 = np.zeros((12, 12))

    posns = \
        [(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 11, 10, 9, 9),
         (3, 2, 1, 0, 1, 2, 3, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11)]

    test_skel4[posns] = 1

    length4 = skeleton_length(test_skel4)

    assert length4 == 11 * np.sqrt(2) + 5


def test_length5():

    test_skel5 = np.zeros((9, 12))

    posns = \
        [(8, 8, 7, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 1, 2, 3, 4, 5),
         (0, 1, 2, 3, 4, 4, 4, 5, 5, 5, 6, 7, 8, 9, 10, 10, 11, 11)]

    test_skel5[posns] = 1

    length5 = skeleton_length(test_skel5)

    assert length5 == 10 + 7 * np.sqrt(2)


def test_length6():

    test_skel6 = np.zeros((4, 4))

    test_skel6[1, 0:2] = 1
    test_skel6[2:4, 2] = 1

    length6 = skeleton_length(test_skel6)

    assert length6 == 2 + np.sqrt(2)


def test_isblockpt():

    # test four different configurations, and once with an added connection

    # 0 1 6
    vallist = [1, 1, 0, 0, 0, 0, 0, 1]

    assert is_blockpoint(vallist)

    vallist = [1, 1, 0, 0, 1, 0, 0, 1]

    assert is_blockpoint(vallist)

    # 1 2 3
    vallist = [0, 1, 1, 1, 0, 0, 0, 0]

    assert is_blockpoint(vallist)

    vallist = [0, 1, 1, 1, 0, 1, 0, 0]

    assert is_blockpoint(vallist)

    # 3 4 5
    vallist = [0, 0, 0, 1, 1, 1, 0, 0, 0]

    assert is_blockpoint(vallist)

    vallist = [1, 0, 0, 1, 1, 1, 0, 0, 0]

    assert is_blockpoint(vallist)

    # 5 6 7
    vallist = [0, 0, 0, 0, 0, 1, 1, 1]

    assert is_blockpoint(vallist)

    vallist = [0, 1, 0, 0, 0, 1, 1, 1]

    assert is_blockpoint(vallist)


def test_isTpoint():

    # 0 6 7
    vallist = [1, 0, 0, 0, 0, 0, 1, 1]

    assert is_tpoint(vallist)

    vallist = [1, 0, 0, 1, 0, 0, 1, 1]

    assert is_tpoint(vallist)

    # 0 1 2
    vallist = [1, 1, 1, 0, 0, 0, 0, 0]

    assert is_tpoint(vallist)

    vallist = [1, 1, 1, 0, 1, 0, 0, 0]

    assert is_tpoint(vallist)

    # 2 3 4
    vallist = [0, 0, 1, 1, 1, 0, 0, 0, 0]

    assert is_tpoint(vallist)

    vallist = [0, 0, 1, 1, 1, 0, 0, 1, 0]

    assert is_tpoint(vallist)

    # 4 5 6
    vallist = [0, 0, 0, 0, 1, 1, 1, 0]

    assert is_tpoint(vallist)

    vallist = [1, 0, 0, 0, 1, 1, 1, 0]

    assert is_tpoint(vallist)

