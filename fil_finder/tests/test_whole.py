# Licensed under an MIT open source license - see LICENSE

from unittest import TestCase

import numpy as np
import numpy.testing as npt

from fil_finder import fil_finder_2D

from _testing_data import *


class Test_FilFinder(TestCase):

    def test_with_rht_branches(self):

        test1 = fil_finder_2D(img, hdr, 10.0, flatten_thresh=95,
                              distance=260, size_thresh=430,
                              glob_thresh=20, save_name="test1",
                              pad_size=10, skeleton_pad_size=10)

        test1.create_mask(border_masking=False)
        test1.medskel()
        test1.analyze_skeletons()
        test1.exec_rht(branches=True)
        test1.find_widths()
        test1.compute_filament_brightness()

        assert test1.number_of_filaments == len(table1["Lengths"])

        for i, param in enumerate(test1.width_fits["Names"]):
            npt.assert_allclose(test1.width_fits["Parameters"][:, i],
                                np.asarray(table1[param]))
            npt.assert_allclose(test1.width_fits["Errors"][:, i],
                                np.asarray(table1[param+" Error"]))

        assert np.allclose(test1.lengths,
                           np.asarray(table1['Lengths']))

        assert (test1.width_fits['Type'] == table1['Fit Type']).all()

        assert np.allclose(test1.total_intensity,
                           np.asarray(table1['Total Intensity']))

        assert np.allclose(test1.filament_brightness,
                           np.asarray(table1['Median Brightness']))

        assert np.allclose(test1.branch_properties["number"],
                           np.asarray(table1['Branches']))

    def test_without_rht_branches(self):
        # Non-branches

        test2 = fil_finder_2D(img, hdr, 10.0, flatten_thresh=95,
                              distance=260, size_thresh=430,
                              glob_thresh=20, save_name="test2",
                              pad_size=10, skeleton_pad_size=10)

        test2.create_mask(border_masking=False)
        test2.medskel()
        test2.analyze_skeletons()
        test2.exec_rht(branches=False)
        test2.find_widths()
        test2.compute_filament_brightness()

        assert test2.number_of_filaments == len(table2["Lengths"])

        for i, param in enumerate(test2.width_fits["Names"]):
            assert np.allclose(test2.width_fits["Parameters"][:, i],
                               np.asarray(table2[param]))
            assert np.allclose(test2.width_fits["Errors"][:, i],
                               np.asarray(table2[param+" Error"]))

        assert np.allclose(test2.lengths,
                           np.asarray(table2['Lengths']))

        assert (test2.width_fits['Type'] == table2['Fit Type']).all()

        assert np.allclose(test2.total_intensity,
                           np.asarray(table2['Total Intensity']))

        assert np.allclose(test2.filament_brightness,
                           np.asarray(table2['Median Brightness']))

        assert np.allclose(test2.branch_properties["number"],
                           np.asarray(table2['Branches']))

        assert np.allclose(test2.rht_curvature['Median'],
                           np.asarray(table2['Orientation']))

        assert np.allclose(test2.rht_curvature['IQR'],
                           np.asarray(table2['Curvature']))
