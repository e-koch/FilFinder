# Licensed under an MIT open source license - see LICENSE

import numpy as np
import numpy.testing as npt
import astropy.units as u
from copy import deepcopy

from .. import fil_finder_2D, FilFinder2D

from ._testing_data import *


def test_with_rht_branches():

    test1 = fil_finder_2D(img, header=hdr, beamwidth=10.0 * u.arcsec,
                          flatten_thresh=95,
                          distance=260 * u.pc, size_thresh=430,
                          glob_thresh=20, save_name="test1")

    test1.create_mask(border_masking=False)
    test1.medskel()
    test1.analyze_skeletons()
    test1.exec_rht(branches=True)
    test1.find_widths()
    test1.compute_filament_brightness()

    assert ((mask1 > 0) == test1.mask).all()
    assert ((skeletons1 > 0) == test1.skeleton).all()

    assert test1.number_of_filaments == len(table1["Lengths"])

    for i, param in enumerate(test1.width_fits["Names"]):
        npt.assert_allclose(test1.width_fits["Parameters"][:, i],
                            np.asarray(table1[param]), rtol=1e-4)
        npt.assert_allclose(test1.width_fits["Errors"][:, i],
                            np.asarray(table1[param + " Error"]),
                            rtol=1e-4)

    npt.assert_allclose(test1.lengths,
                        table1['Lengths'].quantity.value)

    assert (test1.width_fits['Type'] == table1['Fit Type']).all()

    npt.assert_allclose(test1.total_intensity,
                        table1['Total Intensity'].quantity.value)

    npt.assert_allclose(test1.filament_brightness,
                        table1['Median Brightness'].quantity.value)

    npt.assert_allclose(test1.branch_properties["number"],
                        table1['Branches'].quantity.value)


def test_without_rht_branches():
    # Non-branches

    test2 = fil_finder_2D(img, header=hdr, beamwidth=10.0 * u.arcsec,
                          flatten_thresh=95,
                          distance=260 * u.pc, size_thresh=430,
                          glob_thresh=20, save_name="test2")

    test2.create_mask(border_masking=False)
    test2.medskel()
    test2.analyze_skeletons()
    test2.exec_rht(branches=False)
    test2.find_widths()
    test2.compute_filament_brightness()

    assert ((mask2 > 0) == test2.mask).all()
    assert ((skeletons2 > 0) == test2.skeleton).all()

    assert test2.number_of_filaments == len(table2["Lengths"])

    for i, param in enumerate(test2.width_fits["Names"]):
        npt.assert_allclose(test2.width_fits["Parameters"][:, i],
                            np.asarray(table2[param]), rtol=1e-4)
        npt.assert_allclose(test2.width_fits["Errors"][:, i],
                            np.asarray(table2[param + " Error"]),
                            rtol=1e-4)

    npt.assert_allclose(test2.lengths,
                        table2['Lengths'].quantity.value)

    assert (test2.width_fits['Type'] == table2['Fit Type']).all()

    npt.assert_allclose(test2.total_intensity,
                        table2['Total Intensity'].quantity.value)

    npt.assert_allclose(test2.filament_brightness,
                        table2['Median Brightness'].quantity.value)

    npt.assert_allclose(test2.branch_properties["number"],
                        table2['Branches'].quantity.value)

    npt.assert_allclose(test2.rht_curvature['Orientation'],
                        table2['Orientation'].quantity.value)

    npt.assert_allclose(test2.rht_curvature['Curvature'],
                        table2['Curvature'].quantity.value,
                        atol=5e-3)


def test_equal_branches():
    '''
    Ensure the filament arrays are equal with and without computing the
    RHT branches.
    '''

    test1 = fil_finder_2D(img, header=hdr, beamwidth=10.0 * u.arcsec,
                          flatten_thresh=95,
                          distance=260 * u.pc, size_thresh=430,
                          glob_thresh=20, save_name="test1")

    test1.create_mask(border_masking=False)
    test1.medskel()
    test1.analyze_skeletons()
    test1.exec_rht(branches=True)

    test_copy = deepcopy(test1)

    test_copy.exec_rht(branches=False)

    for arr1, arr2 in zip(test1.filament_arrays['final'],
                          test_copy.filament_arrays['final']):
        assert np.allclose(arr1, arr2)

    for arr1, arr2 in zip(test1.filament_arrays['long path'],
                          test_copy.filament_arrays['long path']):
        assert np.allclose(arr1, arr2)


def test_FilFinder2D_w_rhtbranches():
    '''
    Make sure the new FilFinder2D gives consistent results
    '''

    from fil_finder.tests._testing_data import img, hdr
    from fil_finder import FilFinder2D, fil_finder_2D

    test1 = FilFinder2D(img, header=hdr, beamwidth=10.0 * u.arcsec,
                        distance=260 * u.pc, save_name="test1")

    test1.preprocess_image(flatten_percent=95)
    test1.create_mask(glob_thresh=np.nanpercentile(img, 20),
                      size_thresh=430 * u.pix**2,
                      border_masking=False)

    test1_old = fil_finder_2D(img, header=hdr, beamwidth=10.0 * u.arcsec,
                              flatten_thresh=95,
                              distance=260 * u.pc, size_thresh=430,
                              glob_thresh=20, save_name="test1_old")

    test1_old.create_mask(border_masking=False)

    # Enforce the masks to be the same
    assert (test1.mask == test1_old.mask).all()

    # Now the skeletons
    test1.medskel()
    test1_old.medskel()

    assert (test1.skeleton == test1_old.skeleton).all()

    test1.analyze_skeletons(skel_thresh=40 * u.pix, branch_thresh=2.0 * u.pix)
    test1_old.analyze_skeletons()

    assert test1.number_of_filaments == test1_old.number_of_filaments
    assert (test1.skeleton == test1_old.skeleton).all()
    npt.assert_allclose(test1.lengths.value,
                        test1_old.lengths / test1_old.imgscale, atol=0.5)

    # Same branch properties
    for branch, branch_old in zip(test1.branch_properties['length'],
                                  test1_old.branch_properties['length']):
        assert len(branch) == len(branch_old)
        npt.assert_allclose(branch.value,
                            np.array(branch_old) / test1_old.imgscale)
    for branch, branch_old in zip(test1.branch_properties['intensity'],
                                  test1_old.branch_properties['intensity']):
        assert len(branch) == len(branch_old)
        npt.assert_allclose(branch, np.array(branch_old))

    for branch, branch_old in zip(test1.branch_properties['number'],
                                  test1_old.branch_properties['number']):
        assert branch == branch_old

    # XXX Remove this once the other parts of Filament2D are implemented
    print(argh)

    test1.exec_rht(branches=True)
    test1.find_widths()
    test1.compute_filament_brightness()

    assert ((mask1 > 0) == test1.mask).all()
    assert ((skeletons1 > 0) == test1.skeleton).all()

    assert test1.number_of_filaments == len(table1["Lengths"])

    for i, param in enumerate(test1.width_fits["Names"]):
        npt.assert_allclose(test1.width_fits["Parameters"][:, i],
                            np.asarray(table1[param]), rtol=1e-4)
        npt.assert_allclose(test1.width_fits["Errors"][:, i],
                            np.asarray(table1[param + " Error"]),
                            rtol=1e-4)

    npt.assert_allclose(test1.lengths,
                        table1['Lengths'].quantity.value)

    assert (test1.width_fits['Type'] == table1['Fit Type']).all()

    npt.assert_allclose(test1.total_intensity,
                        table1['Total Intensity'].quantity.value)

    npt.assert_allclose(test1.filament_brightness,
                        table1['Median Brightness'].quantity.value)

    npt.assert_allclose(test1.branch_properties["number"],
                        table1['Branches'].quantity.value)
