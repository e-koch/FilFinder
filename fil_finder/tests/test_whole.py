# Licensed under an MIT open source license - see LICENSE

import numpy as np
import numpy.testing as npt
import astropy.units as u
from copy import deepcopy
import warnings

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
    test1.find_widths(auto_cut=True)
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
    test2.find_widths(auto_cut=True)
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

    # Don't use iterative pruning in order to match old version.
    test1.analyze_skeletons(skel_thresh=40 * u.pix, branch_thresh=2.0 * u.pix,
                            max_prune_iter=1)
    test1_old.analyze_skeletons()

    assert test1.number_of_filaments == test1_old.number_of_filaments
    assert (test1.skeleton == test1_old.skeleton).all()
    npt.assert_allclose(test1.lengths().value,
                        test1_old.lengths / test1_old.imgscale, atol=1e-3)

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

    for branches, branches_old, off in zip(test1.branch_properties['pixels'],
                                           test1_old.branch_properties['pixels'],
                                           test1_old.array_offsets):
        assert len(branches) == len(branches_old)
        for branch, branch_old in zip(branches, branches_old):
            # Adjust the old version by the pixel offset
            branch_old[:, 0] -= off[0][0]
            branch_old[:, 1] -= off[0][1]
            npt.assert_allclose(branch, branch_old)

    for branch, branch_old in zip(test1.branch_properties['number'],
                                  test1_old.branch_properties['number']):
        assert branch == branch_old

    test1_old.exec_rht(branches=False)

    test1.exec_rht(branches=False)

    assert (test1_old.rht_curvature['Orientation'] == test1.orientation.value).all()
    assert (test1_old.rht_curvature['Curvature'] == test1.curvature.value).all()

    test1_old.exec_rht(branches=True)

    test1.exec_rht(branches=True)

    for branch, branch_old in zip(test1.orientation_branches,
                                  test1_old.rht_curvature['Orientation']):

        npt.assert_allclose(branch.value[np.isfinite(branch)], branch_old)

    # How the radial profiles are built in the new and old versions differs
    # quite a bit (this is one of the biggest reasons for the new version).
    # The fit parameters will NOT be the same because the old version is
    # not correctly building portions of the profiles! Check that that
    # the warning is raised
    test1.find_widths(max_dist=0.3 * u.pc, try_nonparam=True, auto_cut=False,
                      pad_to_distance=0 * u.pix)

    with warnings.catch_warnings(record=True) as w:
        test1_old.find_widths(auto_cut=False, pad_to_distance=0.,
                              max_distance=0.3)

    assert len(w) == 1
    assert w[0].category == UserWarning
    assert str(w[0].message) == ("An array offset issue is present in the radial profiles"
                                 "! Please use the new version in FilFinder2D. "
                                 "Double-check all results from this function!")

    # Compare median brightness
    med_bright = test1.median_brightness()

    test1_old.compute_filament_brightness()
    med_bright_old = test1_old.filament_brightness

    if hasattr(med_bright, 'unit'):
        assert (med_bright.value == np.array(med_bright_old)).all()
    else:
        assert (med_bright == np.array(med_bright_old)).all()

    # Compute model image
    fil_model = test1.filament_model(bkg_subtract=True)
    # Cannot compare with old version due to radial profile discrepancies.

    # Same for the total intensities, but run it to make sure it works.
    total_intensity = test1.total_intensity()

    cov_frac = test1.covering_fraction()
    npt.assert_allclose(0.544, cov_frac, atol=0.001)

    fil_posns = test1.filament_positions()
    fil_posns = test1.filament_positions(world_coord=True)

    tab = test1.output_table()
    tab = test1.output_table(world_coord=True)


def test_FilFinder2D_iterat_prune():
    '''
    Check that iterative pruning converges to the longest path.
    '''

    test1 = FilFinder2D(img, header=hdr, beamwidth=10.0 * u.arcsec,
                        distance=260 * u.pc, save_name="test1")

    test1.preprocess_image(flatten_percent=95)
    test1.create_mask(glob_thresh=np.nanpercentile(img, 20),
                      size_thresh=430 * u.pix**2,
                      border_masking=False)
    test1.medskel()

    # Don't use iterative pruning in order to match old version.
    # Extremely high branch threshold should force all branches off of the
    # longest path to be removed.
    test1.analyze_skeletons(skel_thresh=40 * u.pix,
                            branch_thresh=400 * u.pix,
                            max_prune_iter=20,
                            prune_criteria='length')

    # All of the skeletons should now be equal to the longest path
    assert (test1.skeleton == test1.skeleton_longpath).all()
