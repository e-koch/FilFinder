# Licensed under an MIT open source license - see LICENSE

import numpy as np
import numpy.testing as npt
import astropy.units as u
from copy import deepcopy
import warnings

from .. import fil_finder_2D, FilFinder2D
from .testing_utils import generate_filament_model

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
    # # XXX Remove this once the other parts of Filament2D are implemented
    # print(argh)

    # test1.compute_filament_brightness()


def test_simple_filament():
    '''
    Check the outputs using a simple straight filament with a Gaussian profile.
    '''

    mod = generate_filament_model(return_hdu=True, pad_size=30, shape=150,
                                  width=10., background=0.1)[0]

    mask = mod.data > 0.5

    test = FilFinder2D(mod, distance=250 * u.pc, mask=mask)

    test.preprocess_image(flatten_percent=85)

    test.create_mask(border_masking=True, verbose=False,
                     use_existing_mask=True)
    test.medskel(verbose=False)
    test.analyze_skeletons()
    test.find_widths(auto_cut=False, max_dist=30 * u.pix)

    fil1 = test.filaments[0]

    test1_old = fil_finder_2D(mod,
                              flatten_thresh=85,
                              distance=250 * u.pc,
                              glob_thresh=0, save_name="test1_old",
                              skeleton_pad_size=30,
                              mask=mask)

    test1_old.create_mask(border_masking=False, use_existing_mask=True)
    test1_old.medskel()
    test1_old.analyze_skeletons()
    test1_old.find_widths(auto_cut=False, verbose=False, max_distance=0.3,
                          try_nonparam=False)

    # Compare lengths
    # Straight skeleton, so length is sum minus 1. Then add the FWHM width on
    # Beam is set to 3 pixels FWHM, so deconvolve before adding
    exp_length = (test.skeleton.sum() - 1) + np.sqrt(10**2 - 3**2) * 2.35

    old_length = test1_old.lengths[0] / test1_old.imgscale

    new_length = test.lengths()[0].value

    # Require the length be within half the beam.
    npt.assert_allclose(exp_length, old_length, atol=1.5)
    npt.assert_allclose(exp_length, new_length, atol=1.5)

    # Now compare the widths
    # Expected profile properties
    exp_pars = [1.1, 10.0, 0.1, np.sqrt(10**2 - 3**2) * 2.35]

    old_pars = test1_old.width_fits['Parameters'][0]
    # Convert widths into pix units
    old_pars[1] = old_pars[1] / test1_old.imgscale
    old_pars[-1] = old_pars[-1] / test1_old.imgscale

    new_pars = [par.value for par in fil1.radprof_params] + \
        [fil1.radprof_fwhm()[0].value]
    # The new modeling correctly separates the Gaussian and bkg.
    # Add the bkg to the amplitude
    new_pars[0] += new_pars[2]

    npt.assert_allclose(exp_pars, old_pars, rtol=0.05)
    npt.assert_allclose(exp_pars, new_pars, rtol=0.05)
