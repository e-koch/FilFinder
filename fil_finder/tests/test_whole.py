# Licensed under an MIT open source license - see LICENSE

import numpy as np
import numpy.testing as npt
import astropy.units as u
from copy import deepcopy
import warnings
import os

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
    import numpy.testing as npt
    import warnings

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

    # Compare median brightness
    med_bright = test1.median_brightness()

    test1_old.compute_filament_brightness()
    med_bright_old = test1_old.filament_brightness

    assert (med_bright == np.array(med_bright_old)).all()

    # Compute model image
    fil_model = test1.filament_model(bkg_subtract=True)
    # Cannot compare with old version due to radial profile discrepancies.

    # Same for the total intensities, but run it to make sure it works.
    total_intensity = test1.total_intensity()

    cov_frac = test1.covering_fraction()
    npt.assert_allclose(0.544, cov_frac.value, atol=0.001)


def test_simple_filament():
    '''
    Check the outputs using a simple straight filament with a Gaussian profile.
    '''

    from fil_finder import fil_finder_2D, FilFinder2D
    from fil_finder.tests.testing_utils import generate_filament_model

    mod = generate_filament_model(return_hdu=True, pad_size=30, shape=150,
                                  width=10., background=0.1)[0]

    mask = mod.data > 0.5

    test = FilFinder2D(mod, distance=250 * u.pc, mask=mask,
                       save_name='test1')

    test.preprocess_image(flatten_percent=85)

    test.create_mask(border_masking=True, verbose=False,
                     use_existing_mask=True)
    test.medskel(verbose=False)
    test.analyze_skeletons()
    test.find_widths(auto_cut=False, max_dist=30 * u.pix)

    test.exec_rht(branches=False)
    test.exec_rht(branches=True)

    # Should be oriented along the x-axis. Set to be pi/2.
    npt.assert_allclose(np.pi / 2., test.orientation[0].value)
    npt.assert_allclose(np.pi / 2., test.orientation_branches[0][0].value)

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

    # Test the non-param fitting in the new code
    test.find_widths(auto_cut=False, max_dist=30 * u.pix,
                     fit_model='nonparam')

    new_pars = [par.value for par in fil1.radprof_params]

    # There's a larger discrepancy compared with the Gaussian model
    npt.assert_allclose(exp_pars[:-1], new_pars, rtol=0.2)

    # Use the Gaussian fit for the model comparisons below.
    test.find_widths(auto_cut=False, max_dist=30 * u.pix)


    # Test other output of the new code.

    npt.assert_allclose(1.1, fil1.median_brightness(mod.data))
    npt.assert_allclose(mod.data[(mod.data - 0.1) > 0.5].sum(),
                        fil1.total_intensity().value, rtol=0.01)
    npt.assert_allclose((mod.data - 0.1)[(mod.data - 0.1) > 0.5].sum(),
                        fil1.total_intensity(bkg_subtract=True).value,
                        rtol=0.01)

    fil_model = test.filament_model(bkg_subtract=False)

    # Max difference should be where the background isn't covered
    assert ((mod.data - fil_model.value) <= 0.1 + 1e-7).all()

    # Now compare bkg subtracted versions
    fil_model = test.filament_model(bkg_subtract=True)
    assert ((mod.data - fil_model.value) <= 0.1 + 1e-3).all()

    # Covering fraction
    cov_frac = test.covering_fraction()
    act_frac = (mod.data - 0.1).sum() / np.sum(mod.data)
    npt.assert_allclose(cov_frac.value, act_frac, atol=1e-4)

    # Ridge profile along skeleton. Should all equal 1.1
    ridge = fil1.ridge_profile(test.image)
    assert ridge.unit == u.K
    assert (ridge.value == 1.1).all()

    # Make sure the version from FilFinder2D is the same
    ridge_2 = test.ridge_profiles()
    assert (ridge_2[0] == ridge).all()

    # Test radial profiles
    dists, profs = fil1.profile_analysis(test.image)

    # Width is 10 pixels
    exp_profile = np.exp(- dists[0].value**2 / (2 * 10.**2)) + 0.1

    for prof in profs:
        npt.assert_allclose(prof.value, exp_profile)

    # Test saving methods from Filament2D
    from astropy.table import Table

    tab_rad = fil1.radprof_table()
    npt.assert_allclose(fil1.radprofile[0], tab_rad['distance'])
    npt.assert_allclose(fil1.radprofile[1], tab_rad['values'])
    del tab_rad

    tab_branch = fil1.branch_table()
    npt.assert_allclose(fil1.branch_properties['length'],
                        tab_branch['length'])
    npt.assert_allclose(fil1.branch_properties['intensity'],
                        tab_branch['intensity'])
    del tab_branch

    # With RHT info
    tab_branch = fil1.branch_table(include_rht=True)
    npt.assert_allclose(fil1.branch_properties['length'],
                        tab_branch['length'])
    npt.assert_allclose(fil1.branch_properties['intensity'],
                        tab_branch['intensity'])
    npt.assert_allclose(fil1.orientation_branches,
                        tab_branch['orientation'])
    npt.assert_allclose(fil1.curvature_branches,
                        tab_branch['curvature'])
    del tab_branch

    # Test table output from FilFinder2D

    branch_tables = test.branch_tables()
    assert (branch_tables[0] == fil1.branch_table()).all()

    branch_tables = test.branch_tables(include_rht=True)
    assert (branch_tables[0] == fil1.branch_table(include_rht=True)).all()

    out_tab = test.output_table()


    # Compare saving filament stamps.
    from astropy.io import fits

    fil1.save_fits("test_image_output.fits", test.image)

    hdu = fits.open("test_image_output.fits")
    skel = fil1.skeleton(pad_size=20)
    npt.assert_allclose(skel, hdu[1].data.astype(bool))

    skel = fil1.skeleton(pad_size=20, out_type='longpath')
    npt.assert_allclose(skel, hdu[2].data.astype(bool))

    mod = fil1.model_image()
    npt.assert_allclose(mod.value, hdu[3].data)

    os.remove("test_image_output.fits")
    del hdu

    test.save_stamp_fits()
    hdu = fits.open("test1_stamp_0.fits")
    skel = fil1.skeleton(pad_size=20)
    npt.assert_allclose(skel, hdu[1].data.astype(bool))

    skel = fil1.skeleton(pad_size=20, out_type='longpath')
    npt.assert_allclose(skel, hdu[2].data.astype(bool))

    mod = fil1.model_image()
    npt.assert_allclose(mod.value, hdu[3].data)

    os.remove("test1_stamp_0.fits")
    del hdu

    # Compare saving whole skeleton/mask/model

    test.save_fits()
    hdu = fits.open("test1_image_output.fits")

    mod = test.filament_model()

    npt.assert_allclose(test.mask, hdu[0].data)
    npt.assert_allclose(test.skeleton, hdu[1].data > 0)
    npt.assert_allclose(test.skeleton_longpath, hdu[2].data > 0)
    npt.assert_allclose(mod.value, hdu[3].data)

    os.remove("test1_image_output.fits")
    del hdu
