# Licensed under an MIT open source license - see LICENSE

import pytest
import numpy as np
import numpy.testing as npt
import astropy.units as u
import os

from .. import FilFinder2D


@pytest.mark.openfiles_ignore
def test_simple_filament_noheader(simple_filament_model):
    '''
    Check the outputs using a simple straight filament with a Gaussian profile.
    No FITS header is given; outputs must have pixel units in all cases.
    '''

    mod = simple_filament_model

    mask = mod.data > 0.5

    test = FilFinder2D(mod.data, mask=mask, save_name='test1')

    test.preprocess_image(flatten_percent=85)

    test.create_mask(use_existing_mask=True)

    test.medskel(verbose=False)

    # Fails without specifying thresholds for skeleton and branch lengths
    with pytest.raises(ValueError) as exc:
        test.analyze_skeletons()
    assert exc.value.args[0] == "Distance not given. Must specify skel_thresh in pixel units."

    test.analyze_skeletons(nthreads=2, skel_thresh=5 * u.pix)

    test.find_widths(nthreads=2, auto_cut=False, max_dist=30 * u.pix)

    test.exec_rht(nthreads=2, branches=False)
    test.exec_rht(nthreads=2, branches=True)

    # Should be oriented along the x-axis. Set to be pi/2.
    npt.assert_allclose(np.pi / 2., test.orientation[0].value)
    npt.assert_allclose(np.pi / 2., test.orientation_branches[0][0].value)

    fil1 = test.filaments[0]

    # Compare the branch pts with and without image coordinates
    branch_pts = fil1.branch_pts(img_coords=False)
    branch_pts_img = fil1.branch_pts(img_coords=True)

    # Should be sliced down to a single pixel skeleton
    assert len(branch_pts) == 1
    # Padded by 1 for morphological operations
    assert (branch_pts[0][:, 0] == 1).all()
    assert branch_pts[0][:, 1].shape[0] == 151
    assert (branch_pts[0][:, 1] == np.arange(1, 152)).all()

    # Image coordinate branch pts should match the skeleton
    skel_pts = np.where(test.skeleton)
    assert (branch_pts_img[0][:, 0] == skel_pts[0]).all()
    assert (branch_pts_img[0][:, 1] == skel_pts[1]).all()

    # Compare lengths
    # Straight skeleton, so length is sum minus 1. Then add the FWHM width on
    # Beam is set to 3 pixels FWHM, so deconvolve before adding
    exp_length = (test.skeleton.sum() - 1) + np.sqrt(10**2 - 3**2) * 2.35

    new_length = test.lengths()[0].value

    # Require the length be within half the beam.
    npt.assert_allclose(exp_length, new_length, atol=1.5)

    # Now compare the widths
    # Expected profile properties
    exp_pars = [1.1, 10.0, 0.1, np.sqrt(10**2 - 3**2) * 2.35]

    new_pars = [par.value for par in fil1.radprof_params] + \
        [fil1.radprof_fwhm()[0].value]
    # The new modeling correctly separates the Gaussian and bkg.
    # Add the bkg to the amplitude
    new_pars[0] += new_pars[2]

    npt.assert_allclose(exp_pars, new_pars, rtol=0.05)

    # Test the non-param fitting in the new code
    test.find_widths(nthreads=2, auto_cut=False, max_dist=30 * u.pix,
                     fit_model='nonparam')

    new_pars = [par.value for par in fil1.radprof_params]

    # There's a larger discrepancy compared with the Gaussian model
    npt.assert_allclose(exp_pars[:-1], new_pars, rtol=0.2)

    # Use the Gaussian fit for the model comparisons below.
    test.find_widths(nthreads=2, auto_cut=False, max_dist=30 * u.pix)

    # Test other output of the new code.

    npt.assert_allclose(1.1, fil1.median_brightness(mod.data))
    npt.assert_allclose(mod.data[(mod.data - 0.1) > 0.5].sum(),
                        fil1.total_intensity().value, rtol=0.01)
    npt.assert_allclose((mod.data - 0.1)[(mod.data - 0.1) > 0.5].sum(),
                        fil1.total_intensity(bkg_subtract=True).value,
                        rtol=0.01)

    fil_model = test.filament_model(bkg_subtract=False)
    # Some astropy 3.0.1 does not support compound model units. Check
    # the output type here.
    if hasattr(fil_model, 'unit'):
        fil_model = fil_model.value

    # Max difference should be where the background isn't covered
    assert ((mod.data - fil_model) <= 0.1 + 1e-7).all()

    # Now compare bkg subtracted versions
    fil_model = test.filament_model(bkg_subtract=True)
    if hasattr(fil_model, 'unit'):
        fil_model = fil_model.value
    assert ((mod.data - fil_model) <= 0.1 + 1e-3).all()

    # Covering fraction
    cov_frac = test.covering_fraction()
    act_frac = (mod.data - 0.1).sum() / np.sum(mod.data)
    npt.assert_allclose(cov_frac, act_frac, atol=1e-4)

    # Ridge profile along skeleton. Should all equal 1.1
    ridge = fil1.ridge_profile(test.image)
    assert ridge.unit == u.dimensionless_unscaled
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

    # if os.path.exists("test_image_output.fits"):
    #     os.remove("test_image_output.fits")

    fil1.save_fits("test_image_output.fits", test.image, overwrite=True)

    # Test saving additional extensions
    image_dict = {'ext1': test.image.value, 'ext2': test.image.value}
    fil1.save_fits("test_image_output.fits",
                   test.image,
                   image_dict=image_dict,
                   overwrite=True)
    assert len(fits.open("test_image_output.fits")) == 5

    hdu = fits.open("test_image_output.fits")
    skel = fil1.skeleton(pad_size=20)
    npt.assert_allclose(skel, hdu[1].data.astype(bool))

    skel = fil1.skeleton(pad_size=20, out_type='longpath')
    npt.assert_allclose(skel, hdu[2].data.astype(bool))

    mod = fil1.model_image()
    if hasattr(mod, 'unit'):
        mod = mod.value
    npt.assert_allclose(mod, hdu[3].data)

    # os.remove("test_image_output.fits")
    hdu.close()
    del hdu

    image_dict = {'ext1': test.image.value, 'ext2': test.image.value}
    fil1.save_fits("test_image_output.fits",
                   test.image,
                   image_dict=image_dict,
                   overwrite=True)

    hdu = fits.open("test1_stamp_0.fits")
    skel = fil1.skeleton(pad_size=20)
    npt.assert_allclose(skel, hdu[1].data.astype(bool))

    skel = fil1.skeleton(pad_size=20, out_type='longpath')
    npt.assert_allclose(skel, hdu[2].data.astype(bool))

    mod = fil1.model_image()
    if hasattr(mod, 'unit'):
        mod = mod.value
    npt.assert_allclose(mod, hdu[3].data)

    # os.remove("test1_stamp_0.fits")
    hdu.close()
    del hdu

    # Compare saving whole skeleton/mask/model
    # if os.path.exists("test_image_output.fits"):
    #     os.remove("test_image_output.fits")

    test.save_fits(overwrite=True)

    # Test saving additional extensions
    image_dict = {'ext1': test.image.value, 'ext2': test.image.value}
    test.save_fits("test_image_output.fits",
                   test.image,
                   image_dict=image_dict,
                   overwrite=True)
    assert len(fits.open("test_image_output.fits")) == 5

    hdu = fits.open("test1_image_output.fits")

    mod = test.filament_model()
    if hasattr(mod, 'unit'):
        mod = mod.value

    npt.assert_allclose(test.mask, hdu[0].data)
    npt.assert_allclose(test.skeleton, hdu[1].data > 0)
    npt.assert_allclose(test.skeleton_longpath, hdu[2].data > 0)
    npt.assert_allclose(mod, hdu[3].data)

    # os.remove("test1_image_output.fits")
    hdu.close()
    del hdu


def test_simple_filament_noheader_angscale(simple_filament_model):
    '''
    Check the outputs using a simple straight filament with a Gaussian profile.
    No FITS header is given; outputs must have pixel units in all cases.
    '''

    mod = simple_filament_model

    mask = mod.data > 0.5

    ang_scale = np.abs(mod.header['CDELT1']) * u.deg

    test = FilFinder2D(mod.data, ang_scale=ang_scale, mask=mask,
                       save_name='test1')

    test.preprocess_image(flatten_percent=85)

    test.create_mask(use_existing_mask=True)

    test.medskel(verbose=False)

    # Fails without specifying thresholds for skeleton and branch lengths
    with pytest.raises(ValueError) as exc:
        test.analyze_skeletons()
    assert exc.value.args[0] == "Distance not given. Must specify skel_thresh in pixel units."

    test.analyze_skeletons(nthreads=2, skel_thresh=5 * u.pix)

    test.find_widths(nthreads=2, auto_cut=False, max_dist=30 * u.pix)

    test.exec_rht(nthreads=2, branches=False)
    test.exec_rht(nthreads=2, branches=True)

    # Should be oriented along the x-axis. Set to be pi/2.
    npt.assert_allclose(np.pi / 2., test.orientation[0].value)
    npt.assert_allclose(np.pi / 2., test.orientation_branches[0][0].value)

    fil1 = test.filaments[0]

    # Compare lengths
    # Straight skeleton, so length is sum minus 1. Then add the FWHM width on
    # Beam is set to 3 pixels FWHM, so deconvolve before adding
    exp_length = (test.skeleton.sum() - 1) + np.sqrt(10**2 - 3**2) * 2.35

    new_length = test.lengths()[0].value

    # Require the length be within half the beam.
    npt.assert_allclose(exp_length, new_length, atol=1.5)

    # Now compare the widths
    # Expected profile properties
    exp_pars = [1.1, 10.0, 0.1, np.sqrt(10**2 - 3**2) * 2.35]

    new_pars = [par.value for par in fil1.radprof_params] + \
        [fil1.radprof_fwhm()[0].value]
    # The new modeling correctly separates the Gaussian and bkg.
    # Add the bkg to the amplitude
    new_pars[0] += new_pars[2]

    npt.assert_allclose(exp_pars, new_pars, rtol=0.05)

    # Test the non-param fitting in the new code
    test.find_widths(nthreads=2, auto_cut=False, max_dist=30 * u.pix,
                     fit_model='nonparam')

    new_pars = [par.value for par in fil1.radprof_params]

    # There's a larger discrepancy compared with the Gaussian model
    npt.assert_allclose(exp_pars[:-1], new_pars, rtol=0.2)

    # Use the Gaussian fit for the model comparisons below.
    test.find_widths(nthreads=2, auto_cut=False, max_dist=30 * u.pix)

    # Test other output of the new code.

    npt.assert_allclose(1.1, fil1.median_brightness(mod.data))
    npt.assert_allclose(mod.data[(mod.data - 0.1) > 0.5].sum(),
                        fil1.total_intensity().value, rtol=0.01)
    npt.assert_allclose((mod.data - 0.1)[(mod.data - 0.1) > 0.5].sum(),
                        fil1.total_intensity(bkg_subtract=True).value,
                        rtol=0.01)

    fil_model = test.filament_model(bkg_subtract=False)
    # Some astropy 3.0.1 does not support compound model units. Check
    # the output type here.
    if hasattr(fil_model, 'unit'):
        fil_model = fil_model.value

    # Max difference should be where the background isn't covered
    assert ((mod.data - fil_model) <= 0.1 + 1e-7).all()

    # Now compare bkg subtracted versions
    fil_model = test.filament_model(bkg_subtract=True)
    if hasattr(fil_model, 'unit'):
        fil_model = fil_model.value
    assert ((mod.data - fil_model) <= 0.1 + 1e-3).all()

    # Covering fraction
    cov_frac = test.covering_fraction()
    act_frac = (mod.data - 0.1).sum() / np.sum(mod.data)
    npt.assert_allclose(cov_frac, act_frac, atol=1e-4)

    # Ridge profile along skeleton. Should all equal 1.1
    ridge = fil1.ridge_profile(test.image)
    assert ridge.unit == u.dimensionless_unscaled
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
    # if os.path.exists("test_image_output.fits"):
    #     os.remove("test_image_output.fits")

    fil1.save_fits("test_image_output.fits", test.image, overwrite=True)

    # Test saving additional extensions
    image_dict = {'ext1': test.image.value, 'ext2': test.image.value}
    fil1.save_fits("test_image_output.fits",
                   test.image,
                   image_dict=image_dict,
                   overwrite=True)
    assert len(fits.open("test_image_output.fits")) == 5

    hdu = fits.open("test_image_output.fits")
    skel = fil1.skeleton(pad_size=20)
    npt.assert_allclose(skel, hdu[1].data.astype(bool))

    skel = fil1.skeleton(pad_size=20, out_type='longpath')
    npt.assert_allclose(skel, hdu[2].data.astype(bool))

    mod = fil1.model_image()
    if hasattr(mod, 'unit'):
        mod = mod.value
    npt.assert_allclose(mod, hdu[3].data)

    # os.remove("test_image_output.fits")
    hdu.close()
    del hdu

    test.save_stamp_fits(overwrite=True)
    hdu = fits.open("test1_stamp_0.fits")
    skel = fil1.skeleton(pad_size=20)
    npt.assert_allclose(skel, hdu[1].data.astype(bool))

    skel = fil1.skeleton(pad_size=20, out_type='longpath')
    npt.assert_allclose(skel, hdu[2].data.astype(bool))

    mod = fil1.model_image()
    if hasattr(mod, 'unit'):
        mod = mod.value
    npt.assert_allclose(mod, hdu[3].data)

    # os.remove("test1_stamp_0.fits")
    hdu.close()
    del hdu

    # Compare saving whole skeleton/mask/model
    # if os.path.exists("test_image_output.fits"):
    #     os.remove("test_image_output.fits")

    test.save_fits(overwrite=True)
    hdu = fits.open("test1_image_output.fits")

    mod = test.filament_model()
    if hasattr(mod, 'unit'):
        mod = mod.value

    npt.assert_allclose(test.mask, hdu[0].data)
    npt.assert_allclose(test.skeleton, hdu[1].data > 0)
    npt.assert_allclose(test.skeleton_longpath, hdu[2].data > 0)
    npt.assert_allclose(mod, hdu[3].data)

    # os.remove("test1_image_output.fits")
    hdu.close()
    del hdu

def test_simple_filament_nodistance(simple_filament_model):
    '''
    Check the outputs using a simple straight filament with a Gaussian profile.
    No distance given.
    '''

    mod = simple_filament_model

    mask = mod.data > 0.5

    test = FilFinder2D(mod, mask=mask,
                       save_name='test1')

    test.preprocess_image(flatten_percent=85)

    test.create_mask(use_existing_mask=True)
    test.medskel(verbose=False)

    with pytest.raises(ValueError) as exc:
        test.analyze_skeletons()
    assert exc.value.args[0] == "Distance not given. Must specify skel_thresh in pixel units."

    test.analyze_skeletons(nthreads=2, skel_thresh=5 * u.pix)

    test.find_widths(nthreads=2, auto_cut=False, max_dist=30 * u.pix)

    test.exec_rht(nthreads=2, branches=False)
    test.exec_rht(nthreads=2, branches=True)

    # Should be oriented along the x-axis. Set to be pi/2.
    npt.assert_allclose(np.pi / 2., test.orientation[0].value)
    npt.assert_allclose(np.pi / 2., test.orientation_branches[0][0].value)

    fil1 = test.filaments[0]

    # Compare lengths
    # Straight skeleton, so length is sum minus 1. Then add the FWHM width on
    # Beam is set to 3 pixels FWHM, so deconvolve before adding
    exp_length = (test.skeleton.sum() - 1) + np.sqrt(10**2 - 3**2) * 2.35

    new_length = test.lengths()[0].value

    # Require the length be within half the beam.
    npt.assert_allclose(exp_length, new_length, atol=1.5)

    # Now compare the widths
    # Expected profile properties
    exp_pars = [1.1, 10.0, 0.1, np.sqrt(10**2 - 3**2) * 2.35]

    new_pars = [par.value for par in fil1.radprof_params] + \
        [fil1.radprof_fwhm()[0].value]
    # The new modeling correctly separates the Gaussian and bkg.
    # Add the bkg to the amplitude
    new_pars[0] += new_pars[2]

    npt.assert_allclose(exp_pars, new_pars, rtol=0.05)

    # Test the non-param fitting in the new code
    test.find_widths(nthreads=2, auto_cut=False, max_dist=30 * u.pix,
                     fit_model='nonparam')

    new_pars = [par.value for par in fil1.radprof_params]

    # There's a larger discrepancy compared with the Gaussian model
    npt.assert_allclose(exp_pars[:-1], new_pars, rtol=0.2)

    # Use the Gaussian fit for the model comparisons below.
    test.find_widths(nthreads=2, auto_cut=False, max_dist=30 * u.pix)

    # Test other output of the new code.

    npt.assert_allclose(1.1, fil1.median_brightness(mod.data))
    npt.assert_allclose(mod.data[(mod.data - 0.1) > 0.5].sum(),
                        fil1.total_intensity().value, rtol=0.01)
    npt.assert_allclose((mod.data - 0.1)[(mod.data - 0.1) > 0.5].sum(),
                        fil1.total_intensity(bkg_subtract=True).value,
                        rtol=0.01)

    fil_model = test.filament_model(bkg_subtract=False)
    # Some astropy 3.0.1 does not support compound model units. Check
    # the output type here.
    if hasattr(fil_model, 'unit'):
        fil_model = fil_model.value

    # Max difference should be where the background isn't covered
    assert ((mod.data - fil_model) <= 0.1 + 1e-7).all()

    # Now compare bkg subtracted versions
    fil_model = test.filament_model(bkg_subtract=True)
    if hasattr(fil_model, 'unit'):
        fil_model = fil_model.value
    assert ((mod.data - fil_model) <= 0.1 + 1e-3).all()

    # Covering fraction
    cov_frac = test.covering_fraction()
    act_frac = (mod.data - 0.1).sum() / np.sum(mod.data)
    npt.assert_allclose(cov_frac, act_frac, atol=0.01)

    # Ridge profile along skeleton. Should all equal 1.1
    ridge = fil1.ridge_profile(test.image)
    assert ridge.unit == u.K
    npt.assert_allclose(ridge.value, 1.1, atol=0.005)

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

    # if os.path.exists("test_image_output.fits"):
    #     os.remove("test_image_output.fits")

    fil1.save_fits("test_image_output.fits", test.image, overwrite=True)

    hdu = fits.open("test_image_output.fits")
    skel = fil1.skeleton(pad_size=20)
    npt.assert_allclose(skel, hdu[1].data.astype(bool))

    skel = fil1.skeleton(pad_size=20, out_type='longpath')
    npt.assert_allclose(skel, hdu[2].data.astype(bool))

    mod = fil1.model_image()
    if hasattr(mod, 'unit'):
        mod = mod.value
    npt.assert_allclose(mod, hdu[3].data)

    hdu.close()
    del hdu
    os.remove("test_image_output.fits")

    test.save_stamp_fits(overwrite=True)
    hdu = fits.open("test1_stamp_0.fits")
    skel = fil1.skeleton(pad_size=20)
    npt.assert_allclose(skel, hdu[1].data.astype(bool))

    skel = fil1.skeleton(pad_size=20, out_type='longpath')
    npt.assert_allclose(skel, hdu[2].data.astype(bool))

    mod = fil1.model_image()
    if hasattr(mod, 'unit'):
        mod = mod.value
    npt.assert_allclose(mod, hdu[3].data)

    # os.remove("test1_stamp_0.fits")
    hdu.close()
    del hdu

    # Compare saving whole skeleton/mask/model

    test.save_fits(overwrite=True)
    hdu = fits.open("test1_image_output.fits")

    mod = test.filament_model()
    if hasattr(mod, 'unit'):
        mod = mod.value

    npt.assert_allclose(test.mask, hdu[0].data)
    npt.assert_allclose(test.skeleton, hdu[1].data > 0)
    npt.assert_allclose(test.skeleton_longpath, hdu[2].data > 0)
    npt.assert_allclose(mod, hdu[3].data)

    # os.remove("test1_image_output.fits")
    hdu.close()
    del hdu