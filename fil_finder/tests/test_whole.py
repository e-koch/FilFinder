# Licensed under an MIT open source license - see LICENSE

import numpy as np
import numpy.testing as npt
import astropy.units as u
from copy import deepcopy
import warnings

from .. import FilFinder2D

from ._testing_data import *


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

    # Now the skeletons
    test1.medskel()

    # Don't use iterative pruning in order to match old version.
    test1.analyze_skeletons(skel_thresh=40 * u.pix, branch_thresh=2.0 * u.pix,
                            max_prune_iter=1)

    test1.exec_rht(branches=False)

    test1.exec_rht(branches=True)


    # How the radial profiles are built in the new and old versions differs
    # quite a bit (this is one of the biggest reasons for the new version).
    # The fit parameters will NOT be the same because the old version is
    # not correctly building portions of the profiles! Check that that
    # the warning is raised
    test1.find_widths(max_dist=0.3 * u.pc, try_nonparam=True, auto_cut=False,
                      pad_to_distance=0 * u.pix)

    med_bright = test1.median_brightness()

    # Compute model image
    fil_model = test1.filament_model(bkg_subtract=True)

    # Same for the total intensities, but run it to make sure it works.
    total_intensity = test1.total_intensity()

    cov_frac = test1.covering_fraction()
    npt.assert_allclose(0.564, cov_frac, atol=0.001)

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
