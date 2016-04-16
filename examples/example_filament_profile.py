# Licensed under an MIT open source license - see LICENSE

from fil_finder import fil_finder_2D
from fil_finder.width_profiles import filament_profile
from fil_finder.utilities import eight_con
from astropy.io import fits
import astropy.units as u
import numpy as np
import scipy.ndimage as nd

import matplotlib.pyplot as p

'''
Example of the new radial profiling code for FilFinder.
This functionality is still in testing!
'''

hdu = fits.open("filaments_updatedhdr.fits")[0]
img, hdr = hdu.data, hdu.header

# Add some noise
np.random.seed(500)
noiseimg = img + np.random.normal(0, 0.05, size=img.shape)

# We need the finalized skeletons, so run the first few portions of the
# normal algorithm.

test = fil_finder_2D(noiseimg, header=hdr, beamwidth=10.0*u.arcsec,
                     flatten_thresh=95, distance=260*u.pc,
                     glob_thresh=20)
test.create_mask(verbose=False, border_masking=False, size_thresh=430)
test.medskel(verbose=False)
test.analyze_skeletons(verbose=False)

# Now choose one the longest path skeletons from the labeled array
labels, num = nd.label(test.skeleton_longpath, eight_con())

# Number 3 isn't too long and is in a relatively uncrowded region.
# We enable verbose, which will plot the profile normal to each pixel in the
# skeleton.
# The noise parameter allows an array, of the same shape as the image, to be
# passed and used as weights in the fitting.

# NOTE: If you don't want a million plots showing up, uncomment the next line:
# p.ion()
dists, profiles, extents, fit_table = \
    filament_profile(labels == 3, noiseimg, hdr, max_dist=0.14*u.pc,
                     distance=250.*u.pc, bright_unit="", noise=None,
                     verbose=True)

# The fits aren't perfect (fitting is limited to a fairly naive gaussian right
# now), but the evolution in the profile shape is quite evident!
