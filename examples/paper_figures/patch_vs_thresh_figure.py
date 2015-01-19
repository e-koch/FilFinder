# Licensed under an MIT open source license - see LICENSE

from fil_finder import fil_finder_2D
from astropy.io.fits import getdata
import matplotlib.pyplot as p

img, hdr = getdata("filaments_updatedhdr.fits", header=True)

# Add some noise
import numpy as np
np.random.seed(500)

threshs = [75, 95, 99]
patches = [7, 13, 26]


figure, grid = p.subplots(3, 3, sharex=True, sharey=True, figsize=(12, 12))

for i, patch in enumerate(patches[::-1]):
    for j, thresh in enumerate(threshs):
        noiseimg = img + np.random.normal(0, 0.05, size=img.shape)

        test = fil_finder_2D(noiseimg, hdr, 0.0, 30, 5, 10,
                             flatten_thresh=thresh,
                             distance=260)
        test.create_mask(verbose=False, smooth_size=3.0, adapt_thresh=patch,
                         size_thresh=180, regrid=False, border_masking=False,
                         zero_border=True)
        test.medskel(verbose=False)
        test.analyze_skeletons(verbose=False)

        grid[i, j].imshow(test.mask[10:266, 10:266],
                          cmap='binary', interpolation='nearest')
        grid[i, j].contour(test.skeleton[10:266, 10:266], colors='gray',
                           linewidths=0.3)
        grid[i, j].set_xlim([0, 256])
        grid[i, j].set_ylim([0, 256])

        if i == 2:
            grid[i, j].set_xlabel(thresh, fontsize=14)
        if j == 0:
            grid[i, j].set_ylabel(patch, fontsize=14)

        p.setp(grid[i, j].get_xticklabels(), visible=False)
        p.setp(grid[i, j].get_yticklabels(), visible=False)

figure.text(0.5, 0.04, 'Flatten Percentile', ha='center', va='center')
figure.text(0.04, 0.5, 'Patch Size (pixels)', ha='center', va='center',
            rotation='vertical')

figure.tight_layout(w_pad=0.5, h_pad=0.5)
p.subplots_adjust(bottom=0.08, left=.08, right=.92, top=.92)
p.show()
