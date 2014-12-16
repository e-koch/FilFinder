
from fil_finder import fil_finder_2D
from astropy.io.fits import getdata
import matplotlib.pyplot as p

img, hdr = getdata("filaments_updatedhdr.fits", header=True)

# Add some noise
import numpy as np
np.random.seed(500)

noises = [0.025, 0.05, 0.1]
patches = [7, 13, 26]

frac_labels = ['Half', 'Optimal', 'Twice']

figure, grid = p.subplots(3, 3, sharex=True, sharey=True, figsize=(12, 12))

for i, patch in enumerate(patches):
    for j, noise in enumerate(noises):
        noiseimg = img + np.random.normal(0, noise, size=img.shape)

        test = fil_finder_2D(noiseimg, hdr, 0.0, 30, 5, 10, flatten_thresh=98,
                             distance=260)
        test.create_mask(verbose=False, smooth_size=3.0, adapt_thresh=13.0,
                         size_thresh=180, regrid=False, border_masking=False,
                         zero_border=True)

        grid[i, j].imshow(test.mask[10:266, 10:266],
                          cmap='binary', interpolation='nearest')
        grid[i, j].set_xlim([0, 256])
        grid[i, j].set_ylim([0, 256])

        if i == 2:
            grid[i, j].set_xlabel(frac_labels[j], fontsize=14)
        if j == 0:
            grid[i, j].set_ylabel(frac_labels[i], fontsize=14)

        p.setp(grid[i, j].get_xticklabels(), visible=False)
        p.setp(grid[i, j].get_yticklabels(), visible=False)

figure.text(0.5, 0.04, 'Noise', ha='center', va='center')
figure.text(0.04, 0.5, 'Patch Size', ha='center', va='center',
            rotation='vertical')

figure.tight_layout(w_pad=0.5, h_pad=0.5)
p.subplots_adjust(bottom=0.08, left=.08, right=.92, top=.92)
p.show()
