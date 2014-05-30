
from fil_finder import fil_finder_2D
from astropy.io.fits import getdata

img, hdr = getdata("examples/filaments_updatedhdr.fits", header=True)

# Add some noise
import numpy as np
np.random.seed(500)
noiseimg = img + np.random.normal(0,0.05,size=img.shape)

## Utilize fil_finder_2D class
## See filfind_class.py for inputs
test = fil_finder_2D(noiseimg, hdr, 0.0, 30, 5, 10, 95, distance=260)
test.create_mask(verbose=True)
test.medskel(verbose=True)
test.analyze_skeletons(verbose=True)
test.exec_rht(verbose=True)
test.find_widths(verbose=True)
# test.save_table(save_name="sim_filaments")
# test.save_fits(save_name="sim_filaments")
