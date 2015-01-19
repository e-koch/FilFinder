
from fil_finder import fil_finder_2D
from astropy.io.fits import getdata

img, hdr = getdata("filaments_updatedhdr.fits", header=True)

# Add some noise
import numpy as np
np.random.seed(500)
noiseimg = img + np.random.normal(0,0.05,size=img.shape)

## Utilize fil_finder_2D class
## See filfind_class.py for inputs
test = fil_finder_2D(noiseimg, hdr, 0.0, 30, 5, 10, flatten_thresh=95, distance=260)
test.create_mask(verbose=True, smooth_size=3.0, adapt_thresh=13.0,
                 size_thresh=180, regrid=False, border_masking=False,
                 zero_border=True)
test.medskel(verbose=False)
test.analyze_skeletons(verbose=False)
test.exec_rht(verbose=False)
test.find_widths(verbose=False)
# test.save_table(save_name="sim_filaments")
# test.save_fits(save_name="sim_filaments")
