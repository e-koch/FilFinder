# Licensed under an MIT open source license - see LICENSE

from fil_finder import fil_finder_2D
from astropy.io.fits import getdata

img, hdr = getdata("chamaeleonI-250_normed.fits",
                   header=True)

# Utilize fil_finder_2D class
# See filfind_class.py for inputs
# , region_slice=[580,1620,470,1120])
test = fil_finder_2D(img, hdr, 15.1, 30, 5, 50, distance=170., glob_thresh=20)
test.create_mask(verbose=False, border_masking=True)
test.medskel(verbose=False)
test.analyze_skeletons()
test.compute_filament_brightness()
test.exec_rht(verbose=True, branches=False)
# test.find_widths(verbose=False)

# Or run:
# test.run(verbose=True, save_name="chamaeleonI-250", save_plots=False) ##
# Run entire algorithm
