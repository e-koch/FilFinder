from fil_finder import fil_finder_2D
from astropy.io.fits import getdata

img, hdr = getdata("orionB-350.fits",
    header=True)

img += 26.216 #85.452  #879.063

# Utilize fil_finder_2D class
# See filfind_class.py for inputs
test = fil_finder_2D(img, hdr, 15.1, 30, 5, 50, distance=170., glob_thresh=20) #, region_slice=[580,1620,470,1120])
test.create_mask(verbose=True, border_masking=True)
test.medskel(verbose=True)
test.analyze_skeletons()
test.compute_filament_brightness()
test.exec_rht(verbose=False, branches=True)
test.find_widths(verbose=False)

# Or run:
# test.run(verbose=True, save_name="chamaeleonI-250", save_plots=False) ## Run entire algorithm


