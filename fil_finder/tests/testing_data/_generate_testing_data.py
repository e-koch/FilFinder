# Licensed under an MIT open source license - see LICENSE

from fil_finder import fil_finder_2D
from astropy.io.fits import getdata

img, hdr = getdata("filaments_updatedhdr.fits", header=True)

# This one uses RHT on branches

test1 = fil_finder_2D(img, hdr, 10.0, flatten_thresh=95, distance=260,
                      size_thresh=430, glob_thresh=20, save_name="test1")

test1.create_mask(border_masking=False)
test1.medskel()
test1.analyze_skeletons()
test1.exec_rht(branches=True)
test1.find_widths()
test1.compute_filament_brightness()
test1.save_table(save_name="test1",
                 branch_table_type='hdf5', table_type='hdf5')
test1.save_fits(save_name="test1")

# This one does not

test2 = fil_finder_2D(img, hdr, 10.0, flatten_thresh=95, distance=260,
                      size_thresh=430, glob_thresh=20, save_name="test2")

test2.create_mask(border_masking=False)
test2.medskel()
test2.analyze_skeletons()
test2.exec_rht(branches=False)
test2.find_widths()
test2.compute_filament_brightness()
test2.save_table(save_name="test2",
                 branch_table_type='hdf5', table_type='hdf5')
test2.save_fits(save_name="test2")
