# Licensed under an MIT open source license - see LICENSE

from fil_finder import fil_finder_2D
from astropy.io.fits import getdata
from astropy import units as u
import os

# Remove the old outputs
dir_path = os.path.dirname(__file__)
test1_path = os.path.join(dir_path, "test1")
for file in os.listdir(test1_path):
    if file.startswith("test1_"):
        os.remove(os.path.join(test1_path, file))

img, hdr = getdata("filaments_updatedhdr.fits", header=True)

# This one uses RHT on branches

test1 = fil_finder_2D(img, header=hdr, beamwidth=10.0 * u.arcsec,
                      flatten_thresh=95,
                      distance=260 * u.pc, size_thresh=430,
                      glob_thresh=20, save_name="test1")

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

test2_path = os.path.join(dir_path, "test2")
for file in os.listdir(test2_path):
    if file.startswith("test2_"):
        os.remove(os.path.join(test2_path, file))

test2 = fil_finder_2D(img, header=hdr, beamwidth=10.0 * u.arcsec,
                      flatten_thresh=95,
                      distance=260 * u.pc, size_thresh=430,
                      glob_thresh=20, save_name="test2")

test2.create_mask(border_masking=False)
test2.medskel()
test2.analyze_skeletons()
test2.exec_rht(branches=False)
test2.find_widths()
test2.compute_filament_brightness()
test2.save_table(save_name="test2",
                 branch_table_type='hdf5', table_type='hdf5')
test2.save_fits(save_name="test2")
