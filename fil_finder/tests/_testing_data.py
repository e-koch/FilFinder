
'''
Load in the testing data
'''

import os
from astropy.io import fits
from astropy.table import Table

dir_path = os.path.dirname(__file__)

path1 = os.path.join(dir_path, "testing_data/test1")
path2 = os.path.join(dir_path, "testing_data/test2")

img_path = os.path.join(dir_path, "testing_data")

# Load in the fits file
img, hdr = \
    fits.getdata(os.path.join(img_path, "filaments_updatedhdr.fits"),
                 header=True)

# Load in each dataset

model1 = fits.getdata(os.path.join(path1, "test1_filament_model.fits"))
mask1 = fits.getdata(os.path.join(path1, "test1_mask.fits"))
skeletons1 = \
    fits.getdata(os.path.join(path1, "test1_skeletons.fits"))


model2 = fits.getdata(os.path.join(path2, "test2_filament_model.fits"))
mask2 = fits.getdata(os.path.join(path2, "test2_mask.fits"))
skeletons2 = \
    fits.getdata(os.path.join(path2, "test2_skeletons.fits"))


table1 = Table.read(os.path.join(path1, "test1_table.hdf5"), path="data")
table2 = Table.read(os.path.join(path2, "test2_table.hdf5"), path="data")

branch_tables1 = \
    [Table.read(os.path.join(path1, "test1_table_branch.hdf5"), path="branch_"+str(i))
     for i in range(7)]

branch_tables2 = \
    [Table.read(os.path.join(path2, "test2_table_branch.hdf5"), path="branch_"+str(i))
     for i in range(7)]

