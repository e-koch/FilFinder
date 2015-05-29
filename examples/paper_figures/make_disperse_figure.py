# Licensed under an MIT open source license - see LICENSE

'''
Script to compare fil_finder with Disperse.
Requires read_disperse.py
Recommend running in ipython --pylab mode.
'''

from read_disperse import convert_skeleton
from astropy.io.fits import getdata

# image
img, hdr = getdata("/Volumes/RAIDers_of_the_lost_ark/HGBS/chamaeleonI-350_normed.fits",
                   header=True)


# fil_finder skeletons
fil_skel, skel_hdr = getdata("/Volumes/RAIDers_of_the_lost_ark/HGBS/degrade_all/chamaeleonI-350/chamaeleonI-350_skeletons.fits",
                             header=True)


# If the disperse output isn't in FITS format, create it.
# try:
#     disperse_skel = getdata("chamaeleonI-250_disperseskeleton.fits")
# except IOError:
#     disperse_skel = convert_skeleton("chamaeleonI-250.fits_c36.8.up.NDskl.\
#                                       TRIM.NDskl.S006.BRK.vtk", save=True,
#                                      save_name="chamaeleonI-250")
# del disperse_skel

import aplpy

fig = aplpy.FITSFigure(
    "/Volumes/RAIDers_of_the_lost_ark/HGBS/chamaeleonI-350_normed.fits")

fig.show_grayscale(invert=True, stretch="arcsinh")

fig.tick_labels.set_xformat('hh:mm')
fig.tick_labels.set_yformat('dd:mm')

fig.tick_labels.set_font(size='large', weight='medium', \
                         stretch='normal', family='sans-serif', \
                         style='normal', variant='normal' )

fig.axis_labels.set_font(size='large', weight='medium', \
                         stretch='normal', family='sans-serif', \
                         style='normal', variant='normal')

# fig.add_grid()

fig.show_contour("chamaeleonI-350.fits_c12.up.NDskl.TRIM.S006.BRK.fits",
                 colors="red", linewidths=2, rasterize=True)

fig.show_contour("/Volumes/RAIDers_of_the_lost_ark/HGBS/degrade_all_regrid/chamaeleonI-350/chamaeleonI-350_skeletons.fits",
                 colors="blue", rasterize=True)

fig.show_colorbar()
fig.colorbar.set_label_properties(size='large', weight='medium', \
                         stretch='normal', family='sans-serif', \
                         style='normal', variant='normal')
fig.colorbar.set_axis_label_text('Surface Brightness (MJy/sr)')
fig.colorbar.set_axis_label_font(size='large', weight='medium', \
                         stretch='normal', family='sans-serif', \
                         style='normal', variant='normal')

fig.show_regions("cha_zoomin.reg")

fig.save("chamaeleonI-250_skelcompare.pdf")
fig.save("chamaeleonI-250_skelcompare.png")


