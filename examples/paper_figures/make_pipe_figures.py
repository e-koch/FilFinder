# Licensed under an MIT open source license - see LICENSE

'''
Make the pipe comparison plot for paper.
'''

import numpy as np
import matplotlib.pyplot as p
from astropy.io import fits as fits
from astropy.table import Table
from scipy.ndimage import zoom
from matplotlib.collections import Collection
from matplotlib.artist import allow_rasterization


# pipe_norm = Table.read("pipeCenterB59-250/pipeCenterB59-250_table.csv")
# pipe_deg = Table.read("degraded_pipe_table.csv")

from fil_finder import fil_finder_2D
from astropy import convolution


class ListCollection(Collection):
    def __init__(self, collections, **kwargs):
        Collection.__init__(self, **kwargs)
        self.set_collections(collections)

    def set_collections(self, collections):
        self._collections = collections

    def get_collections(self):
        return self._collections

    @allow_rasterization
    def draw(self, renderer):
        for _c in self._collections:
            _c.draw(renderer)


def insert_rasterized_contour_plot(c, ax):
    collections = c.collections
    for _c in collections:
        _c.remove()
    cc = ListCollection(collections, rasterized=True)
    ax.add_artist(cc)
    return cc


img, hdr = fits.getdata('pipeCenterB59-350.fits', header=True)
beam = 24.9
img = img + 31.697

filfind = fil_finder_2D(img, hdr, beam, glob_thresh=20,
                        distance=145.)
filfind.create_mask()#size_thresh=400)
filfind.medskel()
filfind.analyze_skeletons()
filfind.exec_rht()
filfind.find_widths(verbose=False)

r = 460. / 145.
conv = np.sqrt(r ** 2. - 1) * \
    (beam / np.sqrt(8*np.log(2)) / (np.abs(hdr["CDELT2"]) * 3600.))

kernel = convolution.Gaussian2DKernel(conv)
good_pixels = np.isfinite(img)
nan_pix = np.ones(img.shape)
nan_pix[good_pixels == 0] = np.NaN
conv_img = convolution.convolve(img, kernel, boundary='fill',
                                fill_value=np.NaN)

# Avoid edge effects from smoothing
conv_img = conv_img * nan_pix

filfind2 = fil_finder_2D(conv_img, hdr, conv*beam, glob_thresh=20,
                         distance=145.)
filfind2.create_mask()
filfind2.medskel()
filfind2.analyze_skeletons()
filfind2.exec_rht()
filfind2.find_widths(verbose=False)

# Regrid to same physical scale

good_pixels = np.isfinite(img)
good_pixels = zoom(good_pixels, 1/r, order=0)

conv_img[np.isnan(conv_img)] = 0.0
regrid_conv_img = zoom(conv_img, 1/r)

regrid_conv_img = zoom(regrid_conv_img, r)

# nan_pix = np.ones(regrid_conv_img.shape)
# nan_pix[good_pixels == 0] = np.NaN

regrid_conv_img = regrid_conv_img[:-1, :-1] * nan_pix
# regrid_conv_img = regrid_conv_img * nan_pix

filfind3 = fil_finder_2D(regrid_conv_img, hdr, conv*beam, glob_thresh=20,
                         distance=145.)
filfind3.create_mask()
filfind3.medskel()
filfind3.analyze_skeletons()
filfind3.exec_rht()
filfind3.find_widths(verbose=False)

# Show flattened image with contour.

fig, ax = p.subplots(1)

ax.imshow(filfind.flat_img, interpolation='nearest', cmap='binary',
          origin='lower', vmax=1.0)
norm = ax.contour(filfind.skeleton, colors='g', label="Normal", linewidths=3,
                  linestyles='-')
conv = ax.contour(filfind2.skeleton, colors='b', label='Convolved',
                  linewidths=1.5, linestyles='--')
reg = ax.contour(filfind3.skeleton, colors='r', label='Regridded',
                 linestyles=':')
insert_rasterized_contour_plot(norm, ax)
insert_rasterized_contour_plot(conv, ax)
insert_rasterized_contour_plot(reg, ax)
ax.plot(None, None, label='Normal', color='g', linewidth=6, linestyle="-")
ax.plot(None, None, label='Convolved', color='b', linestyle="--")
ax.plot(None, None, label='Regridded', color='r', linestyle=":")
ax.legend(loc=2, prop={'size': 20})
ax.set_xticks([])
ax.set_yticks([])
fig.show()

raw_input("BLARG: ")

# Histograms plot

fig, (ax1, ax2, ax3, ax4) = p.subplots(4, 1)

fig.set_figheight(12)
fig.set_figwidth(4)

# FWHM
norm_fwhm = filfind.width_fits["Parameters"][:, -1]
deg_fwhm = filfind2.width_fits["Parameters"][:, -1]
reg_fwhm = filfind3.width_fits["Parameters"][:, -1]

w_max = np.max([np.nanmax(norm_fwhm), np.nanmax(deg_fwhm), np.nanmax(reg_fwhm)])
w_min = np.min([np.nanmin(norm_fwhm), np.nanmin(deg_fwhm), np.nanmin(reg_fwhm)])
w_bins = np.linspace(w_min, w_max, 7)
w_bins = np.insert(w_bins, 1, 0.01)

ax2.hist(norm_fwhm[np.isfinite(norm_fwhm)], bins=w_bins,
       color="g", label="Normal", histtype='step', linewidth=3,
       linestyle='solid')
ax2.hist(deg_fwhm[np.isfinite(deg_fwhm)], bins=w_bins,
       color="b", label="Convolved", histtype='step', linewidth=3,
       linestyle='dashed')
ax2.hist(reg_fwhm[np.isfinite(reg_fwhm)], bins=w_bins,
       color="r", label="Regridded", histtype='step', linewidth=3,
       linestyle='dotted')
ax2.set_xlabel("Width (pc)")

# Length

norm_length = filfind.lengths
deg_length = filfind2.lengths
reg_length = filfind3.lengths

l_max = np.max([np.nanmax(norm_length), np.nanmax(deg_length), np.nanmax(reg_length)])
l_min = np.min([np.nanmin(norm_length), np.nanmin(deg_length), np.nanmin(reg_length)])
l_bins = np.linspace(l_min, l_max, 7)

ax1.hist(norm_length[np.isfinite(norm_fwhm)], bins=l_bins,
       color="g", label="Normal", histtype='step', linewidth=3,
       linestyle='solid')
ax1.hist(deg_length[np.isfinite(deg_fwhm)], bins=l_bins,
       color="b", label="Convolved", histtype='step', linewidth=3,
       linestyle='dashed')
ax1.hist(reg_length[np.isfinite(reg_fwhm)], bins=l_bins,
       color="r", label="Regridded", histtype='step', linewidth=3,
       linestyle='dotted')
ax1.set_xlabel("Lengths (pc)")
ax1.legend()

# Orientation

norm_orient = np.asarray(filfind.rht_curvature['Median'])
deg_orient = np.asarray(filfind2.rht_curvature['Median'])
reg_orient = np.asarray(filfind3.rht_curvature['Median'])

o_bins = np.linspace(-np.pi/2, np.pi/2, 7)

ax3.hist(deg_orient[np.isfinite(deg_fwhm)], bins=o_bins,
       color="b", label="Convolved", histtype='step', linewidth=3,
       linestyle='dashed')
ax3.hist(norm_orient[np.isfinite(norm_fwhm)], bins=o_bins,
       color="g", label="Normal", histtype='step', linewidth=3,
       linestyle='solid')
ax3.hist(reg_orient[np.isfinite(reg_fwhm)], bins=o_bins,
       color="r", label="Regridded", histtype='step', linewidth=3,
       linestyle='dotted')
ax3.set_xlim([-np.pi/2, np.pi/2])

ax3.set_xlabel("Orientation")

norm_curv = np.asarray(filfind.rht_curvature['IQR'])
deg_curv = np.asarray(filfind2.rht_curvature['IQR'])
reg_curv = np.asarray(filfind3.rht_curvature['IQR'])

curv_max = np.max([np.nanmax(norm_curv), np.nanmax(deg_curv), np.nanmax(reg_curv)])
curv_min = np.min([np.nanmin(norm_curv), np.nanmin(deg_curv), np.nanmin(reg_curv)])
curv_bins = np.linspace(curv_min, curv_max, 7)

ax4.hist(deg_curv[np.isfinite(deg_fwhm)], bins=curv_bins,
       color="b", label="Convolved", histtype='step', linewidth=3,
       linestyle='dashed')
ax4.hist(norm_curv[np.isfinite(norm_fwhm)], bins=curv_bins,
       color="g", label="Normal", histtype='step', linewidth=3,
       linestyle='solid')
ax4.hist(reg_curv[np.isfinite(reg_fwhm)], bins=curv_bins,
       color="r", label="Regridded", histtype='step', linewidth=3,
       linestyle='dotted')
# ax4.set_xlim([0.4, 1.3])
ax4.set_xlabel("Curvature")

p.tight_layout(h_pad=0.1)
p.show()
