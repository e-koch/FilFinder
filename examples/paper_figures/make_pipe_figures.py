# Licensed under an MIT open source license - see LICENSE

'''
Make the pipe comparison plot for paper.
'''

import numpy as np
import matplotlib.pyplot as p
from astropy.io import fits as fits
from astropy.table import Table


# pipe_norm = Table.read("pipeCenterB59-250/pipeCenterB59-250_table.csv")
# pipe_deg = Table.read("degraded_pipe_table.csv")

from fil_finder import fil_finder_2D
from astropy import convolution

img, hdr = fits.getdata('pipeCenterB59-250.fits', header=True)

filfind = fil_finder_2D(img, hdr, 18.5, 30, 15, 30, glob_thresh=20,
                        distance=145.)
filfind.create_mask(size_thresh=400)
filfind.medskel()
filfind.analyze_skeletons()
filfind.exec_rht()
filfind.find_widths()

r = 460. / 145.
conv = np.sqrt(r ** 2. - 1) * \
    (18.5 / np.sqrt(8*np.log(2)) / (np.abs(hdr["CDELT2"]) * 3600.))

kernel = convolution.Gaussian2DKernel(conv)
good_pixels = np.isfinite(img)
nan_pix = np.ones(img.shape)
nan_pix[good_pixels == 0] = np.NaN
conv_img = convolution.convolve(img, kernel, boundary='fill',
                                fill_value=np.NaN)
# Avoid edge effects from smoothing
conv_img = conv_img * nan_pix

filfind2 = fil_finder_2D(conv_img, hdr, 18.5, 30, 15, 30, glob_thresh=20,
                         distance=145.)
filfind2.create_mask(size_thresh=400)
filfind2.medskel()
filfind2.analyze_skeletons()
filfind2.exec_rht()
filfind2.find_widths()


# p.imshow(filfind.flat_img, interpolation='nearest', cmap='binary',
#          origin='lower')
# p.contour(filfind.skeleton, colors='g', label="Normal", linewidths=6)
# p.contour(filfind2.skeleton, colors='b', label='Degraded')
# p.plot(None, None, label='Normal', color='g', linewidth=6)
# p.plot(None, None, label='Degraded', color='b')
# p.legend(prop={'size': 20})
# p.xticks([])
# p.yticks([])
# p.show()

# Histograms plot

fig, (ax1, ax2, ax3, ax4) = p.subplots(4, 1)

fig.set_figheight(12)
fig.set_figwidth(4)

# FWHM
# p.subplot(411)

norm_fwhm = filfind.width_fits["Parameters"][:, -1]
deg_fwhm = filfind2.width_fits["Parameters"][:, -1]

ax2.hist(deg_fwhm[np.isfinite(deg_fwhm)], bins=7,
       color="b", alpha=0.5, label="Degraded")
ax2.hist(norm_fwhm[np.isfinite(norm_fwhm)], bins=7,
       color="g", alpha=0.5, label="Normal")
ax2.set_xlabel("Width (pc)")
# # Length

# p.subplot(412)

norm_length = filfind.lengths
deg_length = filfind2.lengths

ax1.hist(deg_length[np.isfinite(deg_fwhm)], bins=7,
       color="b", alpha=0.5, label="Degraded")
ax1.hist(norm_length[np.isfinite(norm_fwhm)], bins=7,
       color="g", alpha=0.5, label="Normal")
ax1.set_xlabel("Lengths (pc)")
ax1.legend()

# p.subplot(413)

norm_orient = np.asarray(filfind.rht_curvature['Median'])
deg_orient = np.asarray(filfind2.rht_curvature['Median'])

ax3.hist(deg_orient[np.isfinite(deg_fwhm)], bins=7,
       color="b", alpha=0.5, label="Degraded")
ax3.hist(norm_orient[np.isfinite(norm_fwhm)], bins=7,
       color="g", alpha=0.5, label="Normal")
ax3.set_xlim([-np.pi/2, np.pi/2])

ax3.set_xlabel("Orientation")

# p.subplot(414)

norm_curv = np.asarray(filfind.rht_curvature['IQR'])
deg_curv = np.asarray(filfind2.rht_curvature['IQR'])

ax4.hist(deg_curv[np.isfinite(deg_fwhm)], bins=7,
       color="b", alpha=0.5, label="Degraded")
ax4.hist(norm_curv[np.isfinite(norm_fwhm)], bins=7,
       color="g", alpha=0.5, label="Normal")
ax4.set_xlim([0.4, 1.3])
ax4.set_xlabel("Curvature")

p.tight_layout(h_pad=0.1)
p.show()
