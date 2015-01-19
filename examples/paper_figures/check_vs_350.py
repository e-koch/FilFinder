# Licensed under an MIT open source license - see LICENSE

import astropy.wcs as wcs
import astropy.io.fits as fits
import healpy as hp
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import convolve_fft, Gaussian2DKernel
from skimage.morphology import dilation, disk

cmcdir = '/Volumes/RAIDers_of_the_lost_ark/HGBS/' # exclude_from_filpaperanalysis/'
# cmcfile = 'aur_east_scanam_spire350_0_converted.fits'
cmcfile = 'aquilaM2-350.fits'

#cmcdir = '/srv/astro/erickoch/gould_belt/'
#cmcfile = 'chamaeleonI-350.fits'

orig_img = fits.getdata(cmcdir + cmcfile)
hdr = fits.getheader(cmcdir + cmcfile)
# jypb2Mjysr = 1e-6 / \
#     (2 * np.pi * (25.16 * u.arcsec) ** 2 / (8 * np.log(2))).to(u.sr)
# orig_img = orig_img * jypb2Mjysr

# print np.nanmean(orig_img.value)
# print np.nanmean(orig_img)
# print blah

width = 5 / 60. / hdr['CDELT2'] / np.sqrt(8 * np.log(2))

kernel = Gaussian2DKernel(width)
img = convolve_fft(orig_img, kernel, min_wt=4.0,
                   interpolate_nan=False, normalize_kernel=True)
badmask = np.isnan(orig_img)
struct = disk(width * 4)
badmask = dilation(badmask, struct)
img[np.where(badmask)] = np.nan

w = wcs.WCS(hdr)
x, y = np.meshgrid(np.arange(img.shape[1]),
                   np.arange(img.shape[0]),
                   indexing='xy')

# c = wcs.utils.pixel_to_skycoord(x, y, w, 0)
# Bleeding-edge astropy isn't playing nice. Here's a work-around for now.
l, b = w.all_pix2world(x, y, 0)
c = SkyCoord(l, b, frame='fk5', unit=(u.deg, u.deg))
l = c.galactic.l.radian
b = c.galactic.b.radian

planckfile = 'HFI_SkyMap_857_2048_R1.10_nominal.fits'
planckdata = hp.read_map(planckfile)
#idx = hp.ang2pix(2048,l,b)
planckmap = (hp.pixelfunc.get_interp_val(
    planckdata, np.pi / 2 - b, l, nest=False))
#planckidx = hp.pixelfunc.ang2pix(2048,b,l)


good = np.where(np.isfinite(planckmap) * np.isfinite(img))
m, b = np.polyfit(planckmap[good], img[good], 1)
plt.hexbin(planckmap.ravel(), img.ravel()-b , cmap='gray_r',
           bins='log', gridsize=[100, 100])
print('Offset is {0} MJy/sr'.format(b))
plt.plot(np.linspace(0, 1000, 20), np.linspace(0, 1000, 20))
plt.xlabel(r'Planck 350 $\mu$m')
plt.ylabel(r'Herschel 350 $\mu$m')
plt.xlim([0, np.max(planckmap[good])])
plt.ylim([0, np.max(img[good])])
plt.show()
# plt.savefig('planck_vs_aur.png')
plt.close()
