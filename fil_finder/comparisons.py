
import numpy as np
from astropy.io import fits
from astropy import convolution
import os
from shutil import move
from datetime import datetime
from pandas import DataFrame
from scipy.ndimage import zoom
import warnings

from fil_finder import fil_finder_2D


class FilFinder_Comparison(object):
    """
    Provides the tools to correctly compare data of regions at different
    distances/resolution. This is necessary due to pixellization effects.
    """
    def __init__(self, fits_dict, distances, beamwidths,
                 regrid=True, convolve=True):

        self.fits_dict = fits_dict
        self.distances = distances
        self.beamwidths = beamwidths
        self.regrid = regrid
        self.convolve = convolve

        if not (self.fits_dict.keys() == self.distances.keys()).all():
            # This should be altered to the proper error class
            raise TypeError("The keys in fits_dict and distances must match.")

        self.headers = {}
        self.ang_res = {}
        self.data = {}

        for key in self.fits_dict.keys():

            self.headers[key] = fits.getheader(self.fits_dict[key])
            self.ang_res[key] = np.abs(self.headers[key]['CDELT2'])

            # Load the maps. Should be fine to keep in memory
            # since images are only 2D.
            self.data[key] = fits.getdata(self.fits_dict[key])

    def convolve_to_common(self):
        '''
        We do 2 checks here:
            1) Convolve to the largest angular resolution.
            2) Convolve the maps such that distances are common.
        '''

        max_dist = self.distances.values().max()
        # max_angres = self.ang_res.max()

        for key in self.data.keys():

            r = max_dist / float(self.distance[key])

            img = self.data[key]
            header = self.header[key]

            # Skip if the data is at the maximum distance
            if r != 1.:
                conv = np.sqrt(r ** 2. - 1) * \
                    (self.beamwidths[key] / np.sqrt(8*np.log(2)) /
                     (np.abs(header["CDELT2"]) * 3600.))  # assume CDELT2 in deg
                # Only convolve if Nyquist sampling can be met
                if conv > 1.5:
                    kernel = convolution.Gaussian2DKernel(conv)
                    good_pixels = np.isfinite(img)
                    nan_pix = np.ones(img.shape)
                    nan_pix[good_pixels == 0] = np.NaN
                    img = convolution.convolve(img, kernel, boundary='fill',
                                               fill_value=np.NaN)
                    # Avoid edge effects from smoothing
                    img = img * nan_pix

                    self.beamwidth[key] *= conv

                    self.data[key] = img

                else:
                    warnings.warn("Nyquist sampling not met for %s."
                                  "No convolution can be performed." % (key))

        return self

    def regrid_to_common(self):
        '''
        Regrid to the shortest distance,
        '''

        min_dist = self.distances.values().min()

        for key in self.data.keys():

            r = float(distance[key]) / min_dist

            img = self.data[key]

            if r != 1:

                good_pixels = np.isfinite(img)
                good_pixels = zoom(good_pixels, round(r, 3),
                                   order=0)

                img[np.isnan(img)] = 0.0
                regrid_conv_img = zoom(img, round(r, 3))


                nan_pix = np.ones(regrid_conv_img.shape)
                nan_pix[good_pixels == 0] = np.NaN


                img = regrid_conv_img * nan_pix

                self.distances = min_dist

                self.headers[key]['CDELT2'] /= r

                self.data[key] = img

        return self

    def compute_FilFinder(self):
        '''
        Run FilFinder
        '''

        return self