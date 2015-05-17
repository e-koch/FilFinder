
import numpy as np
from astropy.io import fits
from astropy import convolution
import os
from shutil import move
from datetime import datetime
from pandas import DataFrame
from scipy.ndimage import zoom

from fil_finder import fil_finder_2D


class FilFinder_Comparison(object):
    """
    Provides the tools to correctly compare data of regions at different
    distances/resolution. This is necessary due to pixellization effects.
    """
    def __init__(self, fits_dict, distances, regrid=True, convolve=True):

        self.fits_dict = fits_dict
        self.distances = distances
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

        return self

    def regrid_to_common(self):
        '''
        Regrid to the shortest distance,
        '''

        return self

    def compute_FilFinder(self):
        '''
        Run FilFinder
        '''

        return self