#!/usr/bin/python

'''

Filament class is called to run the fil-finder algorithm

'''

from cores import *
from curvature import *
from length import *
from pixel_ident import *
from utilities import *
from width import *

import numpy as np
import matplotlib.pyplot as p

class fil_finder_2D(object):
    """

    CLASS
    --------------
    Runs the fil-finder algorithm for 2D images

    INPUTS
    -------------
    image - 2D array of image
    hdr   - header from fits file    COULD MAKE THIS OPTIONAL
    beamwidth - fwhm beamwidth (arcseconds) of device used to take data
    glob_thresh - the percentile to cut off search for filamentary structure
    local_thresh - the pixel size of the patch used in the adaptive threshold
    skel_thresh - below this cut off, skeletons with less pixels will be deleted
    branch_thresh - branches shorter than this length (pixels) will be deleted if extraneous
    pad_size - size to which filaments will be padded to build a radial intensity profile
    distance - to object in image
    region_slice - option to slice off regions of the given image -- input as [xmin,xmax,ymin,max]

    """
    def __init__(self, image, hdr, beamwidth, glob_thresh, local_thresh, \
                skel_thresh, branch_thresh, pad_size, distance=None, region_slice=None):
        ## Consider just passing in the filament detection as a fcn or
        ## premade to be skeletonized

        img_dim = len(image.shape)
        if img_dim<2 or img_dim>2:
            raise TypeError("Image must be 2D array. Input was %s dimensions")  % (img_dim)
        if region_slice==None:
            self.image = image
        else:
            slices = (slice(region_slice[0],region_slice[1],None), \
                        slice(region_slice[2],region_slice[3],None))
            self.image = np.pad(image[slices],1,padwithzeros)

        # self.image = np.arctan(self.image)/np.mean(self.image[~np.isnan(self.image)])  ## Rescaling idea -- incomplete
        self.skel_thresh = skel_thresh
        self.branch_thresh = branch_thresh
        self.pad_size = pad_size

        try:
            self.imagefreq = (3*10**14)/hdr["WAVE"]
        except:
            user_freq = raw_input("No wavelength in header. Input frequency now or pass: ")
            if user_freq=="pass" or user_freq=="":
                self.imagefreq = None ## Set a time limit here
            else:
                self.imagefreq = float(user_freq) ## This may fail...

        if distance==None:
            print "No distance given. Results will be in pixel units."
            self.imgscale = 1 ## pixel
            self.beamwidth = beamwidth * (hdr["CDELT2"] * 3600)**(-1) ## where CDELT2 is in degrees

        else:
            self.imgscale = (hdr['CDELT2']*(np.pi/180.0)*distance) ## pc
            self.beamwidth = (beamwidth/np.sqrt(8*np.log(2.))) * (2*np.pi / 206265.) * distance
            # FWHM beamwidth in pc

        self.glob_thresh = glob_thresh
        self.local_thresh = local_thresh
        self.mask = None
        self.lengths = None
        self.widths = None
        self.width_fits = None
        self.curvature = None
        self.filament_arrays = None
        self.labelled_filament_arrays = None
        self.number_of_filaments = None
        self.array_offsets = None
        self.skeleton = None
        self.filament_extents = None
        self.branch_info = None
        self.masked_image = None
        self.medial_axis_distance = None


    def create_mask(self, glob_thresh = None, local_thresh = None, verbose = False): ## Give option to give live inputs to change thresh??
        '''
            Use function makefilamentsappear to create a mask of filaments
        '''


        if glob_thresh is not None:
            self.glob_thresh = glob_thresh
        if local_thresh is not None:
            self.local_thresh = local_thresh

        from scipy import ndimage

        # self.image = ndimage.gaussian_filter(self.image,sigma=2)
        self.mask = makefilamentsappear(self.image,self.glob_thresh,self.local_thresh)


        if verbose:
            scale = 0
            vmax = np.nanmax(self.image)
            while scale==0:
                p.contour(self.mask)
                p.imshow(self.image, vmax=vmax,interpolation=None,origin="lower")
                p.show()

                print "Mean and median value of image are (%s,%s), vmax currently set to %s" \
                    % (np.mean(self.image[~np.isnan(self.image)]),np.median(self.image[~np.isnan(self.image)]),vmax)
                rescale = raw_input("Rescale image? Enter new vmax or no: ")
                if rescale=="no" or rescale=="n":
                    scale = 1
                else:
                    vmax = float(rescale)

        return self

    def medskel(self,glob_thresh=None, local_thresh=None, return_distance=True, verbose = False):

        if return_distance:
            self.skeleton,self.medial_axis_distance = medial_axis(self.mask, return_distance=return_distance)
        else:
            self.skeleton = medial_axis(self.mask)


        self.masked_image = self.image * self.mask

        if verbose: # For examining results of skeleton
            skel_points = np.where(self.skeleton==1)
            for i in range(len(skel_points[0])):
                self.masked_image[skel_points[0][i],skel_points[1][i]] = np.NaN
            p.imshow(self.masked_image,interpolation=None,origin="lower")
            p.show()

        return self

    def analyze_skeletons(self,verbose = False):
        isolated_filaments, new_mask, num, offsets = \
                isolatefilaments(self.skeleton,self.mask,self.skel_thresh)
        self.filament_arrays = isolated_filaments
        self.mask = new_mask
        self.number_of_filaments = num
        self.array_offsets = offsets

        interpts, hubs, ends, filbranches, labelled_fil_arrays =  \
                pix_identify(isolated_filaments, num)

        initial_lengths, filament_pixels = init_lengths(labelled_fil_arrays, filbranches)

        end_nodes, inter_nodes, edge_list, nodes = \
            pre_graph(labelled_fil_arrays, initial_lengths, interpts, ends)

        max_path, extremum = longest_path(edge_list, nodes, initial_lengths, verbose = verbose)

        self.filament_extents = extremum_pts(labelled_fil_arrays, extremum, ends)

        main_lengths, branch_lengths, labelled_fil_arrays, curvature = \
            final_lengths(self.image, max_path, edge_list, labelled_fil_arrays, filament_pixels, interpts, filbranches, \
                            initial_lengths, self.imgscale, self.branch_thresh)

        labelled_fil_arrays, filbranches, final_hubs, branch_lengths = final_analysis(labelled_fil_arrays)

        self.lengths = main_lengths
        self.labelled_filament_arrays = labelled_fil_arrays
        self.branch_info = {"filament_branches":filbranches, "branch_lengths":branch_lengths}
        self.curvature = curvature

        return self

    def find_widths(self, verbose = False):

        dist_transform_all, dist_transform_separate = dist_transform(self.labelled_filament_arrays, \
                    self.array_offsets, self.image.shape, self.pad_size)

        widths, fit_params, fit_errors = gauss_width(self.image, dist_transform_all, dist_transform_separate, \
                                                self.beamwidth, self.imgscale, self.array_offsets, verbose=verbose)

        self.widths = widths
        self.width_fits = {"Parameters":fit_params, "Errors":fit_errors}

        ## Implement check for failed fits and replace with average width from medial_axis_distance

        return self

    def results(self):
        '''
        Separating out failed fits
        '''
        overall_lengths = []
        overall_widths = []
        for i in range(self.number_of_filaments):
            if isinstance(self.widths[i],float):
                if self.lengths[i]>self.widths[i]:
                    overall_lengths.append(self.lengths[i] + self.widths[i])
                    overall_widths.append(self.widths[i])
                else:
                    overall_lengths.append(self.lengths[i])
                    overall_widths.append("Fit Fail")
            else:
                overall_lengths.append(self.lengths[i])
                overall_widths.append(self.widths[i])

        self.lengths = overall_lengths
        self.widths = overall_widths

        return self

    def __str__(self):
            print("%s filaments found.") % (self.number_of_filaments)
            for fil in range(self.number_of_filaments):
                print "Filament: %s, Width: %s, Length: %s, Curvature: %s" % \
                        (fil,self.widths[fil],self.lengths[fil], self.curvature[fil])

    def run(self, verbose = False):
        self.create_mask(verbose = verbose)
        self.medskel(verbose = verbose)
        self.analyze_skeletons(verbose = verbose)
        self.find_widths(verbose = verbose)
        self.results()




if __name__ == "__main__":
    import sys
    fib(int(sys.argv[1]))