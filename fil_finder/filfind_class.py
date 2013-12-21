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
import scipy.ndimage as nd
from skimage.filter import threshold_adaptive
from skimage.morphology import remove_small_objects, medial_axis
from scipy.stats import scoreatpercentile

class fil_finder_2D(object):
    """


    Runs the fil-finder algorithm for 2D images

    INPUTS
    -------------
    image - array
            2D array of image
    hdr   - dictionary
            header from fits file
    beamwidth - float
                fwhm beamwidth (arcseconds) of device used to take data
    glob_thresh - float
                  the percentile to cut off search for filamentary structure
    adapt_thresh - float
                   the pixel size of the patch used in the adaptive threshold
    skel_thresh - float
                  Below this cut off, skeletons with less pixels will be deleted
    branch_thresh - float
                    branches shorter than this length (pixels) will be deleted if extraneous
    pad_size -  int
                size to which filaments will be padded to build a radial intensity profile
    distance - float
               to object in image (in pc)
    region_slice - list
                   option to slice off regions of the given image -- input as [xmin,xmax,ymin,max]

    FUNCTIONS
    ---------
    create_mask - performs image segmentation

    find_optimal_patch_size - finds a suitable patch size to feed into create_mask

    medskel - reduce filaments to skeleton structure

    analyze_skeletons - cleans, labels, and finds the length of the skeletons

    find_widths - fits the radial profiles of the filaments

    results - returns final lengths and widths. Flags bad fits

    save_table - creates a .csv file of the results

    run - the whole shebang


    OUTPUTS
    -------



    """
    def __init__(self, image, hdr, beamwidth, skel_thresh, branch_thresh, pad_size, flatten_thresh, smooth_size=None, \
                size_thresh=None, glob_thresh=None, adapt_thresh=None, distance=None, region_slice=None, mask=None):

        img_dim = len(image.shape)
        if img_dim<2 or img_dim>2:
            raise TypeError("Image must be 2D array. Input array has %s dimensions.")  % (img_dim)
        if region_slice==None:
            self.image = image
        else:
            slices = (slice(region_slice[0],region_slice[1],None), \
                        slice(region_slice[2],region_slice[3],None))
            self.image = np.pad(image[slices],1,padwithzeros)

        self.header = hdr
        self.skel_thresh = skel_thresh
        self.branch_thresh = branch_thresh
        self.pad_size = pad_size

        self.mask = None
        if mask is not None:
            mask[np.isnan(mask)] = 0.0
            self.mask = mask

        if distance==None:
            print "No distance given. Results will be in pixel units."
            self.imgscale = 1.0 ## pixel
            self.beamwidth = beamwidth * (hdr["CDELT2"] * 3600)**(-1) ## where CDELT2 is in degrees
            self.pixel_unit_flag = True
        else:
            self.imgscale = (hdr['CDELT2']*(np.pi/180.0)*distance) ## pc
            self.beamwidth = (beamwidth/np.sqrt(8*np.log(2.))) * (2*np.pi / 206265.) * distance
            self.pixel_unit_flag = False


        self.glob_thresh = glob_thresh
        self.adapt_thresh = adapt_thresh
        self.flatten_thresh = flatten_thresh
        self.smooth_size = smooth_size
        self.size_thresh = size_thresh

        self.smooth_image = None
        self.flat_image = None
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


    def create_mask(self, glob_thresh=None, adapt_thresh=None, smooth_size=None, size_thresh=None, verbose=False): ## Give option to give live inputs to change thresh??

        if self.mask is not None:
            return self ## Skip if pre-made mask given

        if glob_thresh is not None:
            self.glob_thresh = glob_thresh
        if adapt_thresh is not None:
            self.adapt_thresh = adapt_thresh
        if smooth_size is not None:
            self.smooth_size = smooth_size
        if size_thresh is not None:
            self.size_thresh = size_thresh

        if self.size_thresh is None:
            self.size_thresh = round(0.1 * np.pi * (0.5) * (3/40.) * self.imgscale**-2)
            ## Area of ellipse for typical filament size. Divided by 10 to incorporate sparsity.
        if self.adapt_thresh is None:
            self.adapt_thresh = round(0.2/self.imgscale) ## twice average FWHM for filaments
        if self.smooth_size is None:
            self.smooth_size = round(0.05 / self.imgscale) ## half average FWHM for filaments

        self.flat_img = np.arctan(self.image/scoreatpercentile(self.image[~np.isnan(self.image)],self.flatten_thresh))
        self.smooth_img = nd.median_filter(self.flat_img, size=self.smooth_size)
        adapt = threshold_adaptive(self.smooth_img, self.adapt_thresh)
        opening = nd.binary_opening(adapt, structure=np.ones((3,3)))
        cleaned = remove_small_objects(opening, min_size=self.size_thresh)
        self.mask = nd.binary_closing(cleaned, structure=np.ones((3,3)))


        if self.glob_thresh is not None:
            premask = self.flat_img > scoreatpercentile(self.flat_img[~np.isnan(self.flat_img)], self.glob_thresh)
            self.mask = premask * self.mask

        if verbose:
            scale = 0
            p.contour(self.mask)
            vmax = np.nanmax(self.image)
            while scale==0:
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

    def medskel(self, return_distance=True, verbose = False):

        if return_distance:
            self.skeleton,self.medial_axis_distance = medial_axis(self.mask, return_distance=return_distance)
            if self.pixel_unit_flag:
                print "Setting arbitrary width threshold to 2 pixels"
                width_threshold = raw_input("Enter threshold change or pass: ") ## Put time limit on this
                if width_threshold == "":
                    width_threshold = 2
                width_threshold = float(width_threshold)
            else:
                width_threshold = round((0.1/10.)/self.imgscale) # (in pc) Set to be a tenth of expected filament width
            self.skeleton[np.nonzero(self.medial_axis_distance)<width_threshold] = 0 ## Eliminate narrow connections
        else:
            self.skeleton = medial_axis(self.mask)



        if verbose: # For examining results of skeleton
            masked_image = self.image * self.mask
            skel_points = np.where(self.skeleton==1)
            for i in range(len(skel_points[0])):
                masked_image[skel_points[0][i],skel_points[1][i]] = np.NaN
            p.imshow(masked_image,interpolation=None,origin="lower")
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
        if self.medial_axis_distance != None:
            labels, n = nd.label(self.mask, eight_con())
            av_widths = nd.sum(self.medial_axis_distance, labels, range(1, n+1)) / nd.sum(self.skeleton, labels, range(1, n+1))
            if verbose:
                p.hist(av_widths)
                p.show()

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

    def save_table(self, path = None):
        '''

        Save a table results as a csv (in form of pandas dataframe)

        INPUTS
        ------

        path - str
               path where the file should be saved

        '''
        from pandas import DataFrame, Series

        data = {"Lengths" : Series(self.lengths), \
                "Curvature" : Series(self.curvature),\
                "Widths" : Series(self.widths), \
                # "Peak Intensity" : Series(self.width_fits["Parameters"][0]), \
                # "Intensity Error" : Series(self.width_fits["Errors"][0]), \
                # "Gauss. Width" : Series(self.width_fits["Parameters"][1]), \
                # "Width Error" : Series(self.width_fits["Errors"][1]), \
                # "Background" : Series(self.width_fits["Parameters"][2]), \
                # "Background Error" : Series(self.width_fits["Errors"][2]), \
                "Branches" : Series(self.branch_info["filament_branches"]), \
                "Branch Lengths" : Series(self.branch_info["branch_lengths"])}

        df = DataFrame(data)

        if not path:
            filename = "".join([self.header["OBJECT"],".csv"])
        else:
            if path[-1] != "/":
                path = "".join(path,"/")
            filename = "".join([path,self.header["OBJECT"],".csv"])

        df.to_csv(filename)



    def __str__(self):
            print("%s filaments found.") % (self.number_of_filaments)
            for fil in range(self.number_of_filaments):
                print "Filament: %s, Width: %s, Length: %s, Curvature: %s" % \
                        (fil,self.widths[fil],self.lengths[fil], self.curvature[fil])

    def run(self, verbose = False):
        try: ## Check if graphviz is available
            import graphviz
            graph_verbose = verbose
        except ImportError:
            graph_verbose = False

        if verbose:
            print "Best to run in pylab for verbose output."

        self.create_mask(verbose = verbose)
        self.medskel(verbose = verbose)

        self.analyze_skeletons(verbose = graph_verbose)
        self.find_widths(verbose = verbose)
        self.results()
        self.__str__()
        self.save_table()




if __name__ == "__main__":
    import sys
    fib(int(sys.argv[1]))