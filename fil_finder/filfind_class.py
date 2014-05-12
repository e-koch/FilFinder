#!/usr/bin/python

from cores import *
from curvature import *
from length import *
from pixel_ident import *
from utilities import *
from width import *
from rollinghough import rht
from analysis import Analysis

import numpy as np
import matplotlib.pyplot as p
import scipy.ndimage as nd
from skimage.filter import threshold_adaptive
from skimage.morphology import remove_small_objects, medial_axis
from scipy.stats import scoreatpercentile
from astropy.io import fits
from copy import deepcopy
import os

class fil_finder_2D(object):
    """

    fil_finder is intended for use on astronomical images for detecting
    and analyzing filamentary structure in molecular clouds. Our method
    is largely based on mathematical morphology. When properly tuned, it
    is capable of extracting a complete catalog of filaments from an image
    over the complete range of intensities.

    This class acts as an overall wrapper to run the fil-finder algorithm
    on 2D images and enables visualization and saving capabilities.

    Parameters
    ------
    image : numpy.ndarray
            A 2D array of the data to be analyzed.
    hdr   : dictionary
            The header from fits file containing the data.
    beamwidth : float
                The FWHM beamwidth (in arcseconds) of the instrument used to take the data.
    skel_thresh : float
                  Below this cut off, skeletons with less pixels will be deleted
    branch_thresh : float
                    Any branches shorter than this length (in pixels) will be labeled as extraneous and pruned off.
    pad_size :  int
                The size of the pad (in pixels) used to pad the individual filament arrays.
                This is necessary to build the radial intensity profile.
    flatten_thresh : int
                     The percentile of the data used in the normalization of the arctan transform. If the data contains
                     regions of a much higher intensity than the mean, it is recommended this be set >95 percentile.
    smooth_size : int, optional
                  The patch size (in pixels) used to smooth the flatten image before adaptive thresholding is performed.
                  Smoothing is necessary to ensure the extraneous branches on the skeletons is minimized.
                  If None, the patch size is set to ~0.05 pc. This ensures the large scale structure is not affected while
                  smoothing extraneous pixels off the edges.
    size_thresh : int, optional
                  This sets the lower threshold on the size of objects found in the adaptive thresholding. If None, the
                  value is set at ~0.1*pi*(0.5 pc)*(0.75*0.1 pc) which is 0.1* area of ellipse with a length 0.5 pc and
                  0.75(1/10) pc width, which represent the approximate smallest size of a filament [add citation].
                  Multiplying by 0.1 is meant to take into account an extremely curvy filament, likely more than is
                  physically realizable. Any region smaller than this threshold may be safely labeled as an artifact of
                  the thresholding.
    glob_thresh : float, optional
                  This is the percentile to cut off searching for filamentary structure. Any regions with intensities
                  below this percentile are ignored.
    adapt_thresh : int, optional
                   This is the size of the patch used in the adaptive thresholding. Bright structure is not very sensitive
                   to the choice of patch size, but faint structure is very sensitive. If None, the patch size is set to
                   twice the width of a typical filament (~0.2 pc). As the width of filaments is ubiquitous[citation here],
                   this patch size generally segments all filamentary structure in a given image.
    distance : float, optional
               The distance to the region being examined (in pc). If None, the analysis is carried out in pixel and
               angular units. In this case, the physical priors used in other optional parameters is meaningless
               and each must be specified initially.
    region_slice : list, optional
                   This gives the option to examine a specific region in the given image. The expected input
                   is [xmin,xmax,ymin,max].
    mask : numpy.ndarray, optional
           A pre-made, boolean mask may be supplied to skip the segmentation process. The algorithm will skeletonize
           and run the analysis portions only.

    freq : float
           Frequency of the image. This is required for using the cylindrical model (cyl_model) for the widths.

    Examples
    --------
    >>> from fil_finder import fil_finder_2D
    >>> img,hdr = fromfits("/srv/astro/erickoch/gould_belt/chamaeleonI-250.fits")
    >>> filfind = fil_finder_2D(img, hdr, 15.1, 30, 5, 10, 95 ,distance=160,
                                region_slice=[620,1400,430,1700])
    >>> filfind.run(verbose=False, save_name="chamaeleonI-250", save_plots=True)


    References
    ----------

    """
    def __init__(self, image, hdr, beamwidth, skel_thresh, branch_thresh, pad_size, flatten_thresh, smooth_size=None, \
                size_thresh=None, glob_thresh=None, adapt_thresh=None, distance=None, region_slice=None, mask=None, \
                freq=None):


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
        self.freq = freq

        # If pre-made mask is provided, remove nans if any.
        self.mask = None
        if mask is not None:
            mask[np.isnan(mask)] = 0.0
            self.mask = mask

        # Pad the image by the pad size. Avoids slicing difficulties
        # later on.
        self.image = np.pad(self.image, self.pad_size, padwithnans)

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
        self.widths = {"Fitted Width": [], "Estimated Width": []}
        self.width_fits = {"Parameters": [], "Errors": [], "Names": None}
        self.menger_curvature = None
        self.rht_curvature = {"Mean": [], "Std": []}
        self.filament_arrays = None
        self.labelled_filament_arrays = None
        self.number_of_filaments = None
        self.array_offsets = None
        self.skeleton = None
        self.filament_extents = None
        self.branch_info = None
        self.masked_image = None
        self.medial_axis_distance = None

        self.dataframe = None

    def create_mask(self, glob_thresh=None, adapt_thresh=None, smooth_size=None, size_thresh=None, verbose=False, \
                     test_mode=False):
        '''

        This runs the complete segmentation process and returns a mask of the filaments found.
        The process is broken into six steps:
            * An arctan tranform is taken to flatten extremely bright regions. Adaptive thresholding
              is very sensitive to local intensity changes and small, bright objects(ie. cores)
              will leave patch-sized holes in the mask.
            * The flattened image is smoothed over with a median filter. The size of the patch used
              here is set to be much smaller than the typical filament width. Smoothing is necessary
              to minimizing extraneous branches when the medial axis transform is taken.
            * A binary opening is performed using an 8-connected structure element. This is very
              successful at removing small regions around the edge of the data.
            * Objects smaller than a certain threshold (set to be ~1/10 the area of a small filament)
              are removed to ensure only regions which are sufficiently large enough to be real
              structure remain.
            * Finally, a binary closing is used with an 8-connected structure element. This restores
              the sizes of the remaining regions in the image.

        The parameters for this function are as previously defined. They are included here for fine-tuning
        purposes only.

        Parameters
        ----------

        smooth_size : int, optional
                      See previous definition.
        size_thresh : int, optional
                      See previous definition.
        glob_thresh : float, optional
                      See previous definition.
        adapt_thresh : int, optional
                       See previous definition.
        verbose : bool, optional
                  This enables plotting of the mask for visual inspection. If this is run while in pylab
                  mode, the plotting window updates with user-inputted scaling for the image. Since the
                  filaments are picked out over a large range of intensities, visualizing at multiple
                  thresholds is key to determine the performance of the algorithm.

        test_mode : bool, optional
                    This enables a more in-depth look at the individual steps of the masking process.


        Returns
        -------

        self.flat_img : numpy.ndarray
                        The flattened image after the arctan transform.
        self.smooth_img : numpy.ndarray
                          The smoothed version of self.flat_img.
        self.mask : numpy.ndarray
                    The mask containing all detected filamentary structure.

        '''

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
            self.smooth_size = round(0.02 / self.imgscale) ## half average FWHM for filaments

        self.flat_img = np.arctan(self.image/scoreatpercentile(self.image[~np.isnan(self.image)],self.flatten_thresh))
        self.smooth_img = nd.median_filter(self.flat_img, size=self.smooth_size)
        adapt = threshold_adaptive(self.smooth_img, self.adapt_thresh)

        if self.glob_thresh is not None:
          thresh_value = np.max([0.0, scoreatpercentile(self.flat_img[~np.isnan(self.flat_img)], self.glob_thresh)])
          glob = self.flat_img > thresh_value
          adapt = glob * adapt

        opening = nd.binary_opening(adapt, structure=np.ones((3,3)))
        cleaned = remove_small_objects(opening, min_size=self.size_thresh)
        self.mask = nd.median_filter(cleaned, size=self.smooth_size)

        if test_mode:
          # p.subplot(3,3,1)
          p.imshow(np.log10(self.image), origin="lower", interpolation=None)
          p.colorbar()
          p.show()
          # p.subplot(3,3,2)
          p.imshow(self.flat_img, origin="lower", interpolation=None)
          p.colorbar()
          p.show()
          # p.subplot(3,3,3)
          p.imshow(self.smooth_img, origin="lower", interpolation=None)
          p.colorbar()
          p.show()
          # p.subplot(3,3,4)
          p.imshow(adapt, origin="lower", interpolation=None)
          p.show()
          # p.subplot(3,3,5)
          p.imshow(opening, origin="lower", interpolation=None)
          p.show()
          # p.subplot(3,3,6)
          p.imshow(cleaned, origin="lower", interpolation=None)
          p.show()
          # p.subplot(3,3,7)
          p.imshow(self.mask, origin="lower", interpolation=None)
          p.show()

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
                if rescale=="no" or rescale=="n" or rescale=="":
                    scale = 1
                else:
                    vmax = float(rescale)

        return self

    def medskel(self, return_distance=True, verbose=False):
        '''

        This function performs the medial axis transform (skeletonization) on the mask.
        This is essentially a wrapper function of skimage.morphology.medial_axis. The
        returned skeletons are the objects used for the bulk of the analysis.

        If the distance transform is returned from the transform, it is used as a pruning
        step. Regions where the width of a region are far too small (set to >0.01 pc) are
        deleted. This ensures there no unnecessary connections between filaments.

        Parameters
        ----------
        return_distance : bool, optional
                          This sets whether the distance transform is returned from
                          skimage.morphology.medial_axis.

        verbose : bool, optional
                  If True, the image is overplotted with the skeletons for inspection.

        Returns
        -------

        self.skeleton : numpy.ndarray
                        The array containing all of the skeletons.

        self.medial_axis_distance : numpy.ndarray
                                    The distance transform used to create the skeletons.
                                    Only defined is return_distance=True

        '''

        if return_distance:
            self.skeleton,self.medial_axis_distance = medial_axis(self.mask, return_distance=return_distance)
            self.medial_axis_distance  = self.medial_axis_distance * self.skeleton
            if self.pixel_unit_flag:
                print "Setting arbitrary width threshold to 2 pixels"
                width_threshold = raw_input("Enter threshold change or pass: ") ## Put time limit on this
                if width_threshold == "":
                    width_threshold = 2
                width_threshold = float(width_threshold)
            else:
                width_threshold = round((0.1/10.)/self.imgscale) # (in pc) Set to be a tenth of expected filament width
            narrow_pts = np.where(self.medial_axis_distance<width_threshold)
            self.skeleton[narrow_pts] = 0 ## Eliminate narrow connections
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

    def analyze_skeletons(self,verbose=False):
        '''

        This function wraps most of the skeleton analysis. Several steps are
        completed here:
            * isolatefilaments is run to separate each skeleton into its own
              array. If the skeletons are under the threshold set by
              self.size_thresh, the region is removed. An updated mask is
              also returned.
            * pix_identify classifies each of the pixels in a skeleton as a
              body, end, or interestion point. See the documentation on find_filpix
              for a complete explanation. The function labels the branches and
              intersections of each skeletons.
            * init_lengths finds the length of each branch in each skeleton and
              also returns the coordinates of each of these branches for use in
              the graph representation.
            * pre_graph turns the skeleton structures into a graphing format
              compatible with networkx. Hubs in the graph are the intersections
              and end points, labeled as letters and numbers respectively. Edges
              define the connectivity of the hubs and they are weighted by their
              length.
            * longest_path utilizes networkx.shortest_path_length to find the
              overall length of each of the filaments. The returned path is the
              longest path through the skeleton. If loops exist in the skeleton,
              the longest path is chosen (this shortest path algorithm fails when
              used on loops).
            * extremum_pts returns the locations of the longest path's extent so its
              performance can be evaluated.
            * final_lengths takes the path returned from longest_path and calculates
              the overall length of the filament. This step also acts as to prune the
              skeletons. Any branch shorter than self.branch_thresh is deleted. The
              function also returns an experimental method for determining the curvature
              of the filament(see routines in curvature.py for explanation of the Menger
              Curvature).
              *Note:* The results of the Menger Curvature are not in a current form that
              they may be reliably used.
            * final_analysis combines the outputs and returns the results for further
              analysis.

        Parameters
        ----------

        verbose : bool
                  This enables visualizations of the graph created from each of the
                  skeletons.
                  *Note:* pygraphviz is required to view the graphs.

        Returns
        -------
        self.filament_arrays : list of numpy.ndarray
                               Contains individual arrays of each skeleton
        self.number_of_filaments : int
                                   The number of individual filaments.
        self.array_offsets : list
                             A list of coordinates for each filament array.This will
                             be used to recombine the final skeletons into one array.
        self.filament_extents : list
                                This contains the coordinates of the initial and final
                                position of the skeleton's extent. It may be used to
                                test the performance of the shortest path algorithm.
        self.lengths : list
                       Contains the overall lengths of the skeletons
        self.labeled_fil_arrays : list of numpy.ndarray
                                  Contains the final labeled versions of the skeletons.
        self.branch_info : dict
                           The significant branches of the skeletons have their length
                           and number of branches in each skeleton stored here.
                           The keys are: *filament_branches*, *branch_lengths*
        self.menger_curvature : list
                         The results of the Menger Curvature algorithm.

        '''

        try: ## Check if graphviz is available
            import pygraphviz

        except ImportError:
            verbose = False
            print "pygraphviz is not installed. Verbose output for graphs is disabled."

        isolated_filaments, num, offsets = \
                isolatefilaments(self.skeleton, self.skel_thresh, pad_size=self.pad_size)
        self.number_of_filaments = num
        self.array_offsets = offsets

        interpts, hubs, ends, filbranches, labeled_fil_arrays =  \
                pix_identify(isolated_filaments, num)

        initial_lengths, filament_pixels, branch_intensity = init_lengths(labeled_fil_arrays, filbranches, self.array_offsets, self.image)

        edge_list, nodes = pre_graph(labeled_fil_arrays, initial_lengths, branch_intensity, interpts, ends)

        max_path, extremum, G = longest_path(edge_list, nodes, initial_lengths, verbose=verbose)

        labeled_fil_arrays, edge_list, nodes = prune_graph(G, nodes, edge_list, max_path, labeled_fil_arrays, self.branch_thresh)

        self.filament_extents = extremum_pts(labeled_fil_arrays, extremum, ends)

        main_lengths, branch_lengths, labeled_fil_arrays, curvature = \
            final_lengths(self.image, max_path, edge_list, labeled_fil_arrays, filament_pixels, interpts, filbranches, \
                            initial_lengths, self.imgscale, self.branch_thresh)

        labeled_fil_arrays, filbranches, final_hubs, branch_lengths, filament_arrays = final_analysis(labeled_fil_arrays)

        self.lengths = main_lengths
        self.labelled_filament_arrays = labeled_fil_arrays
        self.filament_arrays = filament_arrays
        self.branch_info = {"filament_branches":filbranches, "branch_lengths":branch_lengths}
        self.menger_curvature = curvature

        return self

    def exec_rht(self, radius=10, ntheta=180, background_percentile=25, verbose=False):
      '''

      Implements the Rolling Hough Transform (Clark et al., 2013). The orientation
      of each filament is denoted by the mean value of the RHT. "Curvature"
      is represented by the standard deviation of the transform.

      **NOTE** We recommend using this curvature value rather than the Menger Curvature.

      Parameters
      **********

      radius : int
               Sets the patch size that the RHT uses.

      ntheta : int, optional
               The number of bins to use for the RHT.

      background : int, optional
                   RHT distribution often has a constant background. This sets the percentile to subtract off (see )

      verbose : bool

      Returns
      *******

      self.rht_curvature : dictionary

      References
      **********

      Clark et al. (2013)

      '''

      for n in range(self.number_of_filaments):
        theta, R = rht(self.filament_arrays[n], radius, ntheta, background_percentile)
        ecdf = np.cumsum(R/np.sum(R))

        median = theta[np.where(ecdf==find_nearest(ecdf,0.5))].mean() ## 50th percentile
        twofive = theta[np.where(ecdf==find_nearest(ecdf,0.25))].mean()
        sevenfive = theta[np.where(ecdf==find_nearest(ecdf,0.75))].mean()

        self.rht_curvature["Mean"].append(median)
        self.rht_curvature["Std"].append(np.abs(sevenfive - twofive)) ## Interquartile range

        if verbose:
          ax1 = p.subplot(121, polar=True)
          ax1.plot(theta, R/R.max(), "kD")
          ax1.fill_between(theta, 0, R/R.max(), facecolor="blue", interpolate=True, alpha=0.5)
          ax1.set_rmax(1.0)
          ax1.plot([median]*2, np.linspace(0.0,1.0, 2), "g")
          ax1.plot([twofive]*2, np.linspace(0.0,1.0, 2), "b--")
          ax1.plot([sevenfive]*2, np.linspace(0.0,1.0, 2), "b--")
          ax2 = p.subplot(122, polar=True)
          ax2.plot(theta, ecdf, "k")
          ax2.set_rmax(1.0)
          ax2.set_yticks([0.25, 0.5, 0.75])
          p.show()

      return self

    def find_widths(self, fit_model=gauss_model, try_nonparam=True, verbose=False):
        '''

        The final step of the algorithm is to find the widths of each of the skeletons. We do this
        by:
            * A Euclidean Distance Transform (scipy.ndarray.distance_transform_edt) is performed on
              each skeleton. The skeletons are also recombined onto a single array. The individual
              filament arrays are padded to ensure a proper radial profile is created. If the padded
              arrays fall outside of the original image, they are trimmed.
            * A user-specified model is fit to each of the radial profiles. There are three models included
              in this package; a gaussian, lorentzian and a cylindrical filament model (Arzoumanian et al., 2011).
              This returns the width and central intensity of each filament. The reported widths are the
              deconvolved FWHM of the gaussian width. For faint or crowded filaments, the fit can fail
              due to lack of data to fit to. In this case, the distance transform from the medial axis transform
              (self.medial_axis_distance) may be used to provide an estimate of the width.

        Parameters
        ----------

        fit_model : function
                Function to fit to the radial profile. "cyl_model" and "gauss_model" are available in widths.py.

        try_nonparam : bool, optional
                       If True, uses a non-parametric method to find the properties of the radial profile in
                       cases where the model fails.

        verbose : bool, optional
                  If True, each of the resultant gaussian fits is plotted on the radial profile. The average
                  widths based on the medial axis distance transform are also plotted.

        Returns
        -------

        self.widths : list
                      List of the FWHM widths returned from the fits.

        self.width_fits : dict
                          Contains the fit parameters and estimations of the errors from each fit.
        self.skeleton : numpy.ndarray

                        Updated versions of the array of skeletons.

        '''

        dist_transform_all, dist_transform_separate, self.skeleton = dist_transform(self.filament_arrays, \
                    self.array_offsets, self.image.shape, self.pad_size, self.branch_thresh)

        def red_chisq(data, fit, nparam, sd):
          N = data.shape[0]
          return np.sum(((fit-data)/sd)**2.) / float(N-nparam-1)

        for n in range(self.number_of_filaments):

            if try_nonparam: # Need the unbinned data for the non-parametric fit.
              dist, radprof, weights, unbin_dist, unbin_radprof = \
                radial_profile(self.image, dist_transform_all, dist_transform_separate[n],
                                self.array_offsets[n], self.imgscale)
            else:
              dist, radprof, weights = radial_profile(self.image, dist_transform_all,\
                     dist_transform_separate[n], self.array_offsets[n], self.imgscale)

            if fit_model==cyl_model:
                if self.freq is None:
                    print "Image not converted to column density. Fit parameters will not match physical meaning. \
                       Please specify frequency."
                else:
                    assert isinstance(self.freq, float)
                    radprof = dens_func(planck(20.,self.freq), 0.2, radprof)*(5.7e19)

            fit, fit_error, model, parameter_names, fail_flag = \
                                    fit_model(dist, radprof, weights, self.beamwidth)

            chisq = red_chisq(radprof, model(dist, *fit[:-1]), 3, 1)

            # If the model isn't doing a good job, try it non-parametrically
            if chisq>10.0 and try_nonparam:
              fit, fit_error, fail_flag = nonparam_width(dist, radprof, unbin_dist, unbin_radprof,
                                               self.beamwidth, 5, 99)

            if n==0:
                ## Prepare the storage
                self.width_fits["Parameters"] = np.empty((self.number_of_filaments, len(parameter_names)))
                self.width_fits["Errors"] = np.empty((self.number_of_filaments, len(parameter_names)))


            if verbose:
                print "Fit Parameters: %s \\ Fit Errors: %s" % (fit, fit_error)
                p.subplot(121)
                p.plot(dist, radprof, "kD")
                points = np.linspace(np.min(dist), np.max(dist), 2*len(dist))
                try: # If FWHM is appended on, will get TypeError
                  p.plot(points, model(points, *fit), "r")
                except TypeError:
                  p.plot(points, model(points, *fit[:-1]), "r")
                p.xlabel(r'Radial Distance (pc)')
                p.ylabel(r'Intensity')
                p.grid(True)
                p.subplot(122)
                xlow, ylow = (self.array_offsets[n][0][0], self.array_offsets[n][0][1])
                xhigh, yhigh = (self.array_offsets[n][1][0], self.array_offsets[n][1][1])
                shape = (xhigh-xlow, yhigh-ylow)
                p.contour(self.filament_arrays[n][self.pad_size:shape[0]-self.pad_size, \
                                self.pad_size:shape[1]-self.pad_size], colors="k")
                img_slice = self.image[xlow+self.pad_size:xhigh-self.pad_size, \
                                ylow+self.pad_size:yhigh-self.pad_size]
                vmin = scoreatpercentile(img_slice[np.isfinite(img_slice)], 10)
                p.imshow(img_slice, interpolation=None, vmin=vmin)
                p.colorbar()
                p.show()

            if fail_flag:
                fit = [np.NaN] * len(fit)
                fit_error = [np.NaN] * len(fit)

            self.widths["Fitted Width"].append(fit[-1])
            self.width_fits["Parameters"][n,:] = fit
            self.width_fits["Errors"][n,:] = fit_error
        self.width_fits["Names"] =  parameter_names

        ## Implement check for failed fits and replace with average width from medial_axis_distance
        if self.medial_axis_distance != None:
            self.widths["Estimated Width"] = medial_axis_width(self.medial_axis_distance, \
                                                               self.mask, self.skeleton) * self.imgscale

        return self

    def results(self):
        '''
        Since failed fits a denoted by a string, this function separates out the failed fits.
        The widths which are unrealistic (width>length), are also labeled as a fit fail. The
        realistic widths are added to the overall lengths. This is done because of the slight
        shortening of each skeleton by the skeletonization process.

        Returns
        -------
        self.lengths : list
                       Updated lengths
        self.widths : list
                      Updated widths
        '''
        overall_lengths = []
        overall_widths = []
        for i, width in enumerate(self.widths["Fitted Width"]):
            if np.isfinite(width):
                if self.lengths[i]>width:
                    overall_lengths.append(self.lengths[i] + width) # Adaptive Threshold shortens ends, so add the width on
                    overall_widths.append(width)
                else:
                    overall_lengths.append(self.lengths[i])
                    overall_widths.append(np.NaN)
            else:
                overall_lengths.append(self.lengths[i])
                overall_widths.append(width)

        self.lengths = np.asarray(overall_lengths)
        self.widths["Fitted Width"] = np.asarray(overall_widths)

        return self

    def save_table(self, table_type="csv", path=None, save_name=None):
        '''

        The results of the algorithm are saved as a csv after converting the data into a pandas dataframe.

        Parameters
        ----------

        table_type : str, optional
               Sets the output type of the table. "csv" uses the pandas package.
               "fits" uses astropy to output a FITS table.

        path : str, optional
               The path where the file should be saved.
        save_name : str, optional
                    The prefix for the saved file. If None, the name from the header is used.

        Returns
        -------

        self.dataframe : pandas dataframe
                         The dataframe is returned for use with the Analysis class.

        '''

        if save_name is None:
            save_name = self.header["OBJECT"]


        if not path:
          if table_type=="csv":
            filename = "".join([save_name,"_table",".csv"])
          elif table_type=="fits":
            filename = "".join([save_name,"_table",".fits"])

        else:
            if path[-1] != "/":
                path = "".join(path,"/")
            if table_type=="csv":
              filename = "".join([save_name,"_table",".csv"])
            elif table_type=="fits":
              filename = "".join([save_name,"_table",".fits"])

        data = {"Lengths" : self.lengths, \
                "Menger Curvature" : self.menger_curvature,\
                "Plane Orientation (RHT)" : self.rht_curvature["Mean"],\
                "RHT Curvature" : self.rht_curvature["Std"],\
                # "Estimated Width" : self.widths["Estimated Width"], \
                "Branches" : self.branch_info["filament_branches"], \
                "Branch Lengths" : self.branch_info["branch_lengths"]}

        for i, param in enumerate(self.width_fits["Names"]):
          data[param] = self.width_fits["Parameters"][:,i]
          data[param+" Error"] = self.width_fits["Errors"][:,i]

        if table_type=="csv":
          from pandas import DataFrame, Series

          for key in data.keys():
            data[key] = Series(data[key])

          df = DataFrame(data)
          df.to_csv(filename)

        elif table_type=="fits":
          from astropy.table import Table

          # Branch Lengths contains a list for each entry, which aren't accepted for BIN tables.
          if "Branch Lengths" in data.keys():
            del data["Branch Lengths"]

          df = Table(data)

          df.write(filename)

        else:
          raise NameError("Only formats supported are 'csv' and 'fits'.")


        self.dataframe = df

        return self

    def save_plots(self, save_name=None, percentile=80.):
      '''

      Creates saved PDF plots of several quantities/images.

      '''

      threshold = scoreatpercentile(self.image[~np.isnan(self.image)], percentile)
      p.imshow(self.image, vmax=threshold, origin="lower", interpolation="nearest")
      p.contour(self.mask)
      p.title("".join([save_name," Contours at ", str(round(threshold))]))
      p.savefig("".join([save_name,"_filaments.pdf"]))
      p.close()

      ## Skeletons
      masked_image = self.image * self.mask
      skel_points = np.where(self.skeleton==1)
      for i in range(len(skel_points[0])):
          masked_image[skel_points[0][i],skel_points[1][i]] = np.NaN
      p.imshow(masked_image, vmax=threshold, interpolation=None, origin="lower")
      p.savefig("".join([save_name,"_skeletons.pdf"]))
      p.close()

      # Return histograms of the population statistics
      Analysis(self.dataframe, save=True, save_name=save_name).make_plots()

      return self

    def save_fits(self, save_name=None):
      '''

      This function saves the mask and the skeleton array as FITS files.
      Included in the header are the setting used to create them.

      Parameters
      ----------

      save_name : str, optional
                  The prefix for the saved file. If None, the name from the header is used.


      '''

      ## Save mask
      hdr_mask = deepcopy(self.header)
      hdr_mask.update("BUNIT", value="bool", comment="")
      hdr_mask.add_comment("Mask created by fil_finder. See fil_finder \
                            documentation for more info on parameter meanings.")
      hdr_mask.add_comment("Smoothing Filter Size: "+str(self.smooth_size))
      hdr_mask.add_comment("Area Threshold: "+str(self.size_thresh))
      hdr_mask.add_comment("Global Intensity Threshold: "+str(self.glob_thresh))
      hdr_mask.add_comment("Size of Adaptive Threshold Patch: "+str(self.adapt_thresh))

      fits.writeto("".join([save_name,"_mask.fits"]), self.mask.astype("float"), hdr_mask)

      ## Save skeletons
      hdr_skel = deepcopy(self.header)
      hdr_skel.update("BUNIT", value="bool", comment="")
      hdr_skel.add_comment("Mask created by fil_finder. See fil_finder \
                            documentation for more info on parameter meanings.")
      hdr_skel.add_comment("Smoothing Filter Size: "+str(self.smooth_size))
      hdr_skel.add_comment("Area Threshold: "+str(self.size_thresh))
      hdr_skel.add_comment("Global Intensity Threshold: "+str(self.glob_thresh))
      hdr_skel.add_comment("Size of Adaptive Threshold Patch: "+str(self.adapt_thresh))
      hdr_skel.add_comment("Skeleton Size Threshold: "+str(self.skel_thresh))
      hdr_skel.add_comment("Branch Size Threshold: "+str(self.branch_thresh))

      fits.writeto("".join([save_name,"_skeletons.fits"]), self.skeleton.astype("float"), hdr_skel)

      # Save stamps of all images. Include portion of image and the skeleton for reference.

      # Make a directory for the stamps
      if not os.path.exists("stamps_"+save_name):
        os.makedirs("stamps_"+save_name)

      for n, (offset, skel_arr) in enumerate(zip(self.array_offsets, self.filament_arrays)):
        xlow, ylow = (offset[0][0], offset[0][1])
        xhigh, yhigh = (offset[1][0], offset[1][1])
        shape = (xhigh-xlow, yhigh-ylow)
        skel_stamp = skel_arr[self.pad_size:shape[0]-self.pad_size, \
                                            self.pad_size:shape[1]-self.pad_size]
        img_stamp = self.image[xlow+self.pad_size:xhigh-self.pad_size, \
                              ylow+self.pad_size:yhigh-self.pad_size]

        ## ADD IN SOME HEADERS!
        prim_hdr = deepcopy(self.header)
        prim_hdr["COMMENT"] = "Outputted from fil_finder."
        prim_hdr["COMMENT"] = "Extent in original array: ("+ \
                              str(xlow+self.pad_size)+","+str(ylow+self.pad_size)+")->"+ \
                              "("+str(xhigh-self.pad_size)+","+str(yhigh-self.pad_size)+")"

        hdu = fits.HDUList()
        # Image stamp
        hdu.append(fits.PrimaryHDU(img_stamp, header=prim_hdr))
        # Stamp of final skeleton
        prim_hdr.update("BUNIT", value="bool", comment="")
        hdu.append(fits.PrimaryHDU(skel_stamp, header=prim_hdr))

        hdu.writeto("stamps/"+save_name+"_object_"+str(n+1)+".fits")

      return self

    def __str__(self):
            print("%s filaments found.") % (self.number_of_filaments)
            for fil in range(self.number_of_filaments):
                print "Filament: %s, Width: %s, Length: %s, Curvature: %s" % \
                        (fil,self.widths["Fitted Width"][fil],self.lengths[fil], self.rht_curvature["Std"][fil])

    def run(self, verbose=False, save_plots=False, save_name=None):
        '''
        The whole algorithm in one easy step. Individual parameters have not been included in this
        batch run. If fine-tuning is needed, it is recommended to run each step individually.
        **This currently contains the saving portion of the plots. This will be changed and updated
        in the near future.**

        Parameters
        ----------
        verbose : bool
                  Enables the verbose option for each of the steps. Also enables printing of the
                  main results of each filament with self.__str__. It is recommended to run verbose
                  mode in Ipython's "pylab" mode.
                  *Note:* if pygraphviz is not installed, the graph plotting will be skipped.
        save_plots : bool
                     If True, enables the saving of the output plots.
        save_name : str
                    The prefix for the saved file. If None, the name from the header is used.

        '''

        if verbose:
            print "Best to run in pylab for verbose output."

        if save_name is None:
            save_name = self.header["OBJECT"]

        self.create_mask(verbose = verbose)
        self.medskel(verbose = verbose)

        self.analyze_skeletons(verbose = verbose)
        self.exec_rht(verbose=verbose)
        self.find_widths(verbose = verbose)
        self.results()
        self.save_table(save_name=save_name, table_type="fits")
        self.save_fits(save_name=save_name)

        if verbose:
            self.__str__()

        if save_plots:
          self.save_plots(save_name=save_name)



        return self
