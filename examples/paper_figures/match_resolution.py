# Licensed under an MIT open source license - see LICENSE

'''
Check resolution effects of masking process.
Degrade an image to match the resolution of a more distant one,
then compare the outputs.
'''

from fil_finder import fil_finder_2D
import numpy as np
from astropy.io.fits import getdata
from astropy import convolution
import matplotlib.pyplot as p

# We want to compare one of the closest regions (Pipe)
# to one of the most distant (Orion-A S).

pipe_img, pipe_hdr = getdata("pipeCenterB59-250.fits", header=True)
pipe_distance = 140.  # pc

# orion_img, orion_hdr = getdata("orionA-S-250.fits", header=True)
orion_distance = 400.  # pc

r = orion_distance / pipe_distance
conv = np.sqrt(r**2. - 1)


## What to do?
compute = False
output = True
downsample = False


if compute:

    kernel = convolution.Gaussian2DKernel(conv)

    pipe_degraded = convolution.convolve(pipe_img, kernel, boundary='fill',
                                         fill_value=np.NaN)

    p.subplot(121)
    p.imshow(np.arctan(pipe_img/np.percentile(pipe_img[np.isfinite(pipe_img)], 95)),
             origin="lower", interpolation="nearest")
    p.subplot(122)
    p.imshow(np.arctan(pipe_degraded/np.percentile(pipe_degraded[np.isfinite(pipe_degraded)], 95)),
             origin="lower", interpolation="nearest")
    p.show()

    filfind = fil_finder_2D(pipe_degraded, pipe_hdr, 18.2, 30, 15, 30, distance=400, glob_thresh=20)
    filfind.run(verbose=False, save_name="degraded_pipe", save_plots=False)

## Analysis
if output:

    from astropy.table import Table

    deg_pipe_analysis = Table.read("degraded_pipe_table.fits")
    pipe_analysis = Table.read("pipeCenterB59-250/pipeCenterB59-250_table.fits")

    # Plot lengths, widths, orientation, curvature. Adjust for distance difference

    # p.subplot2grid((4,2), (0,0))
    p.subplot(411)
    num1 = int(np.sqrt(deg_pipe_analysis["FWHM"][np.isfinite(deg_pipe_analysis["FWHM"])].size))
    num2 = int(np.sqrt(pipe_analysis["FWHM"][np.isfinite(pipe_analysis["FWHM"])].size))
    p.hist(deg_pipe_analysis["FWHM"][np.isfinite(deg_pipe_analysis["FWHM"])] / conv,
           bins=num1, label="Degraded", alpha=0.5, color='g')
    p.hist(pipe_analysis["FWHM"][np.isfinite(pipe_analysis["FWHM"])],
           bins=num2, label="Normal", alpha=0.5, color='b')
    p.xlabel("Width (pc)")
    p.legend()

    # p.subplot2grid((4,2), (0,1))
    p.subplot(412)
    p.hist(deg_pipe_analysis["Lengths"] / conv, bins=num1, label="Degraded", alpha=0.5)
    p.hist(pipe_analysis["Lengths"], bins=num2, label="Normal", alpha=0.5)
    p.xlabel("Length (pc)")
    # p.legend()

    # p.subplot2grid((4,2), (1,0))
    p.subplot(413)
    p.hist(deg_pipe_analysis["Orientation"], bins=num1, label="Degraded", alpha=0.5)
    p.hist(pipe_analysis["Orientation"], bins=num2, label="Normal", alpha=0.5)
    p.xlabel("Orientation")
    # p.legend()

    # p.subplot2grid((4,2), (1,1))
    p.subplot(414)
    p.hist(deg_pipe_analysis["Curvature"], bins=num1, label="Degraded", alpha=0.5)
    p.hist(pipe_analysis["Curvature"], bins=num2, label="Normal", alpha=0.5)
    p.xlabel("Curvature")
    # p.legend()

    # p.savefig("pipe_comparison_hists.pdf")
    # p.savefig("pipe_comparison_hists.eps")
    p.show()

    ## Compare distributions using KS Test

    from scipy.stats import ks_2samp

    fwhm_ks = ks_2samp(deg_pipe_analysis["FWHM"][np.isfinite(deg_pipe_analysis["FWHM"])] / conv,
                       pipe_analysis["FWHM"][np.isfinite(pipe_analysis["FWHM"])])

    l_ks = ks_2samp(deg_pipe_analysis["Lengths"] / conv,
                    pipe_analysis["Lengths"])

    o_ks = ks_2samp(np.sin(deg_pipe_analysis["Orientation"]),
                    np.sin(pipe_analysis["Orientation"]))

    c_ks = ks_2samp(deg_pipe_analysis["Curvature"],
                    pipe_analysis["Curvature"])

    ks_tab = Table([fwhm_ks, l_ks, o_ks, c_ks],
                   names=["FWHM", "Length", "Orientation", "Curvature"])

    # ks_tab.write("pipe_comparison_ks_results.csv")
    # ks_tab.write("pipe_comparison_ks_results.tex")

    ## Compare skeletons

    deg_pipe_skel = getdata("degraded_pipe_skeletons.fits", 0)
    deg_pipe_skel[np.where(deg_pipe_skel>1)] = 1
    deg_pipe_skel = deg_pipe_skel[510:1200, 1440:1920]
    filfind = fil_finder_2D(pipe_img, pipe_hdr, 18.2, 30, 15, 30, distance=400, glob_thresh=20)
    filfind.create_mask(border_masking=True)
    filfind.medskel(verbose=False)
    filfind.analyze_skeletons()
    pipe_skel = filfind.skeleton[30:-30, 30:-30]  #getdata("pipeCenterB59-250/pipeCenterB59-250_skeletons.fits", 0)
    pipe_skel[np.where(pipe_skel>1)] = 1
    pipe_skel = pipe_skel[510:1200, 1440:1920]

    # p.subplot2grid((4,2), (2,0), colspan=2, rowspan=2)

    pipe_img = pipe_img[510:1200, 1440:1920]
    ax = p.imshow(np.arctan(pipe_img/np.percentile(pipe_img[np.isfinite(pipe_img)], 95)),
                  origin="lower", interpolation="nearest", cmap="binary")

    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])

    cont1 = p.contour(pipe_skel, colors="b", linewidths=3, label="Normal")
    cont1.collections[0].set_label("Normal")
    cont2 = p.contour(deg_pipe_skel, colors="g", alpha=0.5, label="Degraded")
    cont2.collections[0].set_label("Degraded")
    p.legend(loc="upper right")
    p.show()

if downsample:
    def downsample_axis(myarr, factor, axis, estimator=np.nanmean, truncate=False):
            """
    Downsample an ND array by averaging over *factor* pixels along an axis.
    Crops right side if the shape is not a multiple of factor.

    This code is pure np and should be fast.

    Parameters
    ----------
    myarr : `~numpy.ndarray`
    The array to downsample
    factor : int
    The factor to downsample by
    axis : int
    The axis to downsample along
    estimator : function
    defaults to mean. You can downsample by summing or
    something else if you want a different estimator
    (e.g., downsampling error: you want to sum & divide by sqrt(n))
    truncate : bool
    Whether to truncate the last chunk or average over a smaller number.
    e.g., if you downsample [1,2,3,4] by a factor of 3, you could get either
    [2] or [2,4] if truncate is True or False, respectively.
    """
            # size of the dimension of interest
            xs = myarr.shape[axis]

            if xs % int(factor) != 0:
                if truncate:
                    view = [slice(None) for ii in range(myarr.ndim)]
                    view[axis] = slice(None,xs-(xs % int(factor)))
                    crarr = myarr[view]
                else:
                    newshape = list(myarr.shape)
                    newshape[axis] = (factor - xs % int(factor))
                    extension = np.empty(newshape) * np.nan
                    crarr = np.concatenate((myarr,extension), axis=axis)
            else:
                crarr = myarr

            def makeslice(startpoint,axis=axis,step=factor):
                # make empty slices
                view = [slice(None) for ii in range(myarr.ndim)]
                # then fill the appropriate slice
                view[axis] = slice(startpoint,None,step)
                return view

            # The extra braces here are crucial: We're adding an extra dimension so we
            # can average across it!
            stacked_array = np.concatenate([[crarr[makeslice(ii)]] for ii in range(factor)])

            dsarr = estimator(stacked_array, axis=0)
            return dsarr


    downsample = downsample_axis(pipe_img, 3, axis=0)
    downsample = downsample_axis(downsample, 3, axis=1)

    print downsample.shape
    p.subplot(1,2,1)
    p.title("Pipe Normal")
    p.imshow(np.arctan(pipe_img/np.percentile(pipe_img[np.isfinite(pipe_img)], 95)),
             origin="lower", interpolation="nearest")
    p.subplot(1,2,2)
    p.title("Downsample")
    p.imshow(np.arctan(downsample/np.percentile(downsample[np.isfinite(downsample)], 95)),
             origin="lower", interpolation="nearest")
    p.show()