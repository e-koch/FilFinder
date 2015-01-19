# Licensed under an MIT open source license - see LICENSE

'''

Make histograms for sets of data

'''

import os
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as p
import sys
import matplotlib.cm as cm
from matplotlib import rc
# rc("font", **{"family": "sans-serif", "size": 16})
# rc("text", usetex=True)
from scipy.stats import scoreatpercentile

widths = {}
lengths = {}
curvature = {}
amplitude = {}
orientation = {}
background = {}
median_bright = {}

# Setup to work on noise. If using from repo, use ["."]
# folders = ["."]
folders = [f for f in os.listdir(".") if os.path.isdir(f) and f[-3:] == '350']

# Proper names to show in plots
labels = {"pipeCenterB59-350": "Pipe",
          "polaris-350": "Polaris",
          "ic5146-350": "IC-5146",
          "orionA-S-350": "Orion-A South",
          "lupusI-350": "Lupus",
          "orionB-350": "Orion-B",
          "taurusN3-350": "Taurus",
          "aquilaM2-350": "Aquila",
          "orionA-C-350": "Orion-A Center",
          "perseus04-350": "Perseus",
          "california_cntr-350": "California Center",
          "california_east-350": "California East",
          "california_west-350": "California West",
          "chamaeleonI-350": "Chamaeleon"}

offsets = {"pipeCenterB59-350": 31.697,
           "polaris-350": 9.330,
           "ic5146-350": 20.728,
           "orionA-S-350": 35.219,
           "lupusI-350": 14.437,
           "orionB-350": 26.216,
           "taurusN3-350": 21.273,
           "aquilaM2-350": 85.452,
           "orionA-C-350": 32.616,
           "perseus04-350": 23.698,
           "california_cntr-350": 9.005,
           "california_east-350": 10.124,
           "california_west-350": 14.678,
           "chamaeleonI-350": -879.063}

for folder in folders:
    csv = [f for f in os.listdir(folder) if f[-3:] == "csv" and not "rht" in f
           and not "deg" in f]
    if csv == []:
        print "No csv file in %s" % (folder)
    else:
        for fil in csv:
            data = Table.read(folder + "/" + fil)
            name = fil[:-10]

            widths[name] = data["FWHM"][np.isfinite(data["FWHM"])]
            amplitude[name] = data["Amplitude"][np.isfinite(data["Amplitude"])]
            lengths[name] = data["Lengths"][np.isfinite(data["FWHM"])]
            curvature[name] = data["Curvature"][np.isfinite(data["FWHM"])]
            orientation[name] = data["Orientation"][np.isfinite(data["FWHM"])]
            background[name] = data["Background"][np.isfinite(data["FWHM"])]
            median_bright[name] = data["Median Brightness"][
                np.isfinite(data["FWHM"])]

# Make scatter plots
scatter = sys.argv[1]
if scatter == "T":
    scatter = True
else:
    scatter = False
# Make histograms
hists = sys.argv[2]
if hists == "T":
    hists = True
else:
    hists = False
# Make triangle plot
triangle_plot = sys.argv[3]
if triangle_plot == "T":
    triangle_plot = True
else:
    triangle_plot = False
# Create KS Test tables
ks_tests = sys.argv[4]
if ks_tests == "T":
    ks_tests = True
else:
    ks_tests = False
# Compute the covering fraction
covering_frac = sys.argv[5]
if covering_frac == "T":
    covering_frac = True
else:
    covering_frac = False

# Stability analysis
stability = sys.argv[6]
if stability == "T":
    stability = True
else:
    stability = False

# Scatter plots
if scatter:
    # print labels
    symb_col = ["bD", "gD", "rD", "kD", "b^", "g^", "r^",
                "k^", "bo", "go", "ro", "ko", "bv", "gh", "rh", "kh"]
    # for i, key in enumerate(widths.keys()):
    #     p.plot(np.log10(widths[key][(widths[key] > 0.0)]), np.log10(amplitude[key][(widths[key] > 0.0)]), symb_col[i], label=labels[key],
    #            markersize=6)
    # p.legend()
    # p.grid(True)
    # p.xlabel("Widths (pc)")
    # p.ylabel("Surface Brightness (MJy/sr)")
    # # p.xlim([0.0, 1.0])
    # # p.ylim([0.0, 2500.])
    # p.show()
    # p.clf()

    # Amplitude vs. Length
    lenmed = []
    lenstd = []
    ampmed = []
    ampstd = []

    fig = p.figure()  # figsize=(12, 9), dpi=100)
    for i, key in enumerate(np.sort(widths.keys())):
        loglength = np.log10(lengths[key])
        lenmed = scoreatpercentile(loglength, 50)
        len75 = scoreatpercentile(loglength, 75)
        len25 = scoreatpercentile(loglength, 25)
        print labels[key]+": "+str([len25, lenmed, len75])
        logamp = np.log10(amplitude[key] - background[key])
        ampmed = scoreatpercentile(logamp, 50)
        amp75 = scoreatpercentile(logamp, 75)
        amp25 = scoreatpercentile(logamp, 25)

        p.errorbar(lenmed, ampmed, fmt=symb_col[i], xerr=[[np.abs(lenmed - len25)], [np.abs(lenmed + len75)]],
                   yerr=[[np.abs(ampmed - amp25)], [np.abs(ampmed - amp75)]], label=labels[key], markersize=10,
                   alpha=0.6)
    p.xlabel("log$_{10}($L/ pc)", fontsize=18)
    p.ylabel("log$_{10}(I$/ MJy/sr)", fontsize=18)
    p.grid(True)
    p.legend(loc="lower right", ncol=2, prop={"size": 12}, markerscale=0.75)
    p.ylim([-0.3, 1.55])
    p.xlim([0.15, 1.7])
    # fig.savefig("length_vs_amp_centroids.eps", format="eps", dpi=1000)
    # p.clf()
    p.tight_layout()
    p.show()

    # Orientation vs. Amplitude

    # for i, key in enumerate(widths.keys()):
    #     weights = median_bright[key]
    #     # factor = 20/weights.min()
    #     # weights *= factor
    #     p.title(key)
    #     p.scatter(orientation[key], (median_bright[key]), c=symb_col[i][0],
    #               marker=symb_col[i][1], alpha=0.3)  # , s=weights)
    #     p.grid(True)
    #     p.xlabel(r"$\theta$")
    #     p.ylabel(r"log$_{10}(I$/ MJy/sr)")
    #     p.show()

    # Length vs Width
    lenmed = []
    lenstd = []
    widthmed = []
    widthstd = []

    fig = p.figure()  # figsize=(12, 9), dpi=100)
    for i, key in enumerate(widths.keys()):
        loglength = np.log10(lengths[key][(widths[key] > 0.0)])
        lenmed = scoreatpercentile(loglength, 50)
        len75 = scoreatpercentile(loglength, 75)
        len25 = scoreatpercentile(loglength, 25)
        # print labels[i]+": "+str([len25, lenmed, len75])
        logwidth = np.log10(widths[key][(widths[key] > 0.0)])
        widthmed = scoreatpercentile(logwidth, 50)
        width75 = scoreatpercentile(logwidth, 75)
        width25 = scoreatpercentile(logwidth, 25)

        p.errorbar(lenmed, widthmed, fmt=symb_col[i], xerr=[[np.abs(lenmed - len25)], [np.abs(lenmed + len75)]],
                   yerr=[
                       [np.abs(widthmed - width25)], [np.abs(widthmed - width75)]],
                   label=labels[key], markersize=10, alpha=0.6)
    p.xlabel("log$_{10}($L/ pc)", fontsize=18)
    p.ylabel("log$_{10}(W$/ pc)", fontsize=18)
    p.grid(True)
    p.legend(loc="lower right", ncol=2, prop={"size": 12}, markerscale=0.75)
    p.ylim([-1.0, -0.05])
    p.xlim([0.18, 1.6])
    # fig.savefig("length_vs_width_centroids.eps", format="eps", dpi=1000)
    # p.clf()
    p.show()

# Triangle plot
if triangle_plot:
    import triangle

    for i, key in enumerate(widths.keys()):
        if i == 0:
            data = np.asarray([np.log10(widths[key][(widths[key] > 0.0)]), np.log10((amplitude[key]-background[key])[(widths[key] > 0.0)]),
                               np.log10(lengths[key][(widths[key] > 0.0)]), curvature[key][(widths[key] > 0.0)]])
        else:
            data = np.hstack([data, np.asarray([np.log10(widths[key][(widths[key] > 0.0)]), np.log10(amplitude[key][(widths[key] > 0.0)]),
                                                np.log10(lengths[key][(widths[key] > 0.0)]), curvature[key][(widths[key] > 0.0)]])])

        # Plot it.
    figure = triangle.corner(data.T, labels=["log$_{10}$(W/ pc)",
                                             "log$_{10}$($I$/ MJy/sr)",
                                             "log$_{10}$(L/ pc)", r"$\delta$$\theta$", "$\theta$"],
                             quantiles=[0.50, 0.85, 0.995], bins=9,
                             show_titles=False, title_args={"fontsize": 18})
    # figure.savefig('hgbs_scatter_hists.pdf', format='pdf', dpi=1000)
    p.show()

# Histograms
if hists:
    pass

# bins = np.linspace(0.0, 1.0, 50)
# n, bins, patches = p.hist([widths[key] for key in widths.keys()], bins, stacked=True, color=colours, label=widths.keys())
# p.legend()
# p.ylabel("N")
# p.xlabel(r"parsecs")
# p.show()
# all_widths = all_widths[all_widths<1.0]
# print "Median Width %s" % (all_widths[all_widths<1.0].median().median())
# print "IQR %s" %
# ((all_widths[all_widths<1.0].quantile(.75)-all_widths[all_widths<1.0].quantile(.25)).mean())

# lengths
# medians = []
# iqr = []
# maximum = []
# for key in lengths.keys():
# medians.append(np.median(lengths[key]))
# iqr.append(scoreatpercentile(lengths[key], 75) - scoreatpercentile(lengths[key], 25))
# maximum.append(np.max(lengths[key]))

# n, bins, patches = p.hist([lengths[key] for key in lengths.keys()], bins=200, stacked=True, color=colours, label=lengths.keys())
# p.ylabel("N")
# p.xlabel(r"parsecs")
# p.legend()
# p.show()

# print "Median Length %s" % (np.median(medians))
# print "IQR %s" % (np.mean(iqr))
# print "Max Length %s" % (np.max(maximum))


# Curvature
# n, bins, patches = p.hist([curvature[key] for key in curvature.keys()], bins=50, stacked=True, color=colours, label=curvature.keys())
# p.ylabel("N")
# p.xlabel(r"$\theta$")
# p.legend()
# p.show()

# medians = []
# iqr = []
# maximum = []
# for key in curvature.keys():
#     medians.append(np.median(curvature[key]))
#     iqr.append(scoreatpercentile(curvature[key], 75) - scoreatpercentile(curvature[key], 25))
#     maximum.append(np.max(curvature[key]))
# print "Median Curvature %s" % (np.median(medians))
# print "IQR %s" % (np.mean(iqr))

if ks_tests:

    from scipy.stats import ks_2samp
    from pandas import DataFrame
    import warnings
    # Because we have non-continuous distributions, we use a
    # bootstrap to create a value pvalue for the KS Test
    try:
        execfile("/Users/ekoch/Dropbox/code_development/misc/R_ksboot.py")
        boot = True
    except:  # You need R and the Matching package for this to work
        warnings.warn("Using scipy ks_2samp, not the bootstrap method.")
        boot = False

    boot = False

    def ks_table(param_dict, boot=boot):
        '''
        '''

        pvals = np.zeros((len(param_dict.keys()), len(param_dict.keys())))
        stats = np.zeros((len(param_dict.keys()), len(param_dict.keys())))

        for i, key in enumerate(np.sort(param_dict.keys())[::-1]):
            print key
            for j, key2 in enumerate(np.sort(param_dict.keys())[::-1]):
                if i == j:
                    pvals[i, j] = 0
                    stats[i, j] = 0
                else:
                    if boot:
                        values = ks_boot(param_dict[key], param_dict[key2])
                    else:
                        values = ks_2samp(param_dict[key], param_dict[key2])
                    pvals[i, j] = values[1]
                    stats[i, j] = values[0]

        return stats, pvals

    ordered_labels = []
    for key in np.sort(widths.keys())[::-1]:
        ordered_labels.append(labels[key])

    # Widths
    width_tables = ks_table(widths, boot=boot)

    width_kd_table = DataFrame(
        width_tables[0], index=ordered_labels, columns=ordered_labels)
    # width_kd_table.to_latex("width_ks_table.tex")
    width_kd_table.to_csv("width_ks_table.csv")

    width_kd_table = DataFrame(
        width_tables[1], index=ordered_labels, columns=ordered_labels)
    # width_kd_table.to_latex("width_ks_table_pvals.tex")
    width_kd_table.to_csv("width_ks_table_pvals.csv")

    # Lengths
    length_tables = ks_table(lengths, boot=boot)

    length_kd_table = DataFrame(
        length_tables[0], index=ordered_labels, columns=ordered_labels)
    # length_kd_table.to_latex("length_ks_table.tex")
    length_kd_table.to_csv("length_ks_table.csv")

    length_kd_table = DataFrame(
        length_tables[1], index=ordered_labels, columns=ordered_labels)
    # length_kd_table.to_latex("length_ks_table_pvals.tex")
    length_kd_table.to_csv("length_ks_table_pvals.csv")

    # Orientations
    # Convert to sin(2*phi) to deal with continuity issues
    for key in orientation.keys():
        orientation[key] = np.sin(2 * orientation[key])
    orientation_tables = ks_table(orientation, boot=boot)
    orientation_kd_table = DataFrame(
        orientation_tables[0], index=ordered_labels, columns=ordered_labels)
    # orientation_kd_table.to_latex("orientation_ks_table.tex")
    orientation_kd_table.to_csv("orientation_ks_table.csv")

    orientation_kd_table = DataFrame(
        orientation_tables[1], index=ordered_labels, columns=ordered_labels)
    # orientation_kd_table.to_latex("orientation_ks_table_pvals.tex")
    orientation_kd_table.to_csv("orientation_ks_table_pvals.csv")

    # Curvature
    curvature_tables = ks_table(curvature, boot=boot)

    curvature_kd_table = DataFrame(
        curvature_tables[0], index=ordered_labels, columns=ordered_labels)
    # curvature_kd_table.to_latex("curvature_ks_table.tex")
    curvature_kd_table.to_csv("curvature_ks_table.csv")

    curvature_kd_table = DataFrame(
        curvature_tables[1], index=ordered_labels, columns=ordered_labels)
    # curvature_kd_table.to_latex("curvature_ks_table_pvals.tex")
    curvature_kd_table.to_csv("curvature_ks_table_pvals.csv")

    # Amplitudes
    amplitude_tables = ks_table(amplitude, boot=boot)

    amplitude_kd_table = DataFrame(
        amplitude_tables[0], index=ordered_labels, columns=ordered_labels)
    # amplitude_kd_table.to_latex("amplitude_ks_table.tex")
    amplitude_kd_table.to_csv("amplitude_ks_table.csv")

    amplitude_kd_table = DataFrame(
        amplitude_tables[1], index=ordered_labels, columns=ordered_labels)
    # amplitude_kd_table.to_latex("amplitude_ks_table_pvals.tex")
    amplitude_kd_table.to_csv("amplitude_ks_table_pvals.csv")

if covering_frac:

    from pandas import DataFrame
    from astropy.io.fits import getdata
    import astropy.units as u

    def compute_cf(length, width, amplitude, background, image, header, distance):
        '''
        Compute a "covering fraction" which roughly describes the extent
        that filaments cover in a region.
        '''

        cdelt2 = header["CDELT2"] * (u.degree)  # /u.pixel
        conv = cdelt2 * (np.pi / 180.) * (1 / u.degree)
        img_scale = distance * u.pc * conv

        fil_cover = np.sum(length * width * (amplitude - background) /
                           2. * (u.pc ** 2. * u.Jy / u.sr * 1e6) * img_scale ** -2.)
        im_cover = np.nansum(image[image > 0.0]) * u.Jy / u.sr * 1e6

        print fil_cover
        print im_cover

        return fil_cover / im_cover

    dist = [145., 175., 260., 400., 150., 170., 235.,
            460., 400., 400., 450., 450., 450.]  # pc

    ordered_labels = []
    cf = np.empty((len(labels), ))
    print widths.keys()
    for i, name in enumerate(np.sort(widths.keys())):
        # Load the image in
        img, head = getdata("../" + name + ".fits", header=True)
        img = img + offsets[name]
        model = getdata(name + "/" + name + "_filament_model.fits")
        cf[i] = np.nansum(model) / np.nansum(img)
        ordered_labels.append(labels[name])
        # cf[i] = compute_cf(lengths[name], widths[name], amplitude[name], background[name], im, head, dist[i]).value
    df = DataFrame(cf, index=ordered_labels, columns=["Covering Fraction"])
    print(df)
    df.to_latex("covering_fracs.tex")
    df.to_csv("covering_fracs.csv")

if stability:

    from astropy.units import G

    # Still need conversion to column densities??

    # Divide by canonical cdens_0 value??
    M_line = lambda cdens : 2*cdens / G

    # Plotting width versus \lambda / \lambda_0
    # Split - Orion + California against the rest
    from seaborn import kdeplot

    widths1 = np.empty((0))
    widths2 = np.empty((0))
    intens1 = np.empty((0))
    intens2 = np.empty((0))

    for key in widths.keys():
        if key[0] == 'c':
            widths1 = np.append(widths1, widths[key])
            intens1 = np.append(intens1, amplitude[key])
        else:
            widths2 = np.append(widths2, widths[key])
            intens2 = np.append(intens2, amplitude[key])

    kdeplot(widths1, data2=intens1, shade=True)
    kdeplot(widths2, data2=intens2, shade=True)
    p.show()
