# Licensed under an MIT open source license - see LICENSE

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
import itertools

widths = {}
lengths = {}
curvature = {}
amplitude = {}
orientation = {}
background = {}
median_bright = {}
branches = {}

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
    # csv = [f for f in os.listdir(folder) if f[-3:] == "csv" and not "rht" in f
    #        and not "deg" in f]
    csv = [f for f in os.listdir(folder) if f[-4:] == "fits" and 'table' in f]

    if csv == []:
        print "No csv file in %s" % (folder)
    else:
        for fil in csv:
            data = Table.read(folder + "/" + fil)
            name = fil[:-11]

            widths[name] = data["FWHM"][np.isfinite(data["FWHM"])]
            amplitude[name] = data["Amplitude"][np.isfinite(data["Amplitude"])]
            lengths[name] = data["Lengths"][np.isfinite(data["FWHM"])]
            curvature[name] = data["Curvature"][np.isfinite(data["FWHM"])]
            orientation[name] = data["Orientation"][np.isfinite(data["FWHM"])]
            background[name] = data["Background"][np.isfinite(data["FWHM"])]
            median_bright[name] = data["Median Brightness"][
                np.isfinite(data["FWHM"])]

            # branches[name] = data['Branch Length']

# Make scatter plots
scatter = sys.argv[1]
if scatter == "T":
    scatter = True
else:
    scatter = False
# Make triangle plot
triangle_plot = sys.argv[2]
if triangle_plot == "T":
    triangle_plot = True
else:
    triangle_plot = False
# Create KS Test tables
ks_tests = sys.argv[3]
if ks_tests == "T":
    ks_tests = True
else:
    ks_tests = False
# Compute the covering fraction
covering_frac = sys.argv[4]
if covering_frac == "T":
    covering_frac = True
else:
    covering_frac = False

# Examine branch lengths
bran_len = sys.argv[5]
if bran_len == "T":
    bran_len = True
else:
    bran_len = False

# Return numbers of unresolved widths and non param usage
width_stats = sys.argv[6]
if width_stats == "T":
    width_stats = True
else:
    width_stats = False


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
    p.legend(loc="lower right", ncol=2, prop={"size": 12}, markerscale=0.75,
             numpoints=1)
    p.ylim([-0.3, 1.55])
    p.xlim([0.1, 1.7])
    # fig.savefig("length_vs_amp_centroids.eps", format="eps", dpi=1000)
    # p.clf()
    p.tight_layout()
    p.show()

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
    # p.ylim([-1.0, -0.05])
    # p.xlim([0.18, 1.6])
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

    truths = np.array([-1.26, np.NaN, np.NaN, np.NaN])

        # Plot it.
    figure = triangle.corner(data.T, labels=["log$_{10}$(W/ pc)",
                                             "log$_{10}$($I$/ MJy/sr)",
                                             "log$_{10}$(L/ pc)", r"$\delta$$\theta$", "$\theta$"],
                             quantiles=[0.15, 0.50, 0.85, 0.995], bins=7,
                             show_titles=False, title_args={"fontsize": 18},
                             truths=truths, truth_color='r')
    # figure.savefig('hgbs_scatter_hists.pdf', format='pdf', dpi=1000)
    p.show()

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

    # width_kd_table = DataFrame(
    #     width_tables[0], index=ordered_labels, columns=ordered_labels)
    # # width_kd_table.to_latex("width_ks_table.tex")
    # width_kd_table.to_csv("width_ks_table.csv")

    width_kd_table = DataFrame(
        width_tables[1], index=ordered_labels, columns=ordered_labels)
    # width_kd_table.to_latex("width_ks_table_pvals.tex")
    width_kd_table.to_csv("width_ks_table_pvals.csv")

    # Lengths
    # length_tables = ks_table(lengths, boot=boot)

    # length_kd_table = DataFrame(
    #     length_tables[0], index=ordered_labels, columns=ordered_labels)
    # # length_kd_table.to_latex("length_ks_table.tex")
    # length_kd_table.to_csv("length_ks_table.csv")

    # length_kd_table = DataFrame(
    #     length_tables[1], index=ordered_labels, columns=ordered_labels)
    # # length_kd_table.to_latex("length_ks_table_pvals.tex")
    # length_kd_table.to_csv("length_ks_table_pvals.csv")

    # Orientations
    # Convert to sin(2*phi) to deal with continuity issues
    # for key in orientation.keys():
    #     orientation[key] = np.sin(2 * orientation[key])
    # orientation_tables = ks_table(orientation, boot=boot)
    # orientation_kd_table = DataFrame(
    #     orientation_tables[0], index=ordered_labels, columns=ordered_labels)
    # # orientation_kd_table.to_latex("orientation_ks_table.tex")
    # orientation_kd_table.to_csv("orientation_ks_table.csv")

    # orientation_kd_table = DataFrame(
    #     orientation_tables[1], index=ordered_labels, columns=ordered_labels)
    # # orientation_kd_table.to_latex("orientation_ks_table_pvals.tex")
    # orientation_kd_table.to_csv("orientation_ks_table_pvals.csv")

    # Curvature
    curvature_tables = ks_table(curvature, boot=boot)

    # curvature_kd_table = DataFrame(
    #     curvature_tables[0], index=ordered_labels, columns=ordered_labels)
    # # curvature_kd_table.to_latex("curvature_ks_table.tex")
    # curvature_kd_table.to_csv("curvature_ks_table.csv")

    curvature_kd_table = DataFrame(
        curvature_tables[1], index=ordered_labels, columns=ordered_labels)
    # curvature_kd_table.to_latex("curvature_ks_table_pvals.tex")
    curvature_kd_table.to_csv("curvature_ks_table_pvals.csv")

    # Amplitudes
    # amplitude_tables = ks_table(amplitude, boot=boot)

    # amplitude_kd_table = DataFrame(
    #     amplitude_tables[0], index=ordered_labels, columns=ordered_labels)
    # # amplitude_kd_table.to_latex("amplitude_ks_table.tex")
    # amplitude_kd_table.to_csv("amplitude_ks_table.csv")

    # amplitude_kd_table = DataFrame(
    #     amplitude_tables[1], index=ordered_labels, columns=ordered_labels)
    # # amplitude_kd_table.to_latex("amplitude_ks_table_pvals.tex")
    # amplitude_kd_table.to_csv("amplitude_ks_table_pvals.csv")

if covering_frac:

    from pandas import DataFrame
    from astropy.io.fits import getdata

    cf = dict.fromkeys(widths.keys())
    for i, name in enumerate(np.sort(widths.keys())):
        # Load the image in
        img = getdata(name + "/" + name + "_regrid_convolved.fits")
        model = getdata(name + "/" + name + "_filament_model.fits")
        cf[name] = np.nansum(model) / np.nansum(img)
    df = DataFrame(cf.values(), index=cf.keys(), columns=["Covering Fraction"])
    df = df.sort()
    print(df)
    df.to_csv("covering_fracs.csv")

if bran_len:
    new_branches = {}

    for key in branches.keys():
        per_branch = []
        for lis in branches[key]:
            # Split out parts
            str_list = lis[1:-1].split(',')
            float_list = []
            for string in str_list:
                float_list.append(float(string))
            per_branch.append(float_list)
        new_branches[key] = per_branch

    for i, key in enumerate(new_branches.keys()):
        all_branches = list(itertools.chain(*new_branches[key]))
        num_bin = np.sqrt(len(all_branches))

        p.subplot(2, 7, i+1)
        p.title(labels[key])
        p.hist(all_branches, bins=num_bin)

        print labels[key], np.percentile(all_branches, [15, 50, 85])

    p.show()

if width_stats:

    from astropy.table import Table

    csv = []

    for folder in folders:
        test = [f for f in os.listdir(folder) if f[-4:] == 'fits' and 'table' in f]
        try:
            csv.append(test[0])
        except:
            print "Not found for " + folder

    fail_frac = np.empty((len(csv), ))
    unres_frac = np.empty((len(csv), ))
    nonparam_frac = np.empty((len(csv), ))
    nonparam_success = np.empty((len(csv), ))
    num_fils = np.empty((len(csv), ))

    for i, (fil, fold) in enumerate(zip(csv, folders)):
        t = Table.read(fold+"/"+fil)

        # Failed fits
        fail_frac[i, ] = sum(np.isnan(t['FWHM'])) #/ float(t['FWHM'].shape[0])

        # Unresolved widths
        fwhm = t['FWHM']
        fwhm = fwhm[np.isfinite(fwhm)]
        unres_frac[i, ] = sum(fwhm > 0) #/ float(t['FWHM'].shape[0])

        # Number that use non-param fits
        nonparam_frac[i, ] = sum(t['Fit Type'] == 'n') #/ float(t['FWHM'].shape[0])

        # Number of successful nonparam fits
        nonparam_success[i, ] = sum(np.logical_and(t['Fit Type'] == 'n', ~np.isnan(t['FWHM'])))

        # Number of filaments
        num_fils[i, ] = t['FWHM'].shape[0]

    df = Table(np.vstack([csv, num_fils, fail_frac, unres_frac,
                          nonparam_frac, nonparam_success]).T,
                   names=['Names', "Number", 'Fail', 'Resolved',
                          'Nonparam', 'Nonparam Success'])

    print(df)

    print sum(num_fils)