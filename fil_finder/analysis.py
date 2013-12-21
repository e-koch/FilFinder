'''

Analysis routines for the ouput of the filament finder. These can be run from the filament finder or from a saved .csv file.

'''

import numpy as np
from scipy.stats import nanmean, nanmedian, nanstd

class Analysis(object):
    """

    docstring for Analysis

    INPUTS
    ------

    dataframe - Pandas dataframe
                Contains the output of the filament finder

    verbose - bool
              run in verbose mode for visual inspection. If false, will save plots.

    save - bool
           sets whether to save the plots

    save_name - str
                Prefix of any output

    """
    def __init__(self, dataframe, save=False ,save_name=None, verbose=False):
        super(Analysis, self).__init__()
        if isinstance(dataframe, str):
            from pandas import read_csv
            self.dataframe = read_csv(dataframe)
        else:
            self.dataframe = dataframe
        self.save_name = save_name
        self.verbose = verbose
        self.save = save



    def make_plots(self, num_bins=50):
        import matplotlib.pyplot as p

        ## Histogram of Widths
        widths = [float(x) for x in self.dataframe["Widths"] if is_float_try(x)]
        widths_stats = [nanmean(widths), nanstd(widths), nanmedian(widths)]

        ## Histogram of Lengths
        lengths = self.dataframe["Lengths"]
        lengths_stats = [nanmean(lengths), nanstd(lengths), nanmedian(lengths)]

        ## Histogram of Curvature
        curvature = self.dataframe["Curvature"]
        curvature_stats = [nanmean(curvature), nanstd(curvature), nanmedian(curvature)]



        if self.verbose:
            print "Widths Stats: %s" % (widths_stats)
            print "Lengths Stats: %s" % (lengths_stats)
            print "Curvature Stats: %s" % (curvature_stats)

            p.subplot(131)
            p.hist(widths, num_bins)
            p.xlabel("Widths (pc)")
            p.subplot(132)
            p.hist(lengths, num_bins)
            p.xlabel("Lengths (pc)")
            p.subplot(133)
            p.hist(curvature, num_bins)
            p.xlabel("Curvature")
            p.show()
        if self.save:
            p.hist(widths, num_bins)
            p.xlabel("Widths (pc)")
            p.savefig("".join([self.save_name,"_widths.pdf"]))

            p.hist(lengths, num_bins)
            p.xlabel("Lengths (pc)")
            p.savefig("".join([self.save_name,"_lengths.pdf"]))

            p.hist(widths, num_bins)
            p.xlabel("Lengths (pc)")
            p.savefig("".join([self.save_name,"_lengths.pdf"]))

        return self

def is_float_try(str):
    try:
        float(str)
        return True
    except ValueError:
        return False