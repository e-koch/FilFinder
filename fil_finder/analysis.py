'''

Analysis routines for the output of the filament finder. These can be run from the filament finder or from a saved .csv file.

'''

import numpy as np
from scipy.stats import nanmean, nanmedian, nanstd
from astropy.table import Table

class Analysis(object):
    """

    The Analysis class is meant to house the statistical analysis of the fil_finder output.
    *The complete functionality is not yet in place.*

    """
    def __init__(self, dataframe, save=False ,save_name=None, verbose=False):
        '''
        Parameters
        ----------

        dataframe : Pandas dataframe
                    The dataframe contains the output of fil_finder. This is the .csv file
                    saved by the algorithm.

        verbose : bool
                  Enable this for visual inspection of this histograms. If False,
                  it will enable saving the plots.

        save : bool
               This sets whether to save the plots.

        save_name : str
                    The prefix for the saved file. If None, the name from the header is used.
        '''
        super(Analysis, self).__init__()

        if isinstance(dataframe, str):
            self.dataframe = Table.read(dataframe)
        else:
            self.dataframe = dataframe

        self.save_name = save_name
        self.verbose = verbose
        self.save = save



    def make_hists(self, num_bins=50, use_prettyplotlib=True):

        ## Need this since ppl has no show function.
        import matplotlib.pyplot as p

        if use_prettyplotlib:
            try:
                import prettyplotlib as plt
            except ImportError:
                import matplotlib.pyplot as plt
                print "prettyplotlib not installed. Using matplotlib..."
        else:
            import matplotlib.pyplot as plt

        ## Histogram of Widths
        widths = self.dataframe["Width"]
        widths = widths[np.isfinite(widths)]
        widths_stats = [nanmean(widths), nanstd(widths), nanmedian(widths)]

        ## Histogram of Lengths
        lengths = self.dataframe["Lengths"]
        lengths_stats = [nanmean(lengths), nanstd(lengths), nanmedian(lengths)]

        ## Histogram of Curvature
        rht_curvature = self.dataframe["RHT Curvature"]
        rht_curvature_stats = [nanmean(rht_curvature), nanstd(rht_curvature), nanmedian(rht_curvature)]

        # Histogram of Orientation
        rht_orientation = self.dataframe["Plane Orientation (RHT)"]
        rht_orientation_stats = [nanmean(rht_orientation), nanstd(rht_orientation), nanmedian(rht_orientation)]


        if self.verbose:
            print "Widths Stats: %s" % (widths_stats)
            print "Lengths Stats: %s" % (lengths_stats)
            print "Curvature Stats: %s" % (rht_curvature_stats)

            fig, axes = plt.subplots(nrows=2, ncols=2)
            # Widths
            axes[0, 0].hist(widths, num_bins)
            axes[0, 0].set_xlabel("Widths (pc)")
            # Lengths
            axes[0, 1].hist(lengths, num_bins)
            axes[0, 1].set_xlabel("Lengths (pc)")
            # Curvature
            axes[1, 0].hist(rht_curvature, num_bins)
            axes[1, 0].set_xlabel("Curvature")
            # Orientation
            axes[1, 1].hist(rht_orientation, num_bins)
            axes[1, 1].set_xlabel("Orientation Angle")
            p.show()

        if self.save:
            p.hist(widths, num_bins)
            p.xlabel("Widths (pc)")
            p.savefig("".join([self.save_name,"_widths.pdf"]))
            p.close()

            p.hist(lengths, num_bins)
            p.xlabel("Lengths (pc)")
            p.savefig("".join([self.save_name,"_lengths.pdf"]))
            p.close()

            p.hist(rht_curvature, num_bins)
            p.xlabel("RHT Curvature")
            p.savefig("".join([self.save_name,"_rht_curvature.pdf"]))
            p.close()

        return self
