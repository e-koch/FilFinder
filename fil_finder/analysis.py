'''

Analysis routines for the output of the filament finder. These can be run from the filament finder or from a saved .csv file.

'''

import numpy as np
from scipy.stats import nanmean, nanmedian, nanstd
from astropy.table import Table
import aplpy
import matplotlib # Need access to objects

class Analysis(object):
    """

    The Analysis class is meant to house the statistical analysis of the fil_finder output.
    *The complete functionality is not yet in place.*

    """
    def __init__(self, dataframe, save_name=None, verbose=False, columns=None, save_type="pdf",
                 subplot=True):
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

        if columns is None:
          self.columns = self.dataframe.colnames
          # Remove Fit Type column. Contains strings.
          self.columns.remove("Fit Type")
          if dataframe[-3:] == "csv": #  If using the csv table, remove branch lists
            self.columns.remove("Branch Length")
            self.columns.remove("Branch Intensity")
        else:
          if isinstance(columns, list):
            self.columns = columns

        self.save_type = save_type
        self.save_name = save_name
        self.verbose = verbose
        self.subplot = subplot

    def make_hists(self, num_bins=None, use_prettyplotlib=True):

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

        # Setup subplots if plotting together
        if self.subplot:
          num = len(self.columns)
          if num <= 3:
            ncols = 1
            nrows = num
          elif num <= 8:
            ncols = 2
            nrows = num / 2
          else:  # Max columns right now is 12
            ncols = 3
            nrows = num / 3
          # Check if we need an extra row.
          if num % ncols != 0:
            nrows += 1

          # Make the objects
          fig, axes = plt.subplots(nrows=nrows, ncols=ncols)

          # This is a bit awkward to get the indices, but my matplotlib version
          # doesn't use the same object type as prettyplotlib creates.
          posns = np.indices(axes.shape)
          x, y = posns[0].ravel(), posns[1].ravel()

        # Keep the mean, median, std.
        data_stats = {}
        for i, column in enumerate(self.columns):
          data = self.dataframe[column]
          data = data[np.isfinite(data)]
          if num_bins is None:
            num_bins = np.sqrt(len(data))

          data_stats[column] = [nanmean(data), nanstd(data), nanmedian(data)]

          if self.subplot:
            axes[x[i], y[i]].hist(data, num_bins)
            axes[x[i], y[i]].set_xlabel(column)  # ADD UNITS!
          else:
            fig, axes = plt.subplots(1)
            axes.hist(data, num_bins)
            axes.set_xlabel(column)  # ADD UNITS!

          if self.verbose and not self.subplot:
            print column+" Stats: %s" % (data_stats[column])
            p.show()

          elif not self.subplot:
            fig.savefig(self.save_name+"_"+column+"."+self.save_type)
            p.close()

        if self.subplot:
          p.tight_layout()
          if self.verbose:
            for column in self.columns:
              print column+" Stats: %s" % (data_stats[column])
            p.show()
          else:
            fig.savefig(self.save_name+"_"+hists+"."+self.save_type)


        return self


class ImageAnalysis(object):
    """docstring for ImageAnalysis"""
    def __init__(self, image, mask, skeleton=None, save_name=None, verbose=True):
        super(ImageAnalysis, self).__init__()
        self.image = image
        self.mask = mask
        self.skeleton = skeleton

        self.save_name = save_name
        self.verbose = verbose


    def region_images(self, save_name=None, percentile=80.):
      '''

      Creates saved PDF plots of several quantities/images.

      '''

      if self.verbose:
        pass
      else:
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

      return self

    def stamp_images(self):
      pass