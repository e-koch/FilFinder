# Licensed under an MIT open source license - see LICENSE

import numpy as np
from astropy.table import Table
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as p

try:
  import aplpy
except ImportError:
  print("Optional package aplpy could not be imported.")

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

        if use_prettyplotlib:
            try:
                import prettyplotlib as plt
            except ImportError:
                import matplotlib.pyplot as plt
                use_prettyplotlib = False
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

          data_stats[column] = [np.nanmean(data), np.nanstd(data),
                                np.nanmedian(data)]

          if self.subplot:
            if use_prettyplotlib:
              plt.hist(axes[x[i], y[i]],data, num_bins, grid="y")
            else:
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
            fig.savefig(self.save_name+"_hists."+self.save_type)

    def make_scatter(self, use_prettyplotlib=True, hists=True, num_bins=None):
        '''
        Plot two columns against each other. If self.subplot is enabled,
        all comparisons returned in a triangle collection. Inspiration for
        this form came from the package ![triangle.py](https://github.com/dfm/triangle.py).
        Small snippets to set the labels and figure size came from triangle.py.
        '''

        if use_prettyplotlib:
            try:
                import prettyplotlib as plt
            except ImportError:
                import matplotlib.pyplot as plt
                use_prettyplotlib = False
                print "prettyplotlib not installed. Using matplotlib..."
        else:
            import matplotlib.pyplot as plt

        # Setup subplots if plotting together
        if self.subplot:
          # Make the objects
          num = len(self.columns)
          factor = 2.0 # size of one side of one panel
          lbdim = 0.5 * factor # size of left/bottom margin
          trdim = 0.3 * factor # size of top/right margin
          whspace = 0.05 # w/hspace size
          plotdim = factor * num + factor * (num - 1.) * whspace
          dim = lbdim + plotdim + trdim
          fig, axes = plt.subplots(nrows=num, ncols=num, figsize=(dim, dim))

          lb = lbdim / dim
          tr = (lbdim + plotdim) / dim
          fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr,
                              wspace=whspace, hspace=whspace)

        for i, column1 in enumerate(self.columns):
          for j, column2 in enumerate(self.columns):

            data1 = self.dataframe[column1]
            data2 = self.dataframe[column2]

            # Get rid of nans
            nans = np.isnan(data1) + np.isnan(data2)
            data1 = data1[~nans]
            data2 = data2[~nans]

            if self.subplot:
              ax = axes[i, j]
              if j > i: # Don't bother plotting duplicates
                ax.set_visible(False)
                ax.set_frame_on(False)
              else:

                if j == i: # Plot histograms
                  # Set number of bins
                  if num_bins is None:
                      num_bins = np.sqrt(len(data1))
                  if hists == True:
                    if use_prettyplotlib:
                      plt.hist(ax, data1, num_bins, grid="y")
                    else:
                      ax.hist(data1, num_bins)
                      ax.grid(True)
                  else:
                    ax.set_visible(False)
                    ax.set_frame_on(False)

                  ax.set_xticklabels([])
                  ax.set_yticklabels([])

                if j != i:
                  if use_prettyplotlib:
                    plt.scatter(ax, data2, data1)
                  else:
                    ax.scatter(data2, data1)
                  ax.grid(True)
                  ax.xaxis.set_major_locator(MaxNLocator(5))
                  ax.yaxis.set_major_locator(MaxNLocator(5))

                if i < num - 1:
                    ax.set_xticklabels([])
                else:
                    [l.set_rotation(45) for l in ax.get_xticklabels()]
                    ax.set_xlabel(column2)
                    ax.xaxis.set_label_coords(0.5, -0.3)

                if j > 0:
                    ax.set_yticklabels([])
                else:
                    [l.set_rotation(45) for l in ax.get_yticklabels()]
                    ax.set_ylabel(column1)
                    ax.yaxis.set_label_coords(-0.3, 0.5)

            else:
              if j < i:
                fig, axes = plt.subplots(1)
                if use_prettyplotlib:
                  plt.scatter(axes, data2, data1, grid="y")
                else:
                  axes.scatter(data2, data1)
                  axes.grid(True)
                axes.set_xlabel(column2)  # ADD UNITS!
                axes.set_ylabel(column1)  # ADD UNITS!

                if self.verbose:
                  p.show()
                else:
                  fig.savefig(self.save_name+"_"+column1+"_"+column2+"."+self.save_type)
                  p.close()

        if self.subplot:
          # p.tight_layout()
          if self.verbose:
            p.show()
          else:
            fig.savefig(self.save_name+"_"+"scatter"+"."+self.save_type)


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