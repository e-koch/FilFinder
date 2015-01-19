
'''
Make histogram of surface brightness along the filaments.
'''

from astropy.io.fits import getdata, getheader
from astropy import convolution
from astropy.table import Table
import os
import numpy as np
import matplotlib.pyplot as p
import seaborn as sn
sn.set_context('talk')
sn.set_style("darkgrid")


folders = [f for f in os.listdir("degrade_all") if os.path.isdir(f) and f[-3:] == '350']

dists = {"pipeCenterB59-350": 145.,
         "polaris-350": 150.,
         "ic5146-350": 460.,
         "orionA-S-350": 400.,
         "lupusI-350": 175.,
         "orionB-350": 400.,
         "taurusN3-350": 140.,
         "aquilaM2-350": 260.,
         "orionA-C-350": 400.,
         "perseus04-350": 235.,
         "california_cntr-350": 450.,
         "california_east-350": 450.,
         "california_west-350": 450.,
         "chamaeleonI-350": 170.}

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

all_points = []

for num, fol in enumerate(folders):
    values = np.empty((0))

    # Open the skeleton and the image
    skeleton = getdata("degrade_all/"+fol+"/"+fol+"_skeletons.fits")

    data = Table.read("degrade_all/"+fol+"/"+fol+"_table.fits")
    # skeleton[skeleton > 1] = 1

    img = getdata(fol+".fits") + offsets[fol]
    hdr = getheader(fol+".fits")

    r = 460. / dists[fol]
    if r != 1.:
        conv = np.sqrt(r ** 2. - 1) * \
            (18.5 / np.sqrt(8*np.log(2)) / (np.abs(hdr["CDELT2"]) * 3600.))
        if conv > 1.0:
            kernel = convolution.Gaussian2DKernel(conv)
            img = convolution.convolve(img, kernel, boundary='fill',
                                       fill_value=np.NaN)

    for lab in np.unique(skeleton[np.nonzero(skeleton)]):

        skel_pts = np.where(skeleton == lab)

        bkg = data['Background'][lab-1]
        # amp = data['Amplitude'][lab-1]

        if np.isnan(bkg):
            continue

        points = img[skel_pts] - bkg

        points = points[np.isfinite(points)]
        points = points[points > 0]

        if points.shape == (0,):
            continue

        points = np.log10(points)
        # per5 = points > np.percentile(points, 25)
        # per95 = points < np.percentile(points, 90)
        # points = points[np.logical_and(per5, per95)]

        # points = points[np.logical_and(points > -50, points < 120)]

        values = np.append(values, points)
    all_points.append(values)

# p.tight_layout()
# p.show()

all_points.reverse()

p.subplot(111)
# p.xlim([-50, 125])
sn.violinplot(all_points,
              names=[labels[key] for key in np.sort(labels.keys())[::-1]],
              color=sn.color_palette("GnBu_d"), vert=False)
p.xlabel(r" $\log_{10}$ Surface Brightness (MJy/sr)")
p.tight_layout()
p.xticks([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
p.show()


# SFR density versus filament brightness

sfr = {"aquilaM2-350": 1.80,
       "california_cntr-350": 0.32,
       "california_east-350": 0.32,
       "california_west-350": 0.32,
       "chamaeleonI-350": 3.80,
       "ic5146-350": 0.38,
       "lupusI-350": 0.37,
       "orionA-C-350": 0.48,
       "orionA-S-350": 0.48,
       "orionB-350": 0.084,
       "perseus04-350": 1.32,
       "pipeCenterB59-350": 0.032,
       "polaris-350": 0.0,
       "taurusN3-350": 0.147}  # from Heiderman 2010

median_fil_bright = {}

for i, fold in enumerate(folders):
    if fold in sfr.keys():
        median_fil_bright[fold] = np.nanmedian(all_points[-i-1])

# Covering fraction

cfr = {"aquilaM2-350": 0.340836,
       "california_cntr-350": 0.274288,
       "california_east-350": 0.305174,
       "california_west-350": 0.230324,
       "chamaeleonI-350": 0.377073,
       "ic5146-350": 0.323163,
       "lupusI-350": 0.395093,
       "orionA-C-350": 0.682826,
       "orionA-S-350": 0.439604,
       "orionB-350": 0.374889,
       "perseus04-350": 0.360365,
       "pipeCenterB59-350": 0.159801,
       "polaris-350": 0.319689,
       "taurusN3-350": 0.367821}


from pandas import DataFrame, Series

df = DataFrame([Series(median_fil_bright), Series(sfr), Series(cfr)])

p.scatter(df.ix[0, :], df.ix[1, :])
p.xlabel('log$_{10}$ Median Filamentary Surface Brightness / (MJy/sr)')
p.ylabel(r'$\Sigma$(SFR) (Msol Myr$^{-1}$ pc$^{-2}$)')
p.show()
