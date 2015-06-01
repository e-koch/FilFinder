# Licensed under an MIT open source license - see LICENSE

'''
Examine the orientation distributions of the Gould Belt data
'''

import numpy as np
from scipy.stats import gaussian_kde
from pandas import read_csv, DataFrame
import matplotlib.pyplot as p
import os


def make_kde(y):

    # Extend y through to -pi and 2 pi.
    cont_data = np.append(y - np.pi, y)
    cont_data = np.append(y + np.pi, cont_data)

    kde = gaussian_kde(cont_data)
    kde.covariance_factor = lambda: .05
    kde._compute_covariance()
    return kde


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

dists = {"pipeCenterB59-350": 145.,
         "polaris-350": 150.,
         "ic5146-350": 460.,
         "orionA-S-350": 400.,
         "lupusI-350": 150.,
         "orionB-350": 400.,
         "taurusN3-350": 140.,
         "aquilaM2-350": 260.,
         "orionA-C-350": 400.,
         "perseus04-350": 235.,
         "california_cntr-350": 450.,
         "california_east-350": 450.,
         "california_west-350": 450.,
         "chamaeleonI-350": 170.}

# files = [f for f in os.listdir(".") if os.path.isfile(f) and f[-3:] == "csv"
#          and f[3:] != "deg"]

files = [f+"/"+f+"_rht_branches.csv" for f in os.listdir(".")
         if os.path.isdir(f)]

x = np.linspace(-np.pi/2, np.pi/2, 1000)
cols = ["k-", "b-", "g-", "r-", "c-", "m-",
        "k--", "b--", "g--", "r--", "c--", "m--",
        "k-.", "b-.", "g-.", "r-."]


keepers = {"pipeCenterB59-350/pipeCenterB59-350_rht_branches.csv": "Pipe",
           "california_west-350/california_west-350_rht_branches.csv": "California West",
           "chamaeleonI-350/chamaeleonI-350_rht_branches.csv": "Chamaeleon",
           "california_east-350/california_east-350_rht_branches.csv": "California East",
           "orionA-S-350/orionA-S-350_rht_branches.csv": "Orion-A South",
           "aquilaM2-350/aquilaM2-350_rht_branches.csv": "Aquila"}

for i, (f, col) in enumerate(zip(np.sort(keepers.keys()), cols)):
    print f
    t = read_csv(f)

    lengths = np.asarray(t["Length"][np.isfinite(t["Intensity"])])
    med = np.asarray(t["Median"][np.isfinite(t["Intensity"])])
    med = med[lengths > 10]
    intens = np.asarray(t["Intensity"][np.isfinite(t["Intensity"])])
    intens = intens[lengths > 10]
    kde = make_kde(med)

    ax = p.subplot(3, 2, i+1)
    p.title(keepers[f], fontsize=15)
    p.plot(x, 3*kde(x), "k-", label=f[:-21].replace("_", " "), alpha=0.9,
           linewidth=3)
    p.hist(med, bins=15, normed=1, alpha=1.0, color="gray")
           # weights=intens)

    ax.set_xticks([-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2])
    ax.set_yticks([])
    if not i >= 4:
        ax.set_xticklabels([])
    else:
        ax.set_xticklabels([r"$-\pi/2$", "", 0, "", r"$\pi/2$"], fontsize=13)

    ax.set_xlim([-np.pi/2, np.pi/2])
p.show()


# Examine the curvature distributions

def ks_table(param_dict):
    '''
    '''

    from scipy.stats import ks_2samp

    pvals = np.zeros((len(param_dict.keys()), len(param_dict.keys())))
    stats = np.zeros((len(param_dict.keys()), len(param_dict.keys())))

    for i, key in enumerate(np.sort(param_dict.keys())[::-1]):
        print key
        for j, key2 in enumerate(np.sort(param_dict.keys())[::-1]):
            if i == j:
                pvals[i, j] = 0
                stats[i, j] = 0
            else:
                shape1 = param_dict[key].shape[0]
                shape2 = param_dict[key2].shape[0]

                samps1 = param_dict[key]
                samps2 = param_dict[key2]

                if shape1 != shape2:
                    if shape1 > shape2:
                        samps1 = np.random.choice(samps1, shape2)
                    else:
                        samps2 = np.random.choice(samps2, shape1)

                values = ks_2samp(samps1, samps2)
                pvals[i, j] = values[1]
                stats[i, j] = values[0]

    return stats, pvals


curvature = {}

for fil in files:

    t = read_csv(fil)
    key = fil.split("/")[0]
    curvature[key] = np.asarray(t['IQR'])

    # print key
    # p.hist(curvature[key], bins=50)
    # p.show()

ordered_labels = []
for key in np.sort(curvature.keys())[::-1]:
    ordered_labels.append(labels[key])

curv_kstest = ks_table(curvature)

curv_df = DataFrame(curv_kstest[1], index=ordered_labels,
                    columns=ordered_labels)

curv_df.to_csv('curvature_branches_ks_table_pvals.csv')
