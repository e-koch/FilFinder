# Licensed under an MIT open source license - see LICENSE

'''
Examine the orientation distributions of the Gould Belt data
'''

import numpy as np
from scipy.stats import gaussian_kde
from pandas import read_csv
import matplotlib.pyplot as p
import os


def make_kde(y):

    # Extend y through to -pi and 2 pi.
    cont_data = np.append(y - np.pi, y)
    cont_data = np.append(y + np.pi, cont_data)

    kde = gaussian_kde(cont_data)
    kde.covariance_factor = lambda : .05
    kde._compute_covariance()
    return kde


files = [f for f in os.listdir(".") if os.path.isfile(f) and f[-3:]== "csv"
         and f[3:] != "deg"]
print len(files)
x = np.linspace(0, np.pi, 1000)
cols = ["k-", "b-", "g-", "r-", "c-", "m-",
        "k--", "b--", "g--", "r--", "c--", "m--",
        "k-.", "b-.", "g-.", "r-."]


keepers = {"pipeCenterB59-250_rht_branches.csv": "Pipe",
           "california_west-250_normed_rht_branches.csv": "California West",
           "chamaeleonI-250_normed_rht_branches.csv": "Chamaeleon",
           "california_east-250_normed_rht_branches.csv": "California East",
           "orionA-S-250_rht_branches.csv": "Orion-A South",
           "aquilaM2-250_rht_branches.csv": "Aquila"}

for i, (f, col) in enumerate(zip(keepers.keys(), cols)):
    print f
    t = read_csv(f)

    kde = make_kde(t["Median"])
    lengths = np.asarray(t["Length"][np.isfinite(t["Intensity"])])
    med = np.asarray(t["Median"][np.isfinite(t["Intensity"])])
    med = med[lengths > 5]
    intens = np.asarray(t["Intensity"][np.isfinite(t["Intensity"])])
    intens = intens[lengths > 5]

    ax = p.subplot(3, 2, i+1)
    p.title(keepers[f])
    p.plot(x, 3*kde(x), "k-", label=f[:-21].replace("_", " "), alpha=0.9,
           linewidth=3)
    p.hist(med, bins=15, normed=1, alpha=1.0, color="gray")
           # weights=intens)

    ax.set_xticks([0, np.pi/4, np.pi/2, 0.75*np.pi, np.pi])
    ax.set_yticks([])
    if not i >= 4:
        ax.set_xticklabels([])
    else:
        ax.set_xticklabels([0, "", r"$\pi/2$", "", r"$\pi$"], fontsize=13)

    ax.set_xlim([0.0, np.pi])
p.show()


# p.legend()
# p.show()
