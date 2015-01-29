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


files = [f for f in os.listdir(".") if os.path.isfile(f) and f[-3:] == "csv"
         and f[3:] != "deg"]

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

    kde = make_kde(t["Median"])
    lengths = np.asarray(t["Length"][np.isfinite(t["Intensity"])])
    med = np.asarray(t["Median"][np.isfinite(t["Intensity"])])
    med = med[lengths > 5]
    intens = np.asarray(t["Intensity"][np.isfinite(t["Intensity"])])
    intens = intens[lengths > 5]

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


# p.legend()
# p.show()
