# Licensed under an MIT open source license - see LICENSE

'''
KS p-values for different properties.
'''

import numpy as np
from pandas import read_csv
import matplotlib.pyplot as p
import numpy as np
import seaborn as sn
sn.set_context('talk')
sn.set_style('ticks')
# sn.mpl.rc("figure", figsize=(7, 9))

# Widths

widths = read_csv("width_ks_table_pvals.csv")

widths.index = widths["Unnamed: 0"]
del widths["Unnamed: 0"]

widths_arr = np.asarray(widths)

widths_arr[np.arange(0, 14), np.arange(0, 14)] = 1.0

widths_arr = -np.log10(widths_arr)

# p.figure(figsize=(12, 10))
p.subplot(111)
# p.xlabel("Widths")
p.imshow(widths_arr, origin='lower', cmap='binary', interpolation='nearest')
p.xticks(np.arange(0, 14), widths.columns, rotation=90)
# p.xticks(np.arange(0, 14), [], rotation=90)
p.yticks(np.arange(0, 14), widths.columns)

# p.figtext(0.05, 0.95, "a)", fontsize=20)

cb = p.colorbar()
cb.set_label(r'$-\log_{10}$ p-value')
cb.solids.set_edgecolor("face")
p.tight_layout()

p.show()
# Curvature

# curve = read_csv("curvature_ks_table_pvals.csv")

# curve.index = curve["Unnamed: 0"]
# del curve["Unnamed: 0"]

# curve_arr = np.asarray(curve)

# curve_arr[np.arange(0, 14), np.arange(0, 14)] = 1.0

# curve_arr = -np.log10(curve_arr)

# # p.figure(figsize=(12, 10))
# p.subplot(212)
# # p.xlabel("Curvature")
# p.imshow(curve_arr, interpolation='nearest', origin='lower', cmap='binary')
# p.xticks(np.arange(0, 14), curve.columns, rotation=90)
# p.yticks(np.arange(0, 14), curve.columns)

# p.figtext(0.05, 0.55, "b)", fontsize=20)

# cb = p.colorbar()
# cb.set_label(r'$-\log_{10}$ p-value')
# cb.solids.set_edgecolor("face")

# p.tight_layout()

# p.show()
