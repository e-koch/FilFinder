#!/usr/bin/python

from fil_finder import *

img,hdr = fromfits("/srv/astro/erickoch/gould_belt/chamaeleonI-250.fits")

## Utilize fil_finder_2D class
## See filfind_class.py for inputs
test = fil_finder_2D(img, hdr, 15.1, 30, 5, 10, 95 ,distance=160)#,region_slice=[620,1400,430,1700])
# test.create_mask(verbose=True)
test.run(verbose=False, save_name="chamaeleonI-250", save_plots=True) ## Run entire algorithm


