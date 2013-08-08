#!/usr/bin/python

from fil_finder import *

img,hdr = fromfits("/home/eric/gould_belt/chamaeleonI-250.fits")

## Utilize fil_finder_2D class
## See filfind_class.py for inputs
test = fil_finder_2D(img, hdr, 15.1, 50, 50, 10, 5, 10,distance=150)#,region_slice=[620,1400,430,1700])

test.run(verbose=True) ## Run entire algorithm


