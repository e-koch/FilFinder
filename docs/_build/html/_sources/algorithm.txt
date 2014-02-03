Algorithm
*********

Here we provide an in-detail explanation of fil_finder.
Essentially, the algorithm consists of three parts:
    * filament detection
    * cleaning and finding lengths
    * deriving widths

**Filament Detection**

Our method of segmenting filaments from images mostly consists of the use of
mathematical morphology.