# Licensed under an MIT open source license - see LICENSE

"""
**THESE ARE NOT CURRENTLY USED IN THE ALGORITHM**

Finding/Analyzing Cores Routines for fil-finder

Contains: abs_thresh
subtract_cores

"""

# from .utilities import *

# import numpy as np
# from scipy.ndimage import label


# def abs_thresh(arr,thresh_value,img_scale,img_freq):
#     arr = np.array(arr)

#     conversion = planck(20.,img_freq)*1e20*20. * (2.32 * 1.67e-30)*img_scale**-1 #Converted to MJy/sr
#     core_thresh = conversion * thresh_value

#     core_array = arr > core_thresh

#     return core_array

# def subtract_cores(core_array,full_output=False):

#     lab_core,num_core = label(core_array,eight_con())

#     for n in range(num_core):
#         x,y = np.where(lab_core==n+1)

#     if len(x)>10:
#         filarr = core_array[min(x):max(x),min(y):max(y)]

#     try:
#         fit,err = fit2dgaussian(filarr)
#         for i in range(len(x)):
#             core_array[x[i],y[i]] = fit[-1]
#     ## Set Core region to fit background, should be approx of rest of filament

#     except:
#         print("Fit Failed on Core %s" % (n))


#     if full_output:
#         return core_array, fit, err
#     else:
#         return core_array


# if __name__ == "__main__":
#     import sys
#     fib(int(sys.argv[1]))