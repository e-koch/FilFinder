#!/usr/bin/python


from utilities import *
import numpy as np


'''
Routines for calculating the curvature of filaments

Contains:
		curve
		av_curvature

Requires:
		numpy

'''
















def curve(pts):
  #The average curvature of the filament is found using the Menger curvature. The formula relates the area of the triangle created by the three points and the distance between the points. The formula is given as 4*area/|x-y||y-z||z-x|=curvature. The curvature is weighted by the euclidean length of the three pixels.
  lenn = len(pts)
  kappa = [];seg_len = []
  for i in range(lenn-2):
    x1 = pts[i][0];y1 = pts[i][1]
    x2 = pts[i+1][0];y2 = pts[i+1][1]
    x3 = pts[i+2][0];y3 = pts[i+2][1]
    num = abs(2*((x2-x1)*(y2-y1)+(y3-y2)*(x3-x2)))
    den = np.sqrt( (pow((x2-x1),2) + pow((y2-y1),2)) * (pow((x3-x2),2)+pow((y3-y2),2 ))* (pow((x1-x3),2)+pow((y1-y3),2) ) )
    if ( den == 0 ):
      kappa.append(0)
    else:
      kappa.append(num/den)
    seg_len.append(fil_length([[pts[i],pts[i+1],pts[i+2]]],initial=True)[0])      
  numer = sum(kappa[i] * seg_len[i][0] for i in range(len(kappa)))
  denom = sum(seg_len[i][0] for i in range(len(seg_len))) 
  if denom!= 0:
    return numer/denom
  else:
    print n
    print pts
    raise ValueError('Sum of length segments is zero.')
    #return np.NaN



def av_curvature(finalpix,ra_picks=1000,seed=500):
	#Calculates the average curvature using final skeleton points from final_lengths
	# Picks 3 random points on the filament and calculates the curvature. The average is taken of 100 picks.
	import numpy.random as ra
	seed = int(seed)
	ra.seed(seed=int(seed))
	ra_picks = int(ra_picks)

	#Initialize lists
	curvature = []

	for n in range(len(finalpix)):
		if len(finalpix[n])>3:
			trials = []
			for _ in range(ra_picks):
				picks = ra.choice(len(finalpix[n])+1,3,replace=False)
				points = [finalpix[n][i] for i in picks]
				trials.append(curve(points))
			curvature.append(np.mean(trials))
		else:
			curvature.append("Fail")
	return curvature



if __name__ == "__main__":
    import sys
    fib(int(sys.argv[1]))


