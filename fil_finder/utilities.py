#!/usr/bin/python

"""
Utility functions for fil-finder package


"""
try:
  import pyfits
except ImportError:
  import astropy.io.fits as pyfits
import itertools
import numpy as np
from scipy import optimize as op
from skimage import morphology as mo
import operator

def fromfits(fromfile, hdu=0):

	pixelarray,hdr = pyfits.getdata(fromfile,hdu, header=True)
	pixelarray = np.asarray(pixelarray).transpose()

	pixelarrayshape = pixelarray.shape
	print "Shape : (%i,%i)" % (pixelarrayshape[0],pixelarrayshape[1])
	print "BITPIX : %s" % (hdr["BITPIX"])
	print "Array Type :", pixelarray.dtype.name

	return pixelarray,hdr


def removearray(l,arr):
  ind = 0
  size = len(l)
  while ind!=size and not array_equal(l[ind],arr):
    ind += 1
  if ind != size:
    l.pop(ind)
  else:
    raise ValueError('Array not contained in this list.')


def weighted_av(items,weight): # Think this can be simplfied.... or is probably contained in numpy
  # Check for nans
  weight = np.array(weight)[~np.isnan(weight)]
  if len(weight)==0:
    return sum(items)/len(items)
  else:
    items = np.array(items)[~np.isnan(weight)]
    num = sum(items[i] * weight[i] for i in range(len(items)))
    denom = sum(weight[i] for i in range(len(items)))
    return (num/denom) if denom != 0 else None


######################################################################################################################################
### 2D Gaussian Fit Code from http://www.scipy.org/Cookbook/FittingData (functions twodgaussian,moments,fit2dgaussian)
######################################################################################################################################


def twodgaussian(h,cx,cy,wx,wy,b):
  wx = float(wx);wy = float(wy)
  return lambda x,y: h*np.exp(-(((cx-x)/wx)**2. + ((cy-y)/wy)**2.)/2) + b

def moments(data):
  total = data.sum()
  X,Y = indices(data.shape)
  x = (X*data).sum()/total
  y = (Y*data).sum()/total
  col = data[:,int(y)]
  wx = sqrt(abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
  row = data[int(x),:]
  wy = sqrt(abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
  b = abs(np.median(data.ravel()))
  h = data.max()-b
  return h,x,y,wx,wy,b

def fit2dgaussian(data):
  params = moments(data)
  errorfunction = lambda p: np.ravel(twodgaussian(*p)(*indices(data.shape))-data)
  fit,cov = op.leastsq(errorfunction,params,maxfev = (1000*len(data)),full_output=True)[:2]
  if cov is None: ##Bad fit
  	fiterr = np.abs(fit)
  else:
  	fiterr = np.sqrt(np.diag(cov))
  return fit,fiterr

######################################################################################################################################
### Simple fcns used throughout module
######################################################################################################################################


def chunks(l,n):
	return [l[x:x+n] for x in range(0,len(l),n)]

def eight_con():
  return np.ones((3,3))

def distance(x,x1,y,y1):
  return np.sqrt((x-x1)**2.0 + (y-y1)**2.0)

def medial_axis(img):
  return mo.medial_axis(img)

def padwithzeros(vector,pad_width,iaxis,kwargs):
  vector[:pad_width[0]] = 0
  vector[-pad_width[1]:] = 0
  return vector

def round_figs(x,n):
  return round(x,int(n-np.ceil(np.log10(abs(x)))))

def shifter(l,n):
  return l[n:] + l[:n]

def product_gen(n):
  for r in itertools.count(1):
    for i in itertools.product(n,repeat=r):
      yield "".join(i)

def gaussian(x,*p):
  # Peak Height is p[0]
  # Sigma is p[1]
  # Background is p[2]
  return p[0]*np.exp(-1*np.power(x,2) / (2*np.power(p[1],2))) + p[2]

def planck(T,freq):
  return (( 2.0*(6.63*10**(-34))*freq**3)/(9*10**16))* (1/(np.expm1((6.63*10**(-34)*freq)/(1.38*10**(-23)*float(T)))))

def dens_func(B,kappa,I):
  kappa = 100*kappa
  return (I/(B*10**20)) * (1/(kappa))*4787 # into sol.mass/pc


if __name__ == "__main__":
    import sys
    fib(int(sys.argv[1]))



