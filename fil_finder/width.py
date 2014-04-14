#!/usr/bin/python

from utilities import *
import numpy as np
import scipy.ndimage as nd
import scipy.optimize as op
from scipy.integrate import quad
import matplotlib.pyplot as p
import copy
'''
Routines for calculating the widths of filaments.

Contains:


Requires:
		numpy
		ndimage
		scipy.optimize
'''







def dist_transform(labelisofil, offsets, orig_size, pad_size, length_threshold):
	'''

	Recombines the cleaned skeletons from final analysis and takes the
	Euclidean Distance Transform of each. Since each filament is in an
	array defined by its own size, the offsets need to be taken into account
	when adding back into the master array.

	Parameters
	----------

	labelisofil : list
				  Contains arrays of the cleaned individual skeletons

	offsets : list
			  The output from isolatefilaments during the segmentation
			  process. Contains the indices where each skeleton was cut
			  out of the original array.

	orig_size : tuple
				The shape of the original image.

	pad_size : int
			   The size to pad each skeleton array with. If the edges go
			   beyond the original image's size, they are trimmed to size.

	length_threshold : int
					   Threshold length for filaments. Used as a check
					   in making the final filament map.

	Returns
	-------

	dist_transform_all : numpy.ndarray
						 A Euclidean Distance Transform of all of the skeletons
						 combined.

	dist_transform_sep : list
						 Contains the Euclidean Distance Transform of each
						 individual skeleton.

	'''
	num  = len(labelisofil)


	# Initializing lists
	dist_transform_sep = []


	filclean_all = np.ones(orig_size)
	for n in range(num):
	  x_off,y_off = offsets[n][0] ## This is the coords of the bottom left in the master array
	  x_top,y_top = offsets[n][1]

	  ## Now check if padding will put the array outside of the original array size
	  excess_x_top =  x_top - orig_size[0]

	  excess_y_top =  y_top - orig_size[1]

	  pad_labelisofil = copy.copy(labelisofil[n]) # Increase size of arrays for better radial fits

	  if excess_x_top > 0:
	  	pad_labelisofil = pad_labelisofil[:-excess_x_top,:]
	  	print "REDUCED FILAMENT %s TO FIT IN ORIGINAL ARRAY" %(n)
	  if excess_y_top > 0:
	  	pad_labelisofil = pad_labelisofil[:,:-excess_y_top]
	  	print "REDUCED FILAMENT %s TO FIT IN ORIGINAL ARRAY" %(n)

	  if x_off<0:
	  	pad_labelisofil = pad_labelisofil[-x_off:,:]
	  	x_off = 0
	  	print "REDUCED FILAMENT %s TO FIT IN ORIGINAL ARRAY" %(n)

	  if y_off<0:
	  	pad_labelisofil = pad_labelisofil[:,-y_off:]
	  	y_off = 0
	  	print "REDUCED FILAMENT %s TO FIT IN ORIGINAL ARRAY" %(n)

	  x,y = np.where(pad_labelisofil>=1)
	  for i in range(len(x)):
	    pad_labelisofil[x[i],y[i]]=1
	    filclean_all[x[i]+ x_off,y[i]+ y_off]=0
	  dist_transform_sep.append(nd.distance_transform_edt(np.logical_not(pad_labelisofil)))

	dist_transform_all = nd.distance_transform_edt(filclean_all) # Distance Transform of all cleaned filaments

	return dist_transform_all, dist_transform_sep, filclean_all


def cyl_model(distance, rad_profile, img_beam):
	'''

	Fits the radial profile of filament to a cylindrical model (see Arzoumanian et al. (2011)).

	'''

	p0 = (np.max(rad_profile), 0.1, 2.0)

	A_p_func = lambda u,p: (1+u**2.)**(-p/2.)

	def model(r, *params):
		peak_dens, r_flat, p = params[0], params[1], params[2]

		A_p = quad(A_p_func, -np.inf, np.inf, args=(p))[0]

		return A_p * (peak_dens * r_flat)/(1+r/r_flat)**((p-1)/2.)

	try:
		fit,cov = op.curve_fit(model, distance, rad_profile, p0=p0, maxfev=100*(len(distance)+1))
		fit_errors = np.sqrt(np.diag(cov))
	except:
		fit,cov = p0,None
		fit_errors = cov

	# ## Deconvolve the width with the beam size.
	# deconv = (2.35*abs(fit[1])**2.) - img_beam**2.
	# if deconv>0:
	# 	fit[1] = np.sqrt(deconv)
	# else:
	# 	fit[1] = "Neg. FWHM"
	fail_flag = False
	if cov==None or (fit_errors>fit).any():
		fail_flag = True

	parameters = [r"$\pho_c$", "r_{flat}", "p"]

	return fit, fit_errors, model, parameters, fail_flag

def gauss_model(distance, rad_profile, weights, img_beam):
	'''
		Fits a Gaussian to the radial profile of each filament by comparing
	the intensity profile from the center of the skeleton using the output
	of dist_transform. The FWHM width of the Gaussian is deconvolved with
	the beam-size of the image. Errors are estimated from the trace of
	the covariance matrix of the fit.
	'''

	p0 = (np.max(rad_profile), 0.1, np.min(rad_profile))
	parameters = ["Amplitude", "Width", "Background", "FWHM"]

	def gaussian(x,*p):
		'''

		Parameters
		**********

		x : list or numpy.ndarray
			1D array of values where the model is evaluated

		p : tuple
			Components are:
				* p[0] Amplitude
				* p[1] Width
				* p[2] Background

		'''
		return (p[0]-p[2])*np.exp(-1*np.power(x,2) / (2*np.power(p[1],2))) + p[2]

	try:
		fit, cov = op.curve_fit(gaussian, distance, rad_profile, p0=p0, \
							maxfev=100*(len(distance)+1), sigma=weights)
		fit_errors = np.sqrt(np.diag(cov))
	except RuntimeError:
		print "curve_fit failed."
		fit, fit_errors = p0, None
		return fit, fit_errors, gaussian, parameters, True


	## Deconvolve the width with the beam size.
	deconv = (2.35*fit[1])**2. - img_beam**2.
	if deconv>0:
		fit_errors = np.append(fit_errors, (2.35*fit[1]*fit_errors[1])/deconv)
		fit = np.append(fit, np.sqrt(deconv))
	else:
		fit = np.append(fit, img_beam) ## If you can't devolve it, set it to minimum, which is the beam-size.
		fit_errors = np.append(fit_errors, 0.0)

	fail_flag = False
	if fit_errors==None or fit[0]<fit[2]: #or (fit_errors>fit).any():
		fail_flag = True

	return fit, fit_errors, gaussian, parameters, fail_flag

def lorentzian_model(distance, rad_profile, img_beam):
	'''
		Fits a Gaussian to the radial profile of each filament by comparing
	the intensity profile from the center of the skeleton using the output
	of dist_transform. The FWHM width of the Gaussian is deconvolved with
	the beam-size of the image. Errors are estimated from the trace of
	the covariance matrix of the fit.
	'''

	p0 = (np.max(rad_profile), 0.1, np.min(rad_profile))

	def lorentzian(x,*p):
		'''

		Parameters
		**********

		x : list or numpy.ndarray
			1D array of values where the model is evaluated

		p : tuple
			Components are:
				* p[0] Amplitude
				* p[1] FWHM Width
				* p[2] Background

		'''
		return (p[0]-p[2])*(0.5 * p[1])**2 / ((0.5 * p[1])**2 + x**2) + p[2]

	try:
		fit, cov = op.curve_fit(lorentzian, distance, rad_profile, p0=p0, maxfev=100*(len(distance)+1))
		fit_errors = np.sqrt(np.diag(cov))
	except:
		fit, fit_errors = p0, None

	fit = list(fit)
	## Deconvolve the width with the beam size.
	deconv = fit[1]**2. - img_beam**2.
	if deconv>0:
		fit[1] = np.sqrt(deconv)
	else:
		fit[1] = img_beam ## If you can't devolve it, set it to minimum, which is the beam-size.

	fail_flag = False
	if fit_errors==None or (fit_errors>fit).any():
		fail_flag = True

	parameters = ["Amplitude", "Width", "Background"]

	return fit, fit_errors, lorentzian, parameters, fail_flag

def radial_profile(img, dist_transform_all, dist_transform_sep, offsets,\
				   img_scale, bins=None, bintype="linear", weighting="number"):
	'''
	Parameters
	----------

	img : numpy.ndarray
		  The original image.

	dist_transform_all : numpy.ndarray
						 The distance transform of all the skeletons.
						 Outputted from dist_transform.

	dist_transform_sep : list
						 The distance transforms of each individual skeleton.
						 Outputted from dist_transform.


	offsets : list
	   		The output from isolatefilaments during the segmentation
			process. Contains the indices where each skeleton was cut
			out of the original array.
	'''

	width_value = []
	width_distance = []
	x,y = np.where(np.isfinite(dist_transform_sep))
	x_full = x + offsets[0][0] ## Transform into coordinates of master image
	y_full = y + offsets[0][1]

  	for i in range(len(x)):
		# Check overall distance transform to make sure pixel belongs to proper filament
		if dist_transform_sep[x[i],y[i]]<=dist_transform_all[x_full[i],y_full[i]]:
			if img[x_full[i],y_full[i]]!=0.0 and np.isfinite(img[x_full[i],y_full[i]]):
				width_value.append(img[x_full[i],y_full[i]])
				width_distance.append(dist_transform_sep[x[i],y[i]])
	width_value = np.asarray(width_value)
	width_distance = np.asarray(width_distance)
	# Binning
	if bins is None:
		nbins = np.sqrt(len(width_value))
		maxbin = np.max(width_distance)
		if bintype is "log":
			bins = np.logspace(0,np.log10(maxbin),nbins+1) # bins must start at 1 if logspaced
		elif bintype is "linear":
			bins = np.linspace(0,maxbin,nbins+1)

	whichbins = np.digitize(width_distance, bins)
	bin_centers = (bins[1:]+bins[:-1])/2.0
	radial_prof = np.array([np.median(width_value[(whichbins==bin)]) for bin in range(1,int(nbins)+1)])

	if weighting=="number":
		weights = np.array([whichbins[whichbins==bin].sum() for bin in range(1,int(nbins)+1)])
	elif weighting=="var":
		weights = np.array([np.nanvar(width_value[whichbins==bin]) for bin in range(1,int(nbins)+1)])
		weights[np.isnan(weights)] = 0.0 # Empty bins

	# Ignore empty bins
	radial_prof = radial_prof[weights>0]
	bin_centers = bin_centers[weights>0]
	weights = weights[weights>0]

	return bin_centers * img_scale, radial_prof, weights

def medial_axis_width(medial_axis_distance, mask, skeleton):
	'''
	Estimate the filament width using the distance transform from the
	medial axis transform.

	Parameters
	**********

	medial_axis_distance : numpy.ndarray
						   Distance Transform

	mask : numpy.ndarray
		   Mask of filaments

	skeleton : numpy.ndarray
			   Skeletonized mask.

	Returns
	*******

	av_widths : numpy.array
				1D array of the average widths

	'''

	labels, n = nd.label(skeleton, eight_con())
	av_widths = 2. * nd.sum(medial_axis_distance, labels, range(1, n+1)) / nd.sum(skeleton, labels, range(1, n+1))

	return av_widths
