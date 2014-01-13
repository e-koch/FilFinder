#!/usr/bin/python

from utilities import *
import numpy as np
import scipy.ndimage as nd
import scipy.optimize as op
import matplotlib.pyplot as p
'''
Routines for calculating the widths of filaments.

Contains:


Requires:
		numpy
		ndimage
		scipy.optimize
'''







def dist_transform(labelisofil, offsets, orig_size, pad_size):
	'''

	Recombines the cleaned skeletons from final analysis and takes the Euclidean Distance Transform.
	Since each filament is in an array defined by its own size, the offsets need to be taken into account when
	adding back into a master array.

	NOTE: pad_size must be chosen to be smaller than the smallest filament array shape. The filament will be cut out
		  if it needs to be trimmed to fit within the original image

	'''
	num  = len(labelisofil)


	# Initializing lists
	dist_transform_sep = []


	filclean_all = np.ones(orig_size)
	for n in range(num):
	  x_off,y_off = offsets[n][0] ## This is the coords of the bottom left in the master array
	  x_top,y_top = offsets[n][1]

	  ## Now check if padding will put the array outside of the original array size
	  excess_x_top =  x_top - orig_size[0] + 2*pad_size

	  excess_y_top =  y_top - orig_size[1] + 2*pad_size

	  pad_labelisofil = np.pad(labelisofil[n],pad_size,padwithzeros) # Increase size of arrays for better radial fits

	  if excess_x_top > 0:
	  	pad_labelisofil = pad_labelisofil[:-excess_x_top,:]
	  	print "REDUCED FILAMENT %s TO FIT IN ORIGINAL ARRAY" %(n)
	  if excess_y_top > 0:
	  	pad_labelisofil = pad_labelisofil[:,:-excess_y_top]
	  	print "REDUCED FILAMENT %s TO FIT IN ORIGINAL ARRAY" %(n)

	  if x_off<0:
	  	excess_x_bottom =  (x_off - 2*pad_size) * (-1)
	  	pad_labelisofil = pad_labelisofil[excess_x_bottom:,:]
	  	print "REDUCED FILAMENT %s TO FIT IN ORIGINAL ARRAY" %(n)

	  if y_off<0:
	  	excess_y_bottom =  (y_off - 2*pad_size) * (-1)
	  	pad_labelisofil = pad_labelisofil[:,excess_y_bottom:]
	  	print "REDUCED FILAMENT %s TO FIT IN ORIGINAL ARRAY" %(n)

	  x,y = np.where(pad_labelisofil>=1)
	  for i in range(len(x)):
	    pad_labelisofil[x[i],y[i]]=1
	    filclean_all[x[i]+ x_off - pad_size,y[i]+ y_off - pad_size]=0
	  dist_transform_sep.append(nd.distance_transform_edt(np.logical_not(pad_labelisofil)))

	dist_transform_all = nd.distance_transform_edt(filclean_all) # Distance Transform of all cleaned filaments

	return dist_transform_all,dist_transform_sep




def gauss_width(img,dist_transform_all,dist_transform_sep,img_beam,img_scale,offsets,verbose=False):
	# Fits a Gaussian to the radial profile of a filament using the output from dist_transform, the image, beam_width and spatial scale
	num = len(dist_transform_sep)
	#Initialize lists
	fits = []
	fit_errors = []
	widths = []

	param0 = (15.,2.,1.)

	# Try to scale region looked at by image size
	if img.shape[0]>img.shape[1]:
		near_region = img.shape[1]/4.
	else:
		near_region = img.shape[0]/4.

	for n in range(num):
		width_value = []
		width_distance = []
		x,y = np.where(dist_transform_sep[n]<near_region)
		x_full = x + offsets[n][0][0] ## Transform into coords of master image
		y_full = y + offsets[n][0][1]
	  	num_nan = 0
	  	for i in range(len(x)):
			if dist_transform_sep[n][x[i],y[i]]<=dist_transform_all[x_full[i],y_full[i]] and np.isnan(img[x_full[i],y_full[i]])==False:
				# Check overall distance transform to make sure pixel belongs to proper filament
				#print i
				width_value.append(img[x_full[i],y_full[i]])
				width_distance.append(dist_transform_sep[n][x[i],y[i]])
			else: num_nan+=1
		# Binning
	  	av_dists = [];av_val = [];ave_dist = [];ave_val = []
	  	for j in range(150):
			val_bin = [];dists_bin = []
			for i in width_distance:
				if i>=j and i<j+1:
					dists_bin.append(i)
					val_bin.append(abs(width_value[width_distance.index(i)]))
			if len(val_bin)==0 or len(dists_bin)==0: pass
			else:
				av_d = sum(dists_bin)/len(dists_bin) * img_scale
				av_v = sum(val_bin)/len(val_bin)
				av_dists.append(av_d)
				av_val.append(av_v)
		# Attempt to find initial params assuming data is Gaussian-ish
		if len(av_val)>0:
			param = (np.max(av_val),np.sqrt(np.var(av_dists)),np.min(av_val))
		else:
			print "No points to fit"
			param = param0
	  	try:
			opts,cov = op.curve_fit(gaussian,av_dists,av_val,p0=param,maxfev=100*(len(width_value)+1))
			fits.append(opts)
			fit_errors.append(np.sqrt(np.diag(cov)))
	  	except:
	  		print "Fit Fail"
			opts,cov = param,None
			fits.append(opts)
			fit_errors.append(cov)
		deconv = (2.35*abs(opts[1]))**2. - img_beam**2.
		if deconv>0:
			widths.append(np.sqrt(deconv))
		else:
			widths.append("Neg. FWHM")
		if verbose:
			print param
			print opts
			p.plot(av_dists,av_val,"kD",np.linspace(0,1,100),gaussian(np.linspace(0,1,100),*opts),"r")
			p.xlabel(r'Radial Distance (pc)')
			p.ylabel(r'Integrated Intensity ( $\frac{K km}{s}$ )')
			p.grid(True)
			p.show()
		param = None # Reset param for next filament
	return widths,fits,fit_errors


def cyl_model(img,dist_transform_all,dist_transform_sep,img_beam,img_scale,img_freq):
	# Fits the radial profile of filament to a cylindrical model
	num = len(dist_transform_sep)
	p0 = (1e20,0.03,2.)

	# Initialize Lists
	widths = []
	fits = []
	fit_errors =[]


	for n in range(num):
		width_value = []
		width_distance = []
		x,y = np.where(dist_transform_sep[n]<50.0)
		for i in range(len(x)):
			if dist_transform_sep[n][x[i],y[i]]>dist_transform_all[x[i],y[i]]: pass # Check overall distance transform to make sure pixel belongs to proper filament
			else:
				width_value.append(img[x[i],y[i]])
				width_distance.append(dist_transform_sep[n][x[i],y[i]])
		# Binning
	  	av_dists = [];av_val = [];ave_dist = [];ave_val = []
	  	for j in range(50):
			val_bin = [];dists_bin = []
			for i in width_distance:
				if i>=j and i<j+1:
					dists_bin.append(i)
					val_bin.append(abs(width_value[width_distance.index(i)]))
		if len(val_bin)==0 or len(dists_bin)==0: pass
		else:
			av_d = sum(dists_bin)/len(dists_bin)
			av_v = sum(val_bin)/len(val_bin)
			av_dists.append(av_d * img_scale)
			av_val.append(dens_func(planck(20.,img_freq),0.2,av_v*img_scale**-1)*(5.7*10**19))
	    # Fitting
		try:
			fit,cov = op.curve_fit(cyl_model,av_dists,av_val,p0=p0,maxfev=100*(len(av_dists)+1))
			fits.append(fit)
			fit_errors.append(np.sqrt(np.diag(cov)))
		except:
			fit,cov = p0,None
			fits.append(fit)
			fit_errors.append(cov)
		deconv = (2.35*abs(fit[1])**2.) - img_beam**2.
		if deconv>0:
			widths.append(np.sqrt(deconv))
		else:
			widths.append("Neg. FWHM")

	return widths,fits,fit_errors



if __name__ == "__main__":
    import sys
    fib(int(sys.argv[1]))


































