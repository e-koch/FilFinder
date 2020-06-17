
# '''
# Run in python3 only!
# '''

# from pynoise.noisemodule import RidgedMulti
# from pynoise.noiseutil import noise_map_plane
# from astropy.io import fits

# ridge = RidgedMulti()

# width = 256
# height = 256

# data = noise_map_plane(width=width, height=height, source=ridge, upper_x=3,
#                        upper_z=3, seamless=False)

# image = data.reshape((width, height))

# header = fits.Header()
# header['CDELT2'] = 1 / 3600.
# header['CDELT1'] = 1 / 3600.

# hdu = fits.PrimaryHDU(image, header)
# hdu.writeto("test_ridgemulti.fits")
