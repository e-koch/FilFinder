
import numpy as np
from scipy import ndimage as nd
from astropy.io import fits
import astropy.units as u

from ..width_profiles.profile import _line_profile_coordinates


def generate_filament_model(shape=100, pad_size=0,
                            max_dist=100,
                            theta=0.0, width=10.0,
                            amplitude=1.0, background=0.0,
                            return_hdu=False,
                            phys_pixscale=0.01 * u.pc,
                            phys_beamsize=0.03 * u.pc,
                            distance=250 * u.pc):

    if theta >= np.pi / 2. or theta <= - np.pi / 2.:
        raise ValueError("theta must be between -pi/2 and pi/2")

    yy, xx = np.mgrid[-shape // 2:shape // 2,
                      -shape // 2:shape // 2]

    centers = np.zeros(yy.shape, dtype=bool)
    centers[np.abs(yy - np.tan(theta) * xx) < 1.0] = True

    # Keep positions within the max distance
    cent_point = np.ones(yy.shape, dtype=bool)
    cent_point[yy.shape[0] // 2 + 1, yy.shape[1] // 2 + 1] = False
    mid_dist = nd.distance_transform_edt(cent_point)

    centers[mid_dist > max_dist] = False

    if pad_size > 0:
        centers = np.pad(centers, [(pad_size,) * 2, (pad_size,) * 2],
                         mode='constant', constant_values=0)

    radii = nd.distance_transform_edt(~centers)

    filament = amplitude * np.exp(- (radii ** 2) / (2 * width ** 2)) + \
        background

    if not return_hdu:
        return filament, centers

    pixel_scale = (phys_pixscale / distance).value * u.rad
    beam_size = (phys_beamsize / distance).value * u.rad

    header = create_image_header(pixel_scale, beam_size, filament.shape,
                                 1.e11 * u.Hz, u.K)

    return fits.PrimaryHDU(filament, header), centers


# def make_skeleton_mask(shape=(256, 256), nfils=1):
#     '''
#     Generate a skeleton mask defining positions of filaments.
#     '''


#     mask = np.zeros(shape, dtype=bool)

#     for _ in range(nfils):

#         # Sample random start and end points
#         start = (np.random.randint(shape[0]))

def create_image_header(pixel_scale, beamfwhm, imshape,
                        restfreq, bunit):
    '''
    Create a basic FITS header for an image.

    Adapted from: https://github.com/radio-astro-tools/uvcombine/blob/master/uvcombine/tests/utils.py

    Parameters
    ----------
    pixel_scale : `~astropy.units.Quantity`
        Angular scale of one pixel
    beamfwhm : `~astropy.units.Quantity`
        Angular size for a circular Gaussian beam.
    imshape : tuple
        Shape of the data array.
    restfreq : `~astropy.units.Quantity`
        Rest frequency of the spectral line.
    bunit : `~astropy.units.Unit`
        Unit of intensity.

    Returns
    -------
    header : fits.Header
        FITS Header.
    '''

    header = {'CDELT1': -(pixel_scale).to(u.deg).value,
              'CDELT2': (pixel_scale).to(u.deg).value,
              'BMAJ': beamfwhm.to(u.deg).value,
              'BMIN': beamfwhm.to(u.deg).value,
              'BPA': 0.0,
              'CRPIX1': imshape[0] / 2.,
              'CRPIX2': imshape[1] / 2.,
              'CRVAL1': 0.0,
              'CRVAL2': 0.0,
              'CTYPE1': 'GLON-CAR',
              'CTYPE2': 'GLAT-CAR',
              'CUNIT1': 'deg',
              'CUNIT2': 'deg',
              'CRPIX3': 1,
              'RESTFRQ': restfreq.to(u.Hz).value,
              'BUNIT': bunit.to_string(),
              }

    return fits.Header(header)
