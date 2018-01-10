# Licensed under an MIT open source license - see LICENSE

from astropy.io.fits import PrimaryHDU
from astropy.utils.decorators import format_doc
import numpy as np
import astropy.units as u
import re
import warnings

try:
    from spectral_cube import Projection, Slice
    SPECTRALCUBE_INSTALL = True
except ImportError:
    SPECTRALCUBE_INSTALL = False


allowed_types = ["numpy.ndarray", "astropy.io.fits.PrimaryHDU",
                 "spectral_cube.Projection", 'spectral_cube.Slice',
                 "astropy.units.Quantity"]

input_doc = \
    '''
    Accept a variety of input data forms and return those expected by the
    various statistics.

    Parameters
    ----------
    data : {}
        Data to be used with a given statistic or distance metric. no_header
        must be enabled when passing only an array in.
    header : `astropy.io.fits.Header`, optional
        Pass a header when data is a numpy array or `astropy.units.Quantity`.

    Returns
    -------
    ouput_data : tuple or np.ndarray
        A tuple containing the data and the header. Or an array when no_header
        is enabled.
    '''.format(" or ".join(allowed_types))


@format_doc(input_doc)
def input_data(data, header=None):
    '''
    '''

    output_data = None

    if SPECTRALCUBE_INSTALL:
        if isinstance(data, Projection) or isinstance(data, Slice):
            # spectral-cube has dimensionality checks
            output_data = {'data': data.quantity, 'header': data.header}

    if output_data is None:
        if isinstance(data, PrimaryHDU):
            squeeze_data = data.data.squeeze()
            if 'BUNIT' in data.header:
                unit = convert_bunit(data.header['BUNIT'])
            else:
                unit = u.dimensionless_unscaled
            output_data = {"data": squeeze_data * unit,
                           "header": data.header}

        elif isinstance(data, np.ndarray) or isinstance(data, u.Quantity):
            squeeze_data = data.squeeze()

            output_data = {}

            if header is not None:
                if not hasattr(data, 'unit'):
                    # Attach the BUNIT if defined
                    if 'BUNIT' in header:
                        bunit = convert_bunit(header['BUNIT'])
                        squeeze_data = squeeze_data * bunit

                else:
                    # Make sure the BUNIT in the header is the same as the data
                    header = header.copy()
                    header['BUNIT'] = data.unit.to_string()

                output_data['header'] = header

            if not hasattr(data, 'unit'):
                unit = u.dimensionless_unscaled
                squeeze_data = squeeze_data * unit

            output_data["data"] = squeeze_data
        else:
            raise TypeError("Input data is not of an accepted form:"
                            "{}".format(allowed_types))

    if not dim_check(output_data["data"]):
        raise TypeError("Data must be 2D. Please re-check the inputs.")

    return output_data


def dim_check(arr):
    return True if arr.ndim == 2 else False


def convert_bunit(bunit):
    '''
    Convert a BUNIT string to a quantity

    Stolen from spectral-cube: https://github.com/radio-astro-tools/spectral-cube/blob/c6ef0e6b1f8ba54116c8f558466aff4efb8caf60/spectral_cube/cube_utils.py#L423

    Parameters
    ----------
    bunit : str
        String to convert to an `~astropy.units.Unit`

    Returns
    -------
    unit : `~astropy.unit.Unit`
        Corresponding unit.
    '''

    # special case: CASA (sometimes) makes non-FITS-compliant jy/beam headers
    bunit_lower = re.sub("\s", "", bunit.lower())
    if bunit_lower == 'jy/beam':
        unit = u.Jy / u.beam
    else:
        try:
            unit = u.Unit(bunit)
        except ValueError:
            warnings.warn("Could not parse unit {0}".format(bunit))
            unit = None

    return unit
