# Licensed under an MIT open source license - see LICENSE

from astropy.io.fits import PrimaryHDU
# from spectral_cube.lower_dimensional_structures import LowerDimensionalObject
import numpy as np


allowed_types = ["numpy.ndarray", "astropy.io.fits.PrimaryHDU"]  # ,
                 # "spectral_cube.LowerDimensionalObject"]

def input_data(data):
    '''
    Accept a variety of input data forms and return those expected by the
    various statistics.

    Parameters
    ----------
    data : astropy.io.fits.PrimaryHDU, SpectralCube,
           spectral_cube.LowerDimensionalObject, np.ndarray or a tuple/list
           with the data and the header
        Data to be used with a given statistic or distance metric. no_header
        must be enabled when passing only an array in.

    Returns
    -------
    ouput_data : tuple or np.ndarray
        A tuple containing the data and the header. Or an array when no_header
        is enabled.
    '''

    if isinstance(data, PrimaryHDU):
        output_data = {"data": data.data.squeeze(), "header": data.header}

    # elif isinstance(data, LowerDimensionalObject):
    #     output_data = (data.value, data.header)
    elif isinstance(data, np.ndarray):
        output_data = {"data": data.squeeze()}
    else:
        raise TypeError("Input data is not of an accepted form:"
                        " astropy.io.fits.PrimaryHDU, np.ndarray.")

    if len(output_data["data"].shape) != 2:
        raise TypeError("Data must be 2D. Please re-check the inputs.")

    return output_data
