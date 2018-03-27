# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

import astropy.units as u
import numpy as np
from astropy.io import fits
from warnings import warn


try:
    from radio_beam import Beam, NoBeamException
    RADIO_BEAM_INSTALL = True
except ImportError:
    RADIO_BEAM_INSTALL = False


def find_beam_properties(hdr):
    '''
    Try to read beam properties from a header. Uses radio_beam when installed.

    Parameters
    ----------
    hdr : `~astropy.io.fits.Header`
        FITS header.

    Returns
    -------
    bmaj : `~astropy.units.Quantity`
        Major axis of the beam in degrees.
    bmin : `~astropy.units.Quantity`
        Minor axis of the beam in degrees. If this cannot be read from the
        header, assumes `bmaj=bmin`.
    bpa : `~astropy.units.Quantity`
        Position angle of the major axis. If this cannot read from the
        header, assumes an angle of 0 deg.
    '''

    if RADIO_BEAM_INSTALL:
        try:
            beam = Beam.from_fits_header(hdr)
            bmaj = beam.major.to(u.deg)
            bmin = beam.minor.to(u.deg)
            bpa = beam.pa.to(u.deg)
        except NoBeamException:
            bmaj = None
            bmin = None
            bpa = None
    else:
        if not isinstance(hdr, fits.Header):
            raise TypeError("Header is not a FITS header.")

        if "BMAJ" in hdr:
            bmaj = hdr["BMAJ"] * u.deg
        else:
            warn("Cannot find 'BMAJ' in the header. Try installing"
                 " the `radio_beam` package for loading header"
                 " information.")
            bmaj = None

        if "BMIN" in hdr:
            bmin = hdr["BMIN"] * u.deg
        else:
            warn("Cannot find 'BMIN' in the header. Assuming circular beam.")
            bmin = bmaj

        if "BPA" in hdr:
            bpa = hdr["BPA"] * u.deg
        else:
            warn("Cannot find 'BPA' in the header. Assuming PA of 0.")
            bpa = 0 * u.deg

    return bmaj, bmin, bpa


class BaseInfoMixin(object):
    """
    Common celestial information
    """

    @property
    def image(self):
        '''
        Image.
        '''
        return self._image

    @property
    def header(self):
        '''
        FITS Header.
        '''
        return self._header

    @property
    def wcs(self):
        '''
        WCS Object.
        '''
        return self._wcs

    @property
    def beamwidth(self):
        '''
        Beam major axis.
        '''
        return self._beamwidth

    @property
    def _has_beam(self):
        if hasattr(self, '_beamwidth'):
            return True
        return False


class UnitConverter(object):
    """
    Handle pixel, angular, and physical spatial unit conversions. Requires
    pixels to be square. Conversions are not aware of any axis misalignment.
    """
    def __init__(self, wcs=None, distance=None):

        if wcs is not None:
            if not wcs.is_celestial:
                self._wcs = wcs.celestial
            else:
                self._wcs = wcs

            self._ang_size = np.abs(self._wcs.wcs.cdelt[0]) * \
                u.Unit(self._wcs.wcs.cunit[0])
            self._ang_size = self._ang_size.to(u.deg)

            if distance is not None:
                self.distance = distance

    @property
    def ang_size(self):
        '''
        Angular size of one pixel.
        '''
        return self._ang_size

    @property
    def angular_equiv(self):
        return [(u.pix, u.deg, lambda x: x * float(self.ang_size.value),
                lambda x: x / float(self.ang_size.value))]

    @property
    def distance(self):
        if not hasattr(self, "_distance"):
            raise AttributeError("No distance has not been given.")

        return self._distance

    @distance.setter
    def distance(self, value):
        '''
        Value must be a quantity with a valid distance unit. Will keep the
        units given.
        '''

        if not isinstance(value, u.Quantity):
            raise TypeError("Value for distance must an astropy Quantity.")

        if not value.unit.is_equivalent(u.pc):
            raise u.UnitConversionError("Given unit ({}) is not a valid unit"
                                        " of distance.".format(value.unit))

        if not value.isscalar:
            raise TypeError("Distance must be a scalar quantity.")

        self._distance = value

    @property
    def physical_size(self):
        '''
        Physical size of one pixel.
        '''
        if not hasattr(self, "_distance"):
            raise AttributeError("No distance has not been given.")

        return (self.ang_size *
                self.distance).to(self.distance.unit,
                                  equivalencies=u.dimensionless_angles())

    @property
    def physical_equiv(self):
        if not hasattr(self, "_distance"):
            raise AttributeError("No distance has not been given.")

        return [(u.pix, self.distance.unit,
                lambda x: x * float(self.physical_size.value),
                lambda x: x / float(self.physical_size.value))]

    def to_pixel(self, value):
        '''
        Convert from angular or physical scales to pixels.
        '''

        if not isinstance(value, u.Quantity):
            raise TypeError("value must be an astropy Quantity object.")

        # Angular converions
        if value.unit.is_equivalent(u.pix):
            return value
        elif value.unit.is_equivalent(u.deg):
            return value.to(u.pix, equivalencies=self.angular_equiv)
        elif value.unit.is_equivalent(u.pc):
            return value.to(u.pix, equivalencies=self.physical_equiv)
        else:
            raise u.UnitConversionError("value has units of {}. It must have "
                                        "an angular or physical unit."
                                        .format(value.unit))

    def to_pixel_area(self, value):
        '''
        Should have an area-equivalent unit.
        '''
        return self.to_pixel(np.sqrt(value))**2

    def to_angular(self, value, unit=u.deg):
        return value.to(unit, equivalencies=self.angular_equiv)

    def to_physical(self, value, unit=u.pc):
        if not hasattr(self, "_distance"):
            raise AttributeError("No distance has not been given.")

        return value.to(unit, equivalencies=self.physical_equiv)

    def from_pixel(self, pixel_value, unit):
        '''
        Convert a value in pixel units to the given unit.
        '''

        if isinstance(unit, u.Quantity):
            unit = unit.unit

        if unit.is_equivalent(u.pix):
            return pixel_value
        elif unit.is_equivalent(u.deg):
            return self.to_angular(pixel_value, unit)
        elif unit.is_equivalent(u.pc):
            return self.to_physical(pixel_value, unit)
        else:
            raise u.UnitConversionError("unit must be an angular or physical"
                                        " unit.")


def data_unit_check(value, unit):
    '''
    Check that a value has a unit equivalent to the given unit. If no unit is
    attached, add the given unit to the value.
    '''

    if hasattr(value, 'unit'):
        if not value.unit.is_equivalent(unit):
            raise u.UnitConversionError("The given value does not have "
                                        "equivalent units.")
        return value.to(unit)

    else:
        return value * unit
