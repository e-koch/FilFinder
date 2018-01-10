# Licensed under an MIT open source license - see LICENSE

import astropy.units as u
import numpy as np


class BaseInfoMixin(object):
    """
    Common celestial information
    """

    @property
    def header(self):
        return self._header

    @property
    def wcs(self):
        return self._wcs


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
