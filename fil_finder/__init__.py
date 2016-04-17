# Licensed under an MIT open source license - see LICENSE

from ._astropy_init import *

if not _ASTROPY_SETUP_:
    from .analysis import Analysis
    from .filfind_class import fil_finder_2D
