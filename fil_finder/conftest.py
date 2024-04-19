# this contains imports plugins that configure py.test for astropy tests.
# by importing them here in conftest.py they are discoverable by py.test
# no matter how it is invoked within the source tree.
from __future__ import print_function, absolute_import, division

import os
from setuptools._distutils.version import LooseVersion

from astropy.version import version as astropy_version

if astropy_version < '3.0':
    from astropy.tests.pytest_plugins import *
    del pytest_report_header
else:
    from pytest_astropy_header.display import PYTEST_HEADER_MODULES, TESTED_VERSIONS


def pytest_configure(config):

    config.option.astropy_header = True

    PYTEST_HEADER_MODULES['Astropy'] = 'astropy'
