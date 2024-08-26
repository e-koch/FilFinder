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

import pytest
from fil_finder.tests.testing_utils import generate_filament_model


def pytest_configure(config):

    config.option.astropy_header = True

    PYTEST_HEADER_MODULES['Astropy'] = 'astropy'


@pytest.fixture
def simple_filament_model():

    mod = generate_filament_model(return_hdu=True, pad_size=31, shape=150,
                                  width=10., background=0.1)[0]

    yield mod