#!/usr/bin/env python
# Licensed under an MIT open source license - see LICENSE

import glob
import os
import sys

from pkg_resources import parse_version


def check_dependencies():

    def print_fail():
        return "fil_finder will not install dependencies. Install checks" \
               " if they are available, and proceeds if they are."

    try:
        from numpy.version import version as np_version
        if parse_version(np_version) < parse_version('1.7'):
            print("***Before installing, upgrade numpy to 1.7***")
            print_fail()
            raise ImportError
    except:
        raise ImportError(
            "Install or upgrade numpy before installing fil_finder." +
            print_fail())

    try:
        from scipy.version import version as sc_version
        if parse_version(sc_version) < parse_version('0.12'):
            print("***Before installing, upgrade scipy to 0.12***")
            print_fail()
            raise ImportError
    except:
        raise ImportError(
            "Install or upgrade scipy before installing fil_finder." +
            print_fail())

    try:
        import matplotlib
        mpl_version = matplotlib.__version__
        if parse_version(mpl_version) < parse_version('1.2'):
            print("***Before installing, upgrade matplotlib to 1.2***")
            print_fail()
            raise ImportError
    except:
        raise ImportError(
            "Install or upgrade matplotlib before installing fil_finder." +
            print_fail())

    try:
        from astropy.version import version as ast_version
        if parse_version(ast_version[:3]) < parse_version('0.2'):
            print(("""***Before installing, upgrade astropy to 0.2.
                    NOTE: This is the dev version as of 17/06/14.***"""))
            print_fail()
            raise ImportError("")
    except:
        raise ImportError(
            "Install or upgrade astropy before installing fil_finder." +
            print_fail())

    try:
        from networkx.version import version as nx_version
    except:
        raise ImportError(
            "Install networkx before installing fil_finder." +
            print_fail())

    try:
        import skimage
    except:
        raise ImportError(
            "Install or upgrade skimage before installing fil_finder." +
            print_fail())

if __name__ == "__main__":

    import ah_bootstrap
    from setuptools import setup

    from setuptools.command.build_ext import build_ext as _build_ext

    class check_deps(_build_ext):
        """Check if package dependencies are installed."""
        def finalize_options(self):
            _build_ext.finalize_options(self)
            check_dependencies()

    # A dirty hack to get around some early import/configurations ambiguities
    if sys.version_info[0] >= 3:
        import builtins
    else:
        import __builtin__ as builtins
    builtins._ASTROPY_SETUP_ = True

    from astropy_helpers.setup_helpers import (register_commands,
                                               get_debug_option,
                                               get_package_info)
    from astropy_helpers.git_helpers import get_git_devstr
    from astropy_helpers.version_helpers import generate_version_py

    # Get some values from the setup.cfg
    from distutils import config
    conf = config.ConfigParser()
    conf.read(['setup.cfg'])
    metadata = dict(conf.items('metadata'))

    PACKAGENAME = metadata.get('package_name', 'packagename')
    DESCRIPTION = metadata.get('description', 'Astropy affiliated package')
    AUTHOR = metadata.get('author', '')
    AUTHOR_EMAIL = metadata.get('author_email', '')
    LICENSE = metadata.get('license', 'unknown')
    URL = metadata.get('url', 'http://astropy.org')

    # Get the long description from the package's docstring
    __import__(PACKAGENAME)
    package = sys.modules[PACKAGENAME]
    LONG_DESCRIPTION = package.__doc__

    # Store the package name in a built-in variable so it's easy
    # to get from other parts of the setup infrastructure
    builtins._ASTROPY_PACKAGE_NAME_ = PACKAGENAME

    # VERSION should be PEP386 compatible (http://www.python.org/dev/peps/pep-0386)
    VERSION = '1.3'

    # Indicates if this version is a release version
    RELEASE = 'dev' not in VERSION

    if not RELEASE:
        VERSION += get_git_devstr(False)

    # Populate the dict of setup command overrides; this should be done before
    # invoking any other functionality from distutils since it can potentially
    # modify distutils' behavior.
    cmdclassd = register_commands(PACKAGENAME, VERSION, RELEASE)
    cmdclassd['check_deps'] = check_deps

    # Freeze build information in version.py
    generate_version_py(PACKAGENAME, VERSION, RELEASE,
                        get_debug_option(PACKAGENAME))

    # Treat everything in scripts except README.rst as a script to be installed
    scripts = [fname for fname in glob.glob(os.path.join('scripts', '*'))
               if os.path.basename(fname) != 'README.rst']


    # Get configuration information from all of the various subpackages.
    # See the docstring for setup_helpers.update_package_files for more
    # details.
    package_info = get_package_info()

    # Add the project-global data
    package_info['package_data'].setdefault(PACKAGENAME, [])
    package_info['package_data'][PACKAGENAME].append('data/*')

    # Define entry points for command-line scripts
    entry_points = {'console_scripts': []}

    entry_point_list = conf.items('entry_points')
    for entry_point in entry_point_list:
        entry_points['console_scripts'].append('{0} = {1}'.format(entry_point[0],
                                                                  entry_point[1]))

    # Include all .c files, recursively, including those generated by
    # Cython, since we can not do this in MANIFEST.in with a "dynamic"
    # directory name.
    c_files = []
    for root, dirs, files in os.walk(PACKAGENAME):
        for filename in files:
            if filename.endswith('.c'):
                c_files.append(
                    os.path.join(
                        os.path.relpath(root, PACKAGENAME), filename))
    package_info['package_data'][PACKAGENAME].extend(c_files)

    # Note that requires and provides should not be included in the call to
    # ``setup``, since these are now deprecated. See this link for more details:
    # https://groups.google.com/forum/#!topic/astropy-dev/urYO8ckB2uM

    setup(name=PACKAGENAME,
          version=VERSION,
          description=DESCRIPTION,
          scripts=scripts,
          install_requires=['astropy'],
          author=AUTHOR,
          author_email=AUTHOR_EMAIL,
          license=LICENSE,
          url=URL,
          long_description=LONG_DESCRIPTION,
          cmdclass=cmdclassd,
          zip_safe=False,
          use_2to3=True,
          entry_points=entry_points,
          **package_info)
