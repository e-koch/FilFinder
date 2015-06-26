#!/usr/bin/env python
# Licensed under an MIT open source license - see LICENSE

from ez_setup import use_setuptools
use_setuptools()


from setuptools import setup, find_packages
from pkg_resources import parse_version
from setuptools.command.build_ext import build_ext as _build_ext

class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

from setuptools.command.install import install as SetuptoolsInstall


class install(SetuptoolsInstall):
    user_options = SetuptoolsInstall.user_options[:]
    boolean_options = SetuptoolsInstall.boolean_options[:]

    def finalize_options(self):
        check_dependencies()
        SetuptoolsInstall.finalize_options(self)

class check_deps(_build_ext):
    """docstring for check_deps"""
    def finalize_options(self):
        _build_ext.finalize_options(self)
        check_dependencies()

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

    # check_dependencies()

    setup(name='FilFinder',
          version='1.0',
          description='Python package for analysis of filamentary structure in molecular clouds.',
          author='Eric Koch and Erik Rosolowsky',
          author_email='koch.eric.w@gmail.com',
          url='http://github.com/e-koch/fil_finder',
          packages=['fil_finder'],
          # requires=['numpy','astropy','scipy','skimage','networkx', 'matplotlib'],
          cmdclass={'install': install, 'check_deps': check_deps},
          setup_requires=[],
          install_requires=[]
         )
