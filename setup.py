#!/usr/bin/env python
# Licensed under an MIT open source license - see LICENSE

from ez_setup import use_setuptools
use_setuptools()


from setuptools import setup, find_packages
from pkg_resources import parse_version


def check_dependencies():

    try:
        import matplotlib
        mpl_version = matplotlib.__version__
        if parse_version(mpl_version) < parse_version('1.2'):
            print("***Before installing, upgrade matplotlib to 1.2***")
            raise ImportError
    except:
        raise ImportError(
            "Install or upgrade matplotlib before installing fil_finder.")

    try:
        from numpy.version import version as np_version
        if parse_version(np_version) < parse_version('1.7'):
            print("***Before installing, upgrade numpy to 1.7***")
            raise ImportError
    except:
        raise ImportError(
            "Install or upgrade numpy before installing fil_finder.")

    try:
        from scipy.version import version as sc_version
        if parse_version(sc_version) < parse_version('0.12'):
            print("***Before installing, upgrade scipy to 0.12***")
            raise ImportError
    except:
        raise ImportError(
            "Install or upgrade scipy before installing fil_finder.")

    try:
        from astropy.version import version as ast_version
        if parse_version(ast_version[:3]) < parse_version('0.2'):
            print(("""***Before installing, upgrade astropy to 0.2.
                    NOTE: This is the dev version as of 17/06/14.***"""))
            raise ImportError("")
    except:
        raise ImportError(
            "Install or upgrade astropy before installing fil_finder.")

    try:
        from networkx.version import version as nx_version
    except:
        raise ImportError(
            "Install networkx before installing fil_finder.")

    try:
        import skimage
    except:
        raise ImportError(
            "Install or upgrade skimage before installing fil_finder.")

if __name__ == "__main__":

    # check_dependencies()

    setup(name='fil_finder',
          version='1.0',
          description='Python package for analysis of filamentary structure in molecular clouds.',
          author='Eric Koch and Erik Rosolowsky',
          author_email='koch.eric.w@gmail.com',
          url='http://github.com/e-koch/fil_finder',
          packages=['fil_finder'],
          requires=['numpy','astropy','scipy','skimage','networkx', 'matplotlib'],
          # setup_requires=['numpy>=1.7'],
          # install_requires=['matplotlib>=1.2',
          #                   'astropy>=0.2',
          #                   'networkx',
          #                   'scikit-image',
          #                   'scipy>=0.13']
         )
