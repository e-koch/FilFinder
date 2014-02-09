#!/usr/bin/env python

from setuptools import setup

setup(name='fil_finder',
      version='1.0',
      description='Python package for analysis of filamentary structure in molecular clouds.',
      author='Eric Koch and Erik Rosolowsky',
      author_email='koch.eric.w@gmail.com',
      url='http://github.com/e-koch/fil_finder',
      packages=['fil_finder'],
      requires=['numpy','astropy','scipy','skimage','networkx','pandas']
     )
