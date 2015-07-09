Installation
------------

FilFinder can be installed via pip:

>>> pip install FilFinder

To install from the repository, run:

>>> python setup.py install


**NOTE:** Due to install conflicts amongst FilFinder's dependencies, installing the package will **NOT** install the dependencies. To check if you have the necessary packages installed, run:

>>> python setup.py check_deps

Unfortunately, this is only available when installing from the repository.

Package Dependencies
--------------------

Requires:

 *   numpy 1.7.1
 *   matplotlib
 *   astropy
 *   scipy
 *   scikits-image 0.8.0
 *   networkx

Optional:

 *  `prettyplotlib <https://github.com/olgabot/prettyplotlib>`_ - *Will eventually be removed in a future release*
