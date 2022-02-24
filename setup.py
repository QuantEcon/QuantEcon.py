# Use setuptools in preference to distutils
from setuptools import setup, find_packages
import os


# To find version from quantecon/version.py
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
version_path = os.path.join(ROOT_DIR, 'quantecon', 'version.py')
version_dict = {}
with open(version_path) as version_file:
    exec(version_file.read(), version_dict)


#-Meta Information-#
#~~~~~~~~~~~~~~~~~~#

DESCRIPTION = "QuantEcon is a package to support all forms of quantitative economic modelling."       #'Core package of the QuantEcon library'

LONG_DESCRIPTION = """
**QuantEcon** is an organization run by economists for economists with the aim of coordinating
distributed development of high quality open source code for all forms of quantitative economic modelling.

The project website is located at `https://quantecon.org/ <https://quantecon.org/>`_. This website provides
more information with regards to the **quantecon** library, documentation, in addition to some resources
in regards to how you can use and/or contribute to the package.

The **quantecon** Package
-------------------------

The `repository <https://github.com/QuantEcon/QuantEcon.py>`_ includes the Python package ``quantecon``

Assuming you have `pip <https://pypi.python.org/pypi/pip>`_ on your computer --- as will be the case if you've `installed Anaconda <https://python-programming.quantecon.org/getting_started.html#anaconda>`_ --- you can install the latest stable release of ``quantecon`` by typing

    pip install quantecon

at a terminal prompt

Repository
----------

The main repository is hosted on Github `QuantEcon.py <https://github.com/QuantEcon/QuantEcon.py>`_

**Note:** There is also a Julia version available for Julia users `QuantEcon.jl <https://github.com/QuantEcon/QuantEcon.jl>`_

Current Build and Coverage Status
---------------------------------

|Build Status| |Coverage Status|

.. |Build Status| image:: https://github.com/QuantEcon/QuantEcon.py/workflows/build/badge.svg
   :target: https://github.com/QuantEcon/QuantEcon.py/actions?query=workflow%3Abuild
.. |Coverage Status| image:: https://coveralls.io/repos/QuantEcon/QuantEcon.py/badge.png
   :target: https://coveralls.io/r/QuantEcon/QuantEcon.py

Additional Links
----------------

1. `Python Programming for Finance and Economics <https://python-programming.quantecon.org/intro.html>`__
2. `Quantitative Economics with python <https://python.quantecon.org/intro.html>`__
3. `Advanced Quantitative Economics with Python <https://python-advanced.quantecon.org/intro.html>`__

"""

LICENSE = "BSD"

#-Classifier Strings-#
#-https://pypi.python.org/pypi?%3Aaction=list_classifiers-#
CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Topic :: Scientific/Engineering',
]

#-Setup-#
#~~~~~~~#

VERSION = version_dict['version']

setup(name='quantecon',
      packages=find_packages(),
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      license=LICENSE,
      classifiers=CLASSIFIERS,
      author='Thomas J. Sargent and John Stachurski (Project coordinators)',
      author_email='john.stachurski@gmail.com',
      url='https://github.com/QuantEcon/QuantEcon.py',  # URL to the repo
      download_url='https://github.com/QuantEcon/QuantEcon.py/tarball/' + VERSION,
      keywords=['quantitative', 'economics'],
      install_requires=[
          'numba>=0.38',
          'numpy',
          'requests',
          'scipy>=1.0.0',
          'sympy',
          ],
      include_package_data=True
      )
