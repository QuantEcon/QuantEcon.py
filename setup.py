from distutils.core import setup
import os

#-Write Versions File-#
#~~~~~~~~~~~~~~~~~~~~~#

VERSION = '0.1.7'


def write_version_py(filename=None):
    doc = '''
    """
    This is a VERSION file and should NOT be manually altered

    """

    version = "%s"
    ''' % VERSION

    return doc

    if not filename:
        filename = os.path.join(
            os.path.dirname(__file__), 'quantecon', 'version.py')

    fl = open(filename, 'w')
    try:
        fl.write(doc)
    finally:
        fl.close()

write_version_py()  # This is a file used to control the qe.__version__ attribute

#-Meta Information-#
#~~~~~~~~~~~~~~~~~~#

DESCRIPTION = "QuantEcon is a package to support all forms of quantitative economic modelling."       #'Core package of the QuantEcon library'

LONG_DESCRIPTION = """
**QuantEcon** is an organization run by economists for economists with the aim of coordinating 
distributed development of high quality open source code for all forms of quantitative economic modelling. 

The project website is located at [http://quantecon.org/](http://quantecon.org/). This website provides
more information with regards to the **quantecon** library, documentation, in addition to some resources 
in regards to how you can use and/or contribute to the package. 

## The **quantecon** Package

The [repository](https://github.com/QuantEcon/QuantEcon.py) includes the Python package `quantecon`

Assuming you have [pip](https://pypi.python.org/pypi/pip) on your computer --- as will be the case if you've [installed Anaconda](http://quant-econ.net/getting_started.html#installing-anaconda) --- you can install the latest stable release of `quantecon` by typing

    pip install quantecon

at a terminal prompt

## Repository

The main repository is hosted on Github [QuantEcon.py](https://github.com/QuantEcon/QuantEcon.py)

Note
----
There is also a Julia version available for Julia users [QuantEcon.jl](https://github.com/QuantEcon/QuantEcon.jl)

#### Current Build and Coverage Status:

[![Build Status](https://travis-ci.org/QuantEcon/QuantEcon.py.svg?branch=master)](https://travis-ci.org/QuantEcon/QuantEcon.py)
[![Coverage Status](https://coveralls.io/repos/QuantEcon/QuantEcon.py/badge.png)](https://coveralls.io/r/QuantEcon/QuantEcon.py)

## Additional Links

1. [QuantEcon Course Website](http://quant-econ.net) 

"""

LICENSE = "BSD"

#-Classifier Strings-#
#-https://pypi.python.org/pypi?%3Aaction=list_classifiers-#
CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3.3',
    'Programming Language :: Python :: 3.4',
    'Topic :: Scientific/Engineering',
]

#-Setup-#
#~~~~~~~#

setup(name='quantecon',
      packages=['quantecon', 'quantecon.models', "quantecon.tests"],
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      license=LICENSE,
      classifiers=CLASSIFIERS,
      author='Thomas J. Sargent and John Stachurski (Project coordinators)',
      author_email='john.stachurski@gmail.com',
      url='https://github.com/QuantEcon/QuantEcon.py',  # URL to the repo
      keywords=['quantitative', 'economics']
      )
