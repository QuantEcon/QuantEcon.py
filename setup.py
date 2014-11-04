from distutils.core import setup
import os

#-Write a versions.py file for class attribute-#

VERSION = '0.1.6'

def write_version_py(filename=None):
    doc = "\"\"\"\nThis is a VERSION file and should NOT be manually altered\n\"\"\""
    doc += "\nversion = '%s'" % VERSION
    
    if not filename:
        filename = os.path.join(
            os.path.dirname(__file__), 'quantecon', 'version.py')

    fl = open(filename, 'w')
    try:
        fl.write(doc)
    finally:
        fl.close()

write_version_py()

#-Setup-#

setup(name='quantecon',
      packages=['quantecon', 'quantecon.models', "quantecon.tests"],
      version=VERSION,
      description='Core package of the QuantEcon library',
      author='Thomas J. Sargent and John Stachurski (Project coordinators)',
      author_email='john.stachurski@gmail.com',
      url='https://github.com/QuantEcon/QuantEcon.py',  # URL to the github repo
      keywords=['quantitative', 'economics']
      )
