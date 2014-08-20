from distutils.core import setup

setup(name='quantecon',
      packages=['quantecon', 'quantecon.models', "quantecon.tests"],
      version='0.1.5',
      description='Core package of the QuantEcon library',
      author='Thomas J. Sargent and John Stachurski (Project coordinators)',
      author_email='john.stachurski@gmail.com',
      url='https://github.com/jstac/quant-econ',  # URL to the github repo
      download_url='https://github.com/jstac/quant-econ/tarball/0.1.5',
      keywords=['quantitative', 'economics']
      )
