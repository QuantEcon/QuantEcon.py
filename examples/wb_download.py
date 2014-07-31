"""
Origin: QE by John Stachurski and Thomas J. Sargent
Filename: wb_download.py
Authors: John Stachurski, Tomohito Okabe
LastModified: 29/08/2013

Dowloads data from the World Bank site on GDP per capita and plots result for
a subset of countries.

NOTE: This is not dually compatible with Python 3.  Python 2 and Python
3 call the urllib package differently.
"""
import sys
import matplotlib.pyplot as plt
from pandas.io.excel import ExcelFile

if sys.version_info[0] == 2:
    from urllib import urlretrieve
elif sys.version_info[0] == 3:
    from urllib.request import urlretrieve

# == Get data and read into file gd.xls == #
wb_data_file_dir = "http://api.worldbank.org/datafiles/"
file_name = "GC.DOD.TOTL.GD.ZS_Indicator_MetaData_en_EXCEL.xls"
url = wb_data_file_dir + file_name
urlretrieve(url, "gd.xls")

# == Parse data into a DataFrame == #
gov_debt_xls = ExcelFile('gd.xls')
govt_debt = gov_debt_xls.parse('Sheet1', index_col=1, na_values=['NA'])

# == Take desired values and plot == #
govt_debt = govt_debt.transpose()
govt_debt = govt_debt[['AUS', 'DEU', 'FRA', 'USA']]
govt_debt = govt_debt[36:]
govt_debt.plot(lw=2)
plt.show()
