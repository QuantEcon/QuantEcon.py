"""
Origin: QE by John Stachurski and Thomas J. Sargent
Filename: yahoo_fin.py
Authors: John Stachurski, Thomas J. Sargent
LastModified: 11/08/2013

Compute returns on a certain portfolio since the start of the year, and plot
them from best to worst.
"""

from urllib import urlopen, urlencode
from datetime import date, timedelta
from operator import itemgetter

today = date.today()
base_url = 'http://ichart.finance.yahoo.com/table.csv'

def get_stock_price(request_date, ticker):
    """
    Get stock price corresponding to the date and ticker.  

    Parameters
    ===========

    request_date : datetime.date instance
        The desired day

    ticker : string
        The stock, such as 'AAPL'

    Returns
    ========

    The opening price as a float
    """
    
    previous_date = request_date - timedelta(days=1)

    dd1 = str(previous_date.day)
    mm1 = str(previous_date.month - 1)  
    yr1 = str(previous_date.year)  

    dd2 = str(request_date.day)
    mm2 = str(request_date.month - 1)  
    yr2 = str(request_date.year)  

    request_data = {'a': mm1,            # Start month, base zero
                    'b': dd1,            # Start day
                    'c': yr1,            # Start year
                    'd': mm2,            # End month, base zero
                    'e': dd2,            # End day
                    'f': yr2,            # End year
                    'g': 'd',            # Daily data
                    's': ticker,         # Ticker name
                    'ignore': '.csv'}    # Data type

    url = base_url + '?' + urlencode(request_data)
    response = urlopen(url)
    response.next()                 # Skip the first line
    prices = response.next()        
    price = prices.split(',')[1]    # Opening price
    return float(price)

# Find the first Monday of the current year
first_weekday = date(today.year, 1, 1)  # Start at 1st of Jan
while first_weekday.weekday() > 4:      # 5 and 6 correspond to the weekend
    first_weekday += timedelta(days=1)  # Increment date by one day

# Find the most recent weekday, starting yesterday
most_recent_weekday = today - timedelta(days=1)
while most_recent_weekday.weekday() > 4:       # If it's the weekend
    most_recent_weekday -= timedelta(days=1)   # Go back one day

portfolio = open('portfolio.txt')  
percent_change = {}
for line in portfolio:
    ticker, company_name = [item.strip() for item in line.split(',')]
    old_price = get_stock_price(first_weekday, ticker)
    new_price = get_stock_price(most_recent_weekday, ticker)
    percent_change[company_name] = 100 * (new_price - old_price) / old_price
portfolio.close()

items = percent_change.items()

for name, change in sorted(items, key=itemgetter(1), reverse=True):
    print '%-12s %10.2f' % (name, change)
