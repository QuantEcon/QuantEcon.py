from urllib import urlopen, urlencode
from datetime import date
from operator import itemgetter

# Record current day and month as strings, month is base zero
today = date.today()
mm = str(today.month - 1)  
dd = str(today.day)

base_url = 'http://ichart.finance.yahoo.com/table.csv'

request_data = {'a': '00',            # Start month, base zero
                'b': '01',            # Start day
                'c': '2008',          # Start year
                'd': mm,              # End month, base zero
                'e': dd,              # End day
                'f': '2008',          # End year
                'g': 'd',             # Daily
                'ignore': '.csv'}     # Data type

# Main loop

portfolio = open('portfolio.txt')  
percent_change = {}
for line in portfolio:
    ticker, company_name = [item.strip() for item in line.split(',')]
    request_data['s'] = ticker
    response = urlopen(base_url + '?' + urlencode(request_data))
    response.next()  # Skip the first line
    prices = [line.split(',')[-1] for line in response]
    old_price, new_price = float(prices[-1]), float(prices[0])    
    percent_change[company_name] = 100 * (new_price - old_price) / old_price
portfolio.close()

items = percent_change.items()

for name, change in sorted(items, key=itemgetter(1), reverse=True):
    print '%-12s %10.2f' % (name, change)



    



