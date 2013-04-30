f = open('us_cities.txt', 'r')
for line in f:
    city, population = line.split(':')            # Tuple unpacking
    city = city.title()                           # Capitalize city names
    population = '{0:,}'.format(int(population))  # Add commas to numbers
    print(city.ljust(15) + population)
f.close()
