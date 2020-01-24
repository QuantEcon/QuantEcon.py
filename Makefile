# QuantEcon.py Makefile

install:
	python setup.py install

test:
	@echo "Running nosetests on test suite ..."
	nosetests -v
