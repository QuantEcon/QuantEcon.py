# QuantEcon.py Makefile

install:
	python setup.py install

test-all: test test-examples test-solutions

test:
	@echo "Running nosetests on test suite ..."
	nosetests -v

test-examples:
	@echo "Testing examples ..."
	python scripts/test-examples.py

test-solutions:
	@echo "testing solutions ..."
	python scripts/test-solutions.py
