# QuantEcon documentation

This is the main directory for the documentation for the `quantecon` python library.

## Dependencies

The documentation requires a few dependencies beyond those necessary for the quantecon library. These dependencies are (warning, this may be an incomplete list):

* sphinx
* numpydoc
* sphinx_rtd_theme
* mock

You can install these by executing

```
conda install sphinx numpydoc sphinx_rtd_theme mock
```

## Building the docs

In order to generate the documentation, follow these steps:

1. Install the `quantecon` python library locally. Do to this enter the commands below:
```
cd ..
python setup.py install
cd docs
```
2. From this directory, execute the local file `qe_apidoc.py` (for an explanation of what the file does, see the module level docstring in the file)
```
python qe_apidoc.py
```
3. Run sphinx using the Makefile (this is the command for unix based system -- sorry windows users, you will have to google how to do this)
```
make html
```
4. Open the file `build/html/index.html`.

I have added a couple utility commands to the make file:

```
srcclean:
    rm -rf source/modules*
    rm -rf source/models*
    rm -rf source/tools*
    rm -f source/index.rst
    rm -f source/models.rst
    rm -f source/tools.rst

myhtml:
    make srcclean
    cd .. && python setup.py install && cd docs
    python qe_apidoc.py
    make html
```

Notice that we can automate steps 1-3 (and make sure we get a clean build) above by simply running `make myhtml`
