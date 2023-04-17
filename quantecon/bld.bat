"%PYTHON%" setup.py install
if errorlevel 1 exit 1

:: Add more build steps here, if they are necessary.

:: See 
:: https://docs.conda.io/projects/conda-build/en/latest/resources/define-metadata.html
:: for a list of environment variables that are set during the build process. 