#!/bin/bash
# TODO: Add error handling for upload command (for instance for the case that this is run more than once with same package number)
if [ $BRANCH == master ]
then
  pip install twine
  pip install --upgrade setuptools wheel
  python setup.py sdist bdist_wheel
  echo "On branch $BRANCH"
  python -m twine upload --verbose -u $PYPI_USERNAME -p $PYPI_PASSWORD --repository-url https://upload.pypi.org/legacy/ dist/*
else
  echo "Skipping python package upload on non-master branch $BRANCH"
fi