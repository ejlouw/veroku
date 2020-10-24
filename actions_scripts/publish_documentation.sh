#!/bin/bash
if [ $BRANCH == master ]
then
  rm -rf docs || :
  mv ../docs_copy/build/html docs
  git config user.name github-actions
  git config user.email github-actions@github.com
  git add docs/ || :
  git commit -m 'Update documentation.' || :
  git push || :
else
  echo "Skipping package documentation upload on non-master branch $BRANCH"
fi