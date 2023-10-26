#!/bin/bash

echo $1
echo $2

# ignore deleted files, filter for .ipynb only, filter for last 10 commits
# files=$(git diff --name-only --diff-filter=d $1 $2 | grep .ipynb)
files=$(git diff --name-only origin/master~10..HEAD | grep .ipynb | grep -v 'unfinished')
for i in $files;
do
   echo $i
   jupyter nbconvert --to markdown $i
done
