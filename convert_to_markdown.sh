#!/bin/bash

echo $1
echo $2

# only convert newly changed files?
# ignore deleted files, filter for .ipynb only, filter for last 10 commits
if [ "$1" = "diff" ]; then
   files=$(git diff --name-only origin/master~10..HEAD | grep .ipynb | grep -v 'unfinished')
else
   files=$(find . -type f -name "*.ipynb" | grep .ipynb | grep -v 'unfinished')
fi

for i in $files;
do
   echo $i
   jupyter nbconvert --to markdown $i
done
