#!/bin/bash
set -e

host=greatlakes.arc-ts.umich.edu
user=daiweiz
dir=/home/daiweiz/dalea_project/examples/results
src=$user@$host:$dir
dst=.
rsync -am -e ssh --progress -r $src $dst
