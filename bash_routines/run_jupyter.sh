#/bin/bash

cd $2
jupyter notebook --ip=0.0.0.0 --port=$1 --no-browser --allow-root --notebook-dir=$2