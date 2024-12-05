#!/bin/bash
## A simple metrics routine

# Argument validation check
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <model> <scene_path>"
    exit 1
fi

# Logging
logfile="$(date -d "today" +"%Y%m%d%H%M").log"
echo $2/$logfile
touch $2/$logfile
nvidia-smi >> $2/$logfile

# Render
/h/285/kappa/.local/bin/micromamba run -n 3Dmenu python3 ./models/$1/render.py -m $2 >> $2/$logfile

# Metrics
/h/285/kappa/.local/bin/micromamba run -n 3Dmenu python3 ./models/$1/metrics.py -m $2 >> $2/$logfile

echo "Benchmark complete"

exit
