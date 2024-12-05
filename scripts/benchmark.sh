#!/bin/bash
## A simple benchmarking routine

# Argument validation check
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <model> <scene>"
    exit 1
fi

cd models/$1

# Logging
mkdir -p ./output/log
logfile="${2}_$(date -d "today" +"%Y%m%d%H%M").log"
echo $logfile
touch ./output/log/$logfile
nvidia-smi >> ./output/log/$logfile

# Train
/h/285/kappa/.local/bin/micromamba run -n 3Dmenu python3 train.py -s ../../data/$2 --eval -m ./output/$2 >> output/log/$logfile
echo "Benchmark complete"

exit
