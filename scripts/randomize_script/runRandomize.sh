#!/bin/bash  
#SBATCH --job-name="runRandomize"  
#SBATCH --output="runRandomize.%j.%N.out"  
#SBATCH --partition=compute  
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --export=ALL
#SBATCH -t 01:30:00  

/usr/bin/time -v ./takeRandom.py LS002 5 40 5
