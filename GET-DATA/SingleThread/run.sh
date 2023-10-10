#!/usr/bin/bash 
#SBATCH --nodes=1               # node count
#SBATCH --ntasks=1              # total number of tasks across all nodes...   
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)  
#SBATCH --mem-per-cpu=10G         # memory per cpu-core (4G is default)
#SBATCH --time=0-03:00:00          # total run time limit (HH:MM:SS)

# One node has 128 cpus.... 
# so we need to be carefull to coretly adjust "ntasks" and "cpus-per-task", wrt. "nodes"


#SBATCH --job-name=Local-H     # create a short name for your job

module load Julia
julia GetChaosIndicatorsMB.jl 10 100
