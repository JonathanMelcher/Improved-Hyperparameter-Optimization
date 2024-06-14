#!/bin/sh
### Note: No commands may be executed until after the #PBS lines
### Output files (comment out the next 2 lines to get the job name used instead)
# PBS -e ../Out/lgbm.err
# PBS -o ../Out/lgbm.log
### Only send mail when job is aborted or terminates abnormally

### Number of nodes
#PBS -l nodes=1:ppn=9
### Memory
#PBS -l mem=40gb
### Requesting time - format is <days>:<hours>:<minutes>:<seconds>
#PBS -l walltime=15:00:00:00

# Go to the directory from where the job was submitted (initial directory is $HOME)
echo Working directory is $PBS_O_WORKDIR
cd $PBS_O_WORKDIR

### Here follows the user commands:
# Define number of processors
NPROCS=`wc -l < $PBS_NODEFILE`
echo This job has allocated $NPROCS nodes

module load tools computerome_utils/2.0
module load anaconda3/2022.10
source /services/tools/anaconda3/2021.05/etc/profile.d/conda.sh

conda activate /home/projects/ku_00231/people/jonmel/conda/jonathan

module purge
module load tools computerome_utils/2.0

module load cuda/toolkit/11.2.0 cudnn/11.2-8.1.0.77
module load openmpi/gcc/64/4.0.2 gromacs/5.1.2-plumed


module list
conda info -e
conda list

dimensions=("2" "3" "4")
resolutions=("2" "3" "4" "5")


for r in "${resolutions[@]}"; do
# Loop through resolutions
    for d in "${dimensions[@]}"; do
    # Loop through dimensions
        echo "Processing dimension $d at resolution $r"
        mpiexec -n 9 python3 -m mpi4py Lightgbm_analysis_mpi.py --res "$r" --dim "$d" --start_batch "$START_BATCH"
    done
done
