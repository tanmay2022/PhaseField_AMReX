#!/bin/sh

#SBATCH -N 2

#SBATCH --ntasks-per-node=48

#SBATCH --exclusive

#SBATCH --time=00:40:00 

#SBATCH --job-name=GP4ku

#SBATCH --error=job.%J.err_cpu 

#SBATCH --output=job.%J.out_cpu 

#SBATCH --partition=debug

###SBATCH --nodelist=cn321

#export I_MPI_PIN=on
#export I_MPI_PIN_RESPECT_CPUSET=on
#export I_MPI_PIN_RESPECT_HCA=on
#export I_MPI_PIN_CELL=core
#export I_MPI_PIN_DOMAIN=auto
#export I_MPI_PIN_ORDER=compact

cd /home/ext/extvshenoi/amrex-dev/sample_codes/Dendritic/GP_Intel/Exec
source /home/ext/extvshenoi/amrex-dev/sample_codes/Dendritic/GP_Intel/Exec/make-env.sh

ulimit -aH
ulimit -c unlimited
ulimit -s unlimited
#I_MPI_PIN=on I_MPI_PIN_RESPECT_CPUSET=on I_MPI_PIN_RESPECT_HCA=on I_MPI_PIN_CELL=core I_MPI_PIN_DOMAIN=auto I_MPI_PIN_ORDER=compact mpirun -n 96 ./main2d.intel.MPI.ex input2.in
mpirun -n  96 ./main2d.intel.MPI.ex input2.in
