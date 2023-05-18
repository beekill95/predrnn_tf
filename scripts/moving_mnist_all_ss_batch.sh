#!/bin/bash

# Script parameters.
ss_options=("linear" "expo" "sigmoid")
runs_per_config=${1-5}

# Schedule job to run with sbatch.
schedule_slurm_job ()
{
    ss_type=$1

    sbatch <<EOT
#!/bin/bash

#SBATCH -J predrnn_moving_mnist_all_ss
#SBATCH -p gpu
#SBATCH -o slurm_logs/mnist_all_ss_%j.txt
#SBATCH -e slurm_logs/mnist_all_ss_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=qmnguyen@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 4
#SBATCH --gpus-per-node v100:1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH -A r00043

#Run your program
srun scripts/moving_mnist_all_ss.sh $ss_type
EOT
}

# Schedule multiple jobs at once.
for ss_option in "${ss_options[@]}";
do
    for i in $(seq 1 $runs_per_config);
    do
        schedule_slurm_job $ss_option
    done
done
