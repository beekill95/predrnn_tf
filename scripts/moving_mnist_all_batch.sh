#!/bin/bash

# Script parameters.
model_options=("predrnn" "convlstm")
use_ss_options=("True" "False")
runs_per_config=${1-5}

# Schedule job to run with sbatch.
schedule_slurm_job ()
{
    model=$1
    use_ss=$2

    sbatch <<EOT
#!/bin/bash

#SBATCH -J predrnn_moving_mnist_all
#SBATCH -p gpu
#SBATCH -o slurm_logs/mnist_all_%j.txt
#SBATCH -e slurm_logs/mnist_all_%j.err
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
srun scripts/moving_mnist_all.sh $model $use_ss
EOT
}

# Schedule multiple jobs at once.
for model_option in "${model_options[@]}";
do
    for ss_option in "${use_ss_options[@]}";
    do
        # Check invalid configuration.
        if [ "$model_option" = "convlstm" ] && [ "$ss_option" = "True" ]
        then
            continue
        fi

        for i in $(seq 1 $runs_per_config);
        do
            schedule_slurm_job $model_option $ss_option
        done
    done
done
