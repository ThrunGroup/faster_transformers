#!/bin/bash
#SBATCH --account=matx
#SBATCH --partition=matx-interactive
#SBATCH --gres=gpu:6
#SBATCH --mem=400gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=100
#SBATCH --nodelist=matx1
#SBATCH --time=24:00:00  # adjust the time limit as necessary
#SBATCH --job-name="mistral_sparse_exp"
#SBATCH --output=batch_logs/batch_job_output_jan_25_%j.txt  # %j will be replaced with the job ID

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

# Turn of Token Parallelism (throws a warning when using multi-gpus and fast tokenizer at the same time)
export TOKENIZERS_PARALLELISM=false

# Add the commands you want to execute here
source /sailhome/vxbrando/.bashrc
conda activate mistral

cd /sailhome/vxbrando/faster_transformers
sh agg_sparsity.sh
#deepspeed experiments/sparse_finetune.py


# done
echo "Done"