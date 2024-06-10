#!/bin/bash

#SBATCH --job-name=softclt_test
#SBATCH --account=ec35
#SBATCH -e "%u_uh-oh.txt"
#SBATCH --output="%train_test.txt"
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --qos=devel
#SBATCH --mem=12GB
#SBATCH --partition=accel
#SBATCH --gpus=1

module purge  # Clear all loaded modules
module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0

echo "Running GPU check script"

# Run the GPU check script
python check_gpu.py
if [ $? -ne 0 ]; then
    echo "GPU not available. Exiting."
    exit 1
fi

echo "GPU is available. Running test job"

bs=8
data='Cricket'
tau_inst=10
tau_temp=2.0

# Ensure CUDA_VISIBLE_DEVICES is set
export CUDA_VISIBLE_DEVICES=0

python softclt_ts2vec/train.py $data --loader='UEA' --batch-size $bs --eval \
    --tau_inst $tau_inst --tau_temp $tau_temp

echo "End of job"
