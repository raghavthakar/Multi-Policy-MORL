#!/bin/bash
#SBATCH --time=0-36:00:00
#SBATCH --partition=dgx2,dgxh,share,ampere
#SBATCH --constraint=skylake
#SBATCH --mem=46G
#SBATCH -c 12

module load conda

source activate base

conda activate /nfs/stak/users/thakarr/hpc-share/multi-policy-morl-env

cd /nfs/stak/users/thakarr/hpc-share/multi-policy-morl-env/Multi-Policy-MORL

python -m catserl.orchestrator.orchestrator --stage1-alg=td3  --save-data-dir=/nfs/stak/users/thakarr/hpc-share/multi-policy-morl-env/Multi-Policy-MORL/sampledata/ --config=/nfs/stak/users/thakarr/hpc-share/multi-policy-morl-env/Multi-Policy-MORL/catserl/shared/config/halfcheetah.yaml