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

python -m catserl.orchestrator.orchestrator --save-data-dir=/nfs/stak/users/thakarr/hpc-share/mopderl-env/weightconditioned_log_data_stage1_only/MO-HalfCheetah-v2/run_1 --config=/nfs/stak/users/thakarr/hpc-share/multi-policy-morl-env/Multi-Policy-MORL/catserl/shared/config/halfcheetah.yaml --resume-stage2