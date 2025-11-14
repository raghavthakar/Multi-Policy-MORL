import os

def generate_bash_scripts(env, data_dir, config):
    script_path = f"/nfs/stak/users/thakarr/hpc-share/multi-policy-morl-env/Multi-Policy-MORL/experiments/scripts/job_scripts/{env}_{config}.sh"
    
    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(script_path), exist_ok=True)
    
    with open(script_path, 'w') as file:
        file.write("#!/bin/bash\n")
        file.write(f"#SBATCH --time={time}\n")
        file.write("#SBATCH --partition=dgx2,dgxh,share,ampere\n")
        file.write("#SBATCH --constraint=skylake\n")
        file.write("#SBATCH --mem=46G\n")
        file.write("#SBATCH -c 12\n\n")
        # file.write("#SBATCH -G 1\n\n")
        file.write("module load conda\n\n")
        file.write("source activate base\n\n")
        file.write("conda activate /nfs/stak/users/thakarr/hpc-share/multi-policy-morl-env\n\n")
        file.write("cd /nfs/stak/users/thakarr/hpc-share/multi-policy-morl-env/Multi-Policy-MORL\n\n")
        file.write(f'python -m catserl.orchestrator.orchestrator --stage1-alg=td3  --save-data-dir={data_dir} --config=/nfs/stak/users/thakarr/hpc-share/multi-policy-morl-env/Multi-Policy-MORL/catserl/shared/config/halfcheetah.yaml')

    # Make the file executable
    os.chmod(script_path, 0o755)

    print(f"Generated script: {script_path}")

# Example usage:
time = "0-36:00:00"  # Set the time
env = "MO-HalfCheetah-v2"
data_dir="/nfs/stak/users/thakarr/hpc-share/multi-policy-morl-env/Multi-Policy-MORL/sampledata/"
configs=['halfcheetah']

for config in configs:
    generate_bash_scripts(env, data_dir, config)
