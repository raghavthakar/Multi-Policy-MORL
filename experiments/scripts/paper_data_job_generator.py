import os
import yaml
import copy
from pathlib import Path

# ====================================================
# 1. USER CONFIGURATION
# ====================================================

# ----------------- PATHS -----------------

# The "Golden Template" file that contains your default settings
TEMPLATE_CONFIG_PATH = Path("/nfs/stak/users/thakarr/hpc-share/multi-policy-morl-env/Multi-Policy-MORL/catserl/shared/config/model_config.yaml")

# [TARGET] The folder containing the existing runs (run_0, run_1, etc.)
# The script will look for: TARGET_ENV_DIR / run_X
TARGET_ENV_DIR = Path("/nfs/stak/users/thakarr/hpc-share/mopderl-env/savedata/consolidated/mapx/MO-Ant-v2")

# [OUTPUT] Where the Python submit scripts and logs will be generated
EXPERIMENT_TAG = "ant_fixed_beta_finetune" 
JOB_SCRIPTS_DIR = Path(f"/nfs/stak/users/thakarr/hpc-share/multi-policy-morl-env/Multi-Policy-MORL/experiments/scripts/job_scripts/{EXPERIMENT_TAG}")

# ----------------- FIXED PARAMETERS -----------------
# Specify the exact parameters you want to apply to all runs.
FIXED_PARAMS = {
    "mopderl.finetune.awr_beta": 0.5,
    "mopderl.finetune.epochs": 20,
    "mopderl.finetune.awr_clip": 1.0,
    "mopderl.finetune.pretrain_steps": 1000,
    "env.name": "mo-ant-2obj-v5"
}

# ----------------- RUN SELECTION -----------------
# Which existing runs do you want to process? e.g. [0, 1, 2, 3, 4]
RUN_IDS = [0, 1, 2, 3, 4]

# ----------------- SLURM TEMPLATE -----------------
SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --error={log_dir}/error.log
#SBATCH --output={log_dir}/output.log
#SBATCH --time=0-36:00:00
#SBATCH --partition=dgx2,dgxh,share,ampere
#SBATCH --constraint=skylake
#SBATCH --mem=46G
#SBATCH -c 12

module load conda
source activate base
conda activate /nfs/stak/users/thakarr/hpc-share/multi-policy-morl-env

cd /nfs/stak/users/thakarr/hpc-share/multi-policy-morl-env/Multi-Policy-MORL

# Points to the EXISTING data directory and the NEW config file placed inside it
python -m catserl.orchestrator.orchestrator \
    --save-data-dir={target_dir} \
    --config={config_path} \
    --resume-stage2
"""

# ====================================================
# 2. CORE LOGIC
# ====================================================

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_yaml(data, path):
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

def update_by_dot_path(data, dot_path, value):
    """Updates a nested dictionary using a dot-notation string key."""
    keys = dot_path.split('.')
    current = data
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = value

def main():
    print(f"--- Starting Fixed Parameter Run Generation: {EXPERIMENT_TAG} ---")

    # 1. Validation
    if not os.path.exists(TEMPLATE_CONFIG_PATH):
        print(f"Error: Template not found at {TEMPLATE_CONFIG_PATH}")
        return
    if not os.path.exists(TARGET_ENV_DIR):
        print(f"Error: Target Environment directory not found at {TARGET_ENV_DIR}")
        return

    base_config = load_yaml(TEMPLATE_CONFIG_PATH)
    submit_commands = []

    # 2. Iterate over existing Run IDs
    for run_id in RUN_IDS:
        run_folder_name = f"run_{run_id}"
        target_run_dir = TARGET_ENV_DIR / run_folder_name

        # A. Verify directory exists (No cloning)
        if not target_run_dir.exists():
            print(f"  [Skipping] Directory not found: {target_run_dir}")
            continue

        print(f"  [Processing] {run_folder_name} -> Applying Fixed Params")

        # B. Setup Job Script Directory
        # We create a folder for the job scripts to keep logs organized
        unique_id = f"job_run_{run_id}"
        script_dir = JOB_SCRIPTS_DIR / unique_id
        script_dir.mkdir(parents=True, exist_ok=True)

        # C. Generate & Inject Config
        # We take the template, apply fixed params, and save it INTO the existing run dir
        current_config = copy.deepcopy(base_config)
        for key_path, val in FIXED_PARAMS.items():
            update_by_dot_path(current_config, key_path, val)
        
        # NOTE: This overwrites 'config.yaml' in the existing directory.
        config_save_path = target_run_dir / "config.yaml"
        save_yaml(current_config, config_save_path)

        # D. Generate SLURM Script
        script_save_path = script_dir / "submit.sh"
        
        script_content = SLURM_TEMPLATE.format(
            job_name=f"{EXPERIMENT_TAG}_{run_id}",
            log_dir=script_dir,      
            target_dir=target_run_dir, # Points to existing dir
            config_path=config_save_path
        )
        
        with open(script_save_path, "w") as f:
            f.write(script_content)
            
        submit_commands.append(f"sbatch {script_save_path}")

    # 3. Create Master Submit Script
    if submit_commands:
        master_script_path = JOB_SCRIPTS_DIR / f"submit_all_{EXPERIMENT_TAG}.sh"
        with open(master_script_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"# Experiment: {EXPERIMENT_TAG}\n")
            f.write(f"# Fixed Params: {FIXED_PARAMS}\n")
            f.write("\n".join(submit_commands))
        
        os.chmod(master_script_path, 0o755)

        print("-" * 40)
        print("Generation Complete.")
        print(f"1. Scripts saved in: {JOB_SCRIPTS_DIR}")
        print(f"2. Configs updated in: {TARGET_ENV_DIR}/run_X")
        print(f"3. Launch file:      {master_script_path}")
        print("-" * 40)
    else:
        print("No valid run directories found. Check your paths.")

if __name__ == "__main__":
    main()