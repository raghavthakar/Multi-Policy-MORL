import os
import yaml
import copy
import itertools
import shutil
from pathlib import Path

# ====================================================
# 1. USER CONFIGURATION
# ====================================================

# ----------------- PATHS -----------------

# The "Golden Template" file that contains your default settings
TEMPLATE_CONFIG_PATH = "/nfs/stak/users/thakarr/hpc-share/multi-policy-morl-env/Multi-Policy-MORL/catserl/shared/config/model_config.yaml"

# [INPUT] The SPECIFIC 'run_0' folder you want to clone as the starting point
# Example: ".../savedata/consolidated/stage2_tuning/epoch10/run_0"
SOURCE_DATA_DIR = Path("/nfs/stak/users/thakarr/hpc-share/mopderl-env/savedata/consolidated/stage1_only/MO-Ant-v2/run_0")

# [OUTPUT] The Base location where new experiment data will be stored
# The script will create: BASE_SAVE_DATA_DIR / EXPERIMENT_TAG / <job_name> / run_0
BASE_SAVE_DATA_DIR = Path("/nfs/stak/users/thakarr/hpc-share/mopderl-env/savedata/consolidated/stage2_tuning/ant2d")

# [OUTPUT] Where the Python scripts and .sh files will be generated
EXPERIMENT_TAG = "ant_epochs_beta_tuning" 
JOB_SCRIPTS_DIR = Path(f"/nfs/stak/users/thakarr/hpc-share/multi-policy-morl-env/Multi-Policy-MORL/experiments/scripts/job_scripts/{EXPERIMENT_TAG}")


# ----------------- SWEEP GRID -----------------
# Format: "nested.key.path": [list, of, values]
experiment_grid = {
    "mopderl.finetune.awr_beta": [0.1, 1.0, 5.0],
    "mopderl.finetune.epochs": [10, 50],
    "env.name": ["mo-ant-2obj-v5"]
}

# ----------------- SLURM TEMPLATE -----------------
SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --error={log_dir}/error.log
#SBATCH --time=0-36:00:00
#SBATCH --partition=dgx2,dgxh,share,ampere
#SBATCH --constraint=skylake
#SBATCH --mem=46G
#SBATCH -c 12

module load conda
source activate base
conda activate /nfs/stak/users/thakarr/hpc-share/multi-policy-morl-env

cd /nfs/stak/users/thakarr/hpc-share/multi-policy-morl-env/Multi-Policy-MORL

# The script automatically points to the config and UNIQUE data copy for this job
python -m catserl.orchestrator.orchestrator \
    --save-data-dir={save_dir} \
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
    keys = dot_path.split('.')
    current = data
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = value

def generate_short_name(params):
    parts = []
    for k, v in params.items():
        short_key = k.split('.')[-1]
        parts.append(f"{short_key}_{v}")
    return "-".join(parts)

def main():
    print(f"--- Starting Experiment Generation: {EXPERIMENT_TAG} ---")

    # 1. Validation
    if not os.path.exists(TEMPLATE_CONFIG_PATH):
        print(f"Error: Template not found at {TEMPLATE_CONFIG_PATH}")
        return
    if not os.path.exists(SOURCE_DATA_DIR):
        print(f"Error: Source data not found at {SOURCE_DATA_DIR}")
        return

    base_config = load_yaml(TEMPLATE_CONFIG_PATH)

    # 2. Generate Combinations
    keys, values = zip(*experiment_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f"Found {len(param_combinations)} unique configurations.")

    submit_commands = []

    for i, params in enumerate(param_combinations):
        # Define Naming
        unique_id = f"job_{i:03d}__" + generate_short_name(params)
        
        # ---------------------------------------------------------
        # A. Setup Job Script Directory (Where .sh and .yaml go)
        # ---------------------------------------------------------
        script_dir = JOB_SCRIPTS_DIR / unique_id
        script_dir.mkdir(parents=True, exist_ok=True)
        
        # ---------------------------------------------------------
        # B. Setup Data Directory (Where the data is cloned to)
        # Path: BASE_SAVE_DATA_DIR / EXPERIMENT_TAG / unique_id / run_0
        # ---------------------------------------------------------
        dest_data_parent = BASE_SAVE_DATA_DIR / EXPERIMENT_TAG / unique_id
        dest_data_final = dest_data_parent / "run_0"
        
        # COPY LOGIC:
        # If the folder already exists, we skip copying to save time (or you can delete it)
        if dest_data_final.exists():
            print(f"  [Warning] Data dir exists, skipping copy: {dest_data_final}")
        else:
            print(f"  [{i}] Copying data to: {dest_data_final} ...")
            # copytree requires the parent to exist, but the destination dir itself must NOT exist
            dest_data_parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(SOURCE_DATA_DIR, dest_data_final)

        # ---------------------------------------------------------
        # C. Generate Config
        # ---------------------------------------------------------
        current_config = copy.deepcopy(base_config)
        for key_path, val in params.items():
            update_by_dot_path(current_config, key_path, val)
        
        config_save_path = dest_data_final / "config.yaml"
        save_yaml(current_config, config_save_path)

        # ---------------------------------------------------------
        # D. Generate SLURM Script
        # ---------------------------------------------------------
        script_save_path = script_dir / "submit.sh"
        
        script_content = SLURM_TEMPLATE.format(
            job_name=f"exp_{i}_{unique_id}",
            log_dir=script_dir,      # Logs go with the scripts
            save_dir=dest_data_final, # Points to the NEW cloned copy
            config_path=config_save_path
        )
        
        with open(script_save_path, "w") as f:
            f.write(script_content)
            
        submit_commands.append(f"sbatch {script_save_path}")

    # 3. Create Master Submit Script
    master_script_path = JOB_SCRIPTS_DIR / f"submit_all_{EXPERIMENT_TAG}.sh"
    with open(master_script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"# Experiment Tag: {EXPERIMENT_TAG}\n")
        f.write("\n".join(submit_commands))
    
    os.chmod(master_script_path, 0o755)

    print("-" * 40)
    print("Generation Complete.")
    print(f"1. Scripts saved in: {JOB_SCRIPTS_DIR}")
    print(f"2. Data cloned to:   {BASE_SAVE_DATA_DIR}/{EXPERIMENT_TAG}")
    print(f"3. Launch file:      {master_script_path}")
    print("-" * 40)

if __name__ == "__main__":
    main()