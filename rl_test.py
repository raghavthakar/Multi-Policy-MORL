import yaml, torch, numpy as np
from catserl.erl_manager import ERLManager

cfg = yaml.safe_load(open("catserl/config/default.yaml"))
device = torch.device(cfg["device"])

mgr0 = ERLManager(np.array([1,0,0]), cfg, seed=42, device=device)
mgr1 = ERLManager(np.array([0,1,0]), cfg, seed=43, device=device)

for gen in range(5000):          # generations = episodes in this mini demo
    mgr0.train_generation()
    mgr1.train_generation()

    print(f"Gen {gen+1:02d} | "
            f"Obj-0 10-ep mean: {np.mean(mgr0.get_scalar_returns()[-10:]):.2f} "
            f"| Obj-1 10-ep mean: {np.mean(mgr1.get_scalar_returns()[-10:]):.2f}")
