import yaml, torch, numpy as np, random
from catserl.island.island_manager import IslandManager
from catserl.shared.utils.seeding import seed_everything

cfg = yaml.safe_load(open("catserl/shared/config/default.yaml"))
seed = cfg["seed"]
seed_everything(seed)

device = torch.device(cfg["device"])

mgr0 = IslandManager(np.array([0, 1, 0]), cfg,
                  seed=seed + 1, device=device)
# mgr1 = ERLManager(np.array([0, 1, 0]), cfg,
#                   seed=seed + 2, device=device)

for gen in range(500):          # generations = episodes in this mini demo
    mgr0.train_generation()
    # mgr1.train_generation()

    print(f"Gen {gen+1:02d} | "
            f"Obj-0 10-ep mean: {np.mean(mgr0.get_scalar_returns()[-10:]):.2f} "
            f"Obj-0 10-ep mean: {np.mean(mgr0.get_vector_returns()[-10:], axis=0)} ")
            # f"| Obj-1 10-ep mean: {np.mean(mgr1.get_scalar_returns()[-10:]):.2f}")
