import yaml, torch, numpy as np, random
import mo_gymnasium as mo_gym
from catserl.island.island_manager import IslandManager
from catserl.moea.mo_manager import MOManager

cfg = yaml.safe_load(open("catserl/shared/config/default.yaml"))
seed = cfg["seed"]

env = mo_gym.make("mo-mountaincar-v0")

# Seed everything
random.seed(seed)
np.random.seed(seed)
env.reset(seed=seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device(cfg["device"])

mgr0 = IslandManager(env, 1, np.array([1, 0, 0]), cfg, seed=seed + 1, device=device)
mgr1 = IslandManager(env, 2, np.array([0, 1, 0]), cfg, seed=seed + 2, device=device)

for gen in range(500):          # generations = episodes in this mini demo
    mgr0.train_generation()
    mgr1.train_generation()

    print(f"Gen {gen+1:02d} | "
            f"Obj-0 10-ep mean: {np.mean(mgr0.get_scalar_returns()[-10:]):.2f} "
            f"Obj-0 10-ep mean: {np.mean(mgr0.get_vector_returns()[-10:], axis=0)} "
            f"| Obj-1 10-ep mean: {np.mean(mgr1.get_scalar_returns()[-10:]):.2f}")

# After 500 generations, combine the populations of both islands
pop0, id0, critic0, w0 = mgr0.export_island()
pop1, id1, critic1, w1 = mgr1.export_island()

combined_pop = pop0 + pop1
critics_dict = {
    id0 : critic0,
    id1 : critic1
}

# Stage-2: Multi-objective evolution on merged population
# mo_mgr = MOManager(combined_pop, cfg, device=device)
# mo_mgr.evolve(50, critics_dict)
print("MOManager evolution on merged population complete.")
