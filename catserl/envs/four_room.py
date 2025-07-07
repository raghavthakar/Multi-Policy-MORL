# catserl/envs/fourroom.py
from __future__ import annotations
from typing import Optional, Tuple, Any

import gymnasium as gym
import mo_gymnasium            # registers "four-room-v0" with Gymnasium


class FourRoomWrapper(gym.Wrapper):
    """
    Thin pass-through wrapper around ``four-room-v0`` from mo-gymnasium.

    • Returns the external reward vector unmodified.
    • Keeps one place to plug novelty bonuses or logging later.
    """

    ENV_ID = "four-room-v0"

    def __init__(self, seed: Optional[int] = None, **make_kwargs):
        base_env = gym.make(self.ENV_ID, **make_kwargs)
        super().__init__(base_env)

        if seed is not None:
            # Seed ONCE; subsequent episodes get stochastic variation.
            self.reset(seed=seed)

    # Optional type hints for IDEs; behaviour identical to the wrapped env.
    def reset(self, **kwargs) -> Tuple[Any, dict]:
        return super().reset(**kwargs)

    def step(self, action) -> Tuple[Any, Any, bool, bool, dict]:
        return super().step(action)
