# catserl/envs/fourroom.py
from __future__ import annotations
from typing import Optional, Tuple, Any
import numpy as np, gymnasium as gym, mo_gymnasium


class FourRoomWrapper(gym.Wrapper):
    ENV_ID = "four-room-v0"

    def __init__(self, seed: Optional[int] = None, beta: float = 0.0, **kw):
        super().__init__(gym.make(self.ENV_ID, **kw))
        self.beta = float(beta)
        self._visits: dict[Tuple, int] = {}
        if seed is not None:
            self.reset(seed=seed)

    # episode start
    def reset(self, **kw) -> Tuple[Any, dict]:
        obs, info = self.env.reset(**kw)
        self._visits.clear()
        self._visits[self._key(obs)] = 1
        return obs, info

    # main step
    def step(self, action) -> Tuple[Any, np.ndarray, bool, bool, dict]:
        obs, ext, term, trunc, info = self.env.step(action)

        # novelty counts
        k = self._key(obs)
        self._visits[k] = self._visits.get(k, 0) + 1
        intrinsic = self.beta / self._visits[k]

        ext = np.asarray(ext, dtype=np.float32)          # vector reward
        aug = ext + intrinsic                            # add bonus

        # log raw components
        info = dict(info)
        info["extrinsic"] = ext
        info["intrinsic"] = intrinsic

        return obs, aug, term, trunc, info

    # helper
    @staticmethod
    def _key(obs) -> Tuple:
        if isinstance(obs, np.ndarray):
            return tuple(obs.ravel())
        if isinstance(obs, (tuple, list)):
            return tuple(obs)
        return (obs,)
