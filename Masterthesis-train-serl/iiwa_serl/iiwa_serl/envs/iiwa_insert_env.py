from __future__ import annotations

from iiwa_serl.config import IiwaInsertionConfig
from iiwa_serl.envs.base_real_env import BaseIiwaSERLEnv


class KukaIiwaInsertionEnv(BaseIiwaSERLEnv):
    def __init__(
        self,
        config: IiwaInsertionConfig | None = None,
        include_image: bool = True,
        fake_env: bool = False,
        reward_mode: str | None = None,
    ):
        config = config or IiwaInsertionConfig()
        if reward_mode is not None:
            config.reward_mode = reward_mode
        super().__init__(
            config=config,
            include_image=include_image,
            fake_env=fake_env,
        )
