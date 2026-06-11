"""Cooling-screw insertion task (forked from Forge), registered as a Gym environment."""

import gymnasium as gym

# Our InsertionEnv subclasses ForgeEnv (socket-aware targeting + neg-L2 residual reward);
# reuse Forge's RL-Games PPO config.
gym.register(
    id="Isaac-Insertion-CoolingPeg-Direct-v0",
    entry_point=f"{__name__}.insertion_env:InsertionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cooling_tasks_cfg:ForgeTaskCoolingInsertCfg",
        "rl_games_cfg_entry_point": "isaaclab_tasks.direct.forge.agents:rl_games_ppo_cfg.yaml",
    },
)
