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

# Vision variant: same env class, but the cfg adds a wrist RGB-D camera and the agent uses the
# hybrid CNN+proprio rl_games network. Run training/eval with --enable_cameras.
gym.register(
    id="Isaac-Insertion-CoolingPeg-Vision-Direct-v0",
    entry_point=f"{__name__}.insertion_env:InsertionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cooling_tasks_cfg:ForgeTaskCoolingInsertCameraCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_camera_ppo_cfg.yaml",
    },
)

# Generalization target ("across parts"): pb-assembly SCREW insertion (pb_screw into pb_base's Ø10 hole).
# Same InsertionEnv; only the parts/geometry differ (pb_tasks_cfg). Vision variant reuses the cooling
# camera + appearance-DR + aux-head agent config wholesale. FIRST CUT (base alone, geometry TODO-tuned);
# needs a smoke/viz pass before training. See pb_tasks_cfg / assets_cfg / memory pb-screw-task-design.
gym.register(
    id="Isaac-Insertion-PbScrew-Direct-v0",
    entry_point=f"{__name__}.insertion_env:InsertionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pb_tasks_cfg:ForgeTaskPbScrewInsertCfg",
        "rl_games_cfg_entry_point": "isaaclab_tasks.direct.forge.agents:rl_games_ppo_cfg.yaml",
    },
)
gym.register(
    id="Isaac-Insertion-PbScrew-Vision-Direct-v0",
    entry_point=f"{__name__}.insertion_env:InsertionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pb_tasks_cfg:ForgeTaskPbScrewInsertCameraCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_camera_ppo_cfg.yaml",
    },
)
