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

# Sim2real hardware variant: KUKA iiwa7 + custom Y-gripper (replaces the Franka), same cooling-screw
# task. NEW env class (InsertionEnvIiwa) that only re-derives the robot-specific geometry; the Franka
# cooling tasks above are untouched so the validated pipeline / fallback stays intact. See
# cooling_iiwa_tasks_cfg / insertion_env_iiwa.
gym.register(
    id="Isaac-Insertion-CoolingPeg-Iiwa-Direct-v0",
    entry_point=f"{__name__}.insertion_env_iiwa:InsertionEnvIiwa",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cooling_iiwa_tasks_cfg:ForgeTaskCoolingInsertIiwaCfg",
        "rl_games_cfg_entry_point": "isaaclab_tasks.direct.forge.agents:rl_games_ppo_cfg.yaml",
    },
)
gym.register(
    id="Isaac-Insertion-CoolingPeg-Iiwa-Vision-Direct-v0",
    entry_point=f"{__name__}.insertion_env_iiwa:InsertionEnvIiwa",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cooling_iiwa_tasks_cfg:ForgeTaskCoolingInsertIiwaCameraCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_camera_ppo_cfg.yaml",
    },
)

# END-TO-END VISUOMOTOR variant (supervisor-locked 2026-07-30): pure end-to-end insertion on the iiwa
# hardware -- RGB-D + force -> 5-DoF fingertip-relative delta, IK + high-gain joint PD (NOT Forge's OSC),
# -L2 + shaft-axis reward. NEW env class (InsertionEnvE2EIiwa); the residual iiwa tasks above are
# untouched. The STATE task is the state-first debug scaffold (true-delta obs, no camera) to validate the
# control loop before switching to the vision encoder. See cooling_iiwa_tasks_cfg / insertion_env_e2e_iiwa
# / PLANNING.md "END-TO-END VISUOMOTOR INSERTION".
gym.register(
    id="Isaac-Insertion-CoolingPeg-Iiwa-E2E-Direct-v0",
    entry_point=f"{__name__}.insertion_env_e2e_iiwa:InsertionEnvE2EIiwa",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cooling_iiwa_tasks_cfg:ForgeTaskCoolingInsertIiwaE2ECfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_e2e_state_ppo_cfg.yaml",
    },
)
gym.register(
    id="Isaac-Insertion-CoolingPeg-Iiwa-E2E-Vision-Direct-v0",
    entry_point=f"{__name__}.insertion_env_e2e_iiwa:InsertionEnvE2EIiwa",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cooling_iiwa_tasks_cfg:ForgeTaskCoolingInsertIiwaE2EVisionCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_e2e_vision_ppo_cfg.yaml",
    },
)
# Same E2E vision env, PLUS the EXPLICIT ESTIMATOR: aux head predicts the hole gap (aux_label[1:4]) and
# feeds the prediction into the policy. Uses the estimator agent yaml (aux_head + aux_label in obs_groups);
# the run must also pass env.e2e_keep_aux_label=True so the env exposes the aux_label obs group.
gym.register(
    id="Isaac-Insertion-CoolingPeg-Iiwa-E2E-Vision-Estimator-Direct-v0",
    entry_point=f"{__name__}.insertion_env_e2e_iiwa:InsertionEnvE2EIiwa",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cooling_iiwa_tasks_cfg:ForgeTaskCoolingInsertIiwaE2EVisionCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_e2e_vision_estimator_ppo_cfg.yaml",
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

# 2nd generalization target ("across parts + direction"): pb-assembly HORIZONTAL PIPE insertion (pb_pipe
# driven horizontally into pb_base's Y through-bore). Modelled EXACTLY like the cooling E2E-iiwa visuomotor
# task (same iiwa7+pdz robot, IK+PD controller, RGB wrist cam, appearance/dynamics DR, goal-anchor noise,
# squashing reward + all curricula); only the parts + incoming direction change (pb_pipe_horiz_tasks_cfg /
# assets_cfg). InsertionEnvE2EIiwa is reused unchanged. NEEDS a render/smoke pass before training (geometry
# marked TODO-tune). See memory pb-pipe-horizontal-geometry.
gym.register(
    id="Isaac-Insertion-PbPipe-Iiwa-E2E-Direct-v0",  # STATE-first debug scaffold (no camera)
    entry_point=f"{__name__}.insertion_env_e2e_iiwa:InsertionEnvE2EIiwa",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pb_pipe_horiz_tasks_cfg:ForgeTaskPbPipeHorizInsertIiwaE2ECfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_e2e_state_ppo_cfg.yaml",
    },
)
gym.register(
    id="Isaac-Insertion-PbPipe-Iiwa-E2E-Vision-Direct-v0",  # the weekend training target
    entry_point=f"{__name__}.insertion_env_e2e_iiwa:InsertionEnvE2EIiwa",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pb_pipe_horiz_tasks_cfg:ForgeTaskPbPipeHorizInsertIiwaE2EVisionCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_e2e_vision_ppo_cfg.yaml",
    },
)
