# #5: register with gymnasium (the vendored HIL serl_launcher stack is gymnasium-based).
from gymnasium.envs.registration import register


register(
    id="IiwaInsertReal-Vision-v0",
    entry_point="iiwa_serl.envs:KukaIiwaInsertionEnv",
    max_episode_steps=100,
)

register(
    id="IiwaInsertReal-State-v0",
    entry_point="iiwa_serl.envs:KukaIiwaInsertionEnv",
    kwargs={"include_image": False},
    max_episode_steps=100,
)

# Trivial reach-to-target task (rig hello-world): sparse position-only reward.
# Use to prove RLPD learns on hardware before attempting the hard insertion.
register(
    id="IiwaReachReal-Vision-v0",
    entry_point="iiwa_serl.envs:KukaIiwaInsertionEnv",
    kwargs={"reward_mode": "reach"},
    max_episode_steps=100,
)

register(
    id="IiwaReachReal-State-v0",
    entry_point="iiwa_serl.envs:KukaIiwaInsertionEnv",
    kwargs={"include_image": False, "reward_mode": "reach"},
    max_episode_steps=100,
)

register(
    id="IiwaInsertIsaac-Vision-v0",
    entry_point="iiwa_serl.envs:IsaacLabInsertionSERLEnv",
    max_episode_steps=100,
)

register(
    id="IiwaInsertIsaac-State-v0",
    entry_point="iiwa_serl.envs:IsaacLabInsertionSERLEnv",
    kwargs={"include_image": False},
    max_episode_steps=100,
)
