"""Offline dry-run of the franka_plumbers learner init (no robot / no camera / no PS4).

Mirrors train_rlpd.py::main's learner path with fake_env=True:
  1. build the env (fake_env -> no hardware),
  2. construct the SACAgent (loads the frozen pretrained ResNet-10 encoder),
  3. run one sample_actions forward pass.

Run under the franka_env.sh shell:
    source franka_env.sh
    python hil-serl_src/examples/experiments/franka_plumbers/dryrun_learner.py
"""

import jax
import jax.numpy as jnp

from experiments.mappings import CONFIG_MAPPING
from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.utils.launcher import make_sac_pixel_agent


def main(exp_name="franka_plumbers_insert0", seed=0):
    print(f"[dryrun] exp={exp_name}")
    config = CONFIG_MAPPING[exp_name]()

    env = config.get_environment(fake_env=True, save_video=False, classifier=True)
    print("[dryrun] env built (fake_env=True)")
    print("         obs_space keys:", list(env.observation_space.spaces.keys()))
    print("         action_space:", env.action_space)

    assert config.setup_mode in ("single-arm-fixed-gripper", "dual-arm-fixed-gripper"), (
        f"expected fixed-gripper setup, got {config.setup_mode}"
    )
    agent: SACAgent = make_sac_pixel_agent(
        seed=seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=config.image_keys,
        encoder_type=config.encoder_type,
        discount=config.discount,
    )
    print("[dryrun] SACAgent constructed; ResNet-10 encoder loaded")

    rng = jax.random.PRNGKey(seed)
    obs = env.observation_space.sample()
    actions = agent.sample_actions(observations=jax.device_put(obs), seed=rng)
    actions = jax.device_get(actions)
    print("[dryrun] sample_actions OK -> shape", jnp.asarray(actions).shape)
    print("[dryrun] PASS")


if __name__ == "__main__":
    import sys
    main(*(sys.argv[1:] or ["franka_plumbers_insert0"]))
