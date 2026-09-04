import os, numpy as np
os.environ["RESIDUAL_ENABLE"]="1"
from experiments.mappings import CONFIG_MAPPING
from iiwa_serl.envs.hil_wrappers import PS4Intervention
class Stub:
    def get_action(self): return np.zeros(6,dtype=np.float32)
    def is_success(self): return False
    def is_failure(self): return False
    def is_stop_forward(self): return False
    def close(self): pass
_o=PS4Intervention.__init__
PS4Intervention.__init__=lambda self,env,teleop=None,joystick_index=0:_o(self,env,teleop=Stub())
import jax
from serl_launcher.utils.launcher import make_sac_pixel_agent

cfg=CONFIG_MAPPING["iiwa_plumbers_insert1"]()
env=cfg.get_environment(fake_env=True)
print("env built. setup_mode:",cfg.setup_mode,"image_keys:",cfg.image_keys)
rng=jax.random.PRNGKey(0)
agent=make_sac_pixel_agent(
    seed=0,
    sample_obs=env.observation_space.sample(),
    sample_action=env.action_space.sample(),
    image_keys=cfg.image_keys,
    encoder_type=cfg.encoder_type,
    discount=cfg.discount,
)
print("LEARNER AGENT CONSTRUCTED OK:", type(agent).__name__)
# one sample_actions to prove forward pass
a=agent.sample_actions(observations=jax.device_put(env.observation_space.sample()), seed=rng)
print("sample_actions OK, shape", np.asarray(a).shape)
