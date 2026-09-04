import os, numpy as np
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
print("all iiwa keys:", sorted(k for k in CONFIG_MAPPING if k.startswith("iiwa")))
for name in ["iiwa_plumbers_insert2","iiwa_cooling_insert0"]:
    for mode,label in [("1","RESIDUAL"),("0","E2E")]:
        os.environ["RESIDUAL_ENABLE"]=mode
        env=CONFIG_MAPPING[name]().get_environment(fake_env=True)
        o,_=env.reset()
        o,r,term,trunc,info=env.step(np.zeros(env.action_space.shape,dtype=np.float32))
        nk={k:(round(v,3) if isinstance(v,float) else v) for k,v in info.items() if k.startswith("nominal")}
        be=env.unwrapped
        print(f"[{name} {label}] obs={sorted(o.keys())} act={env.action_space.shape} "
              f"term={term} trunc={trunc} maxlen={be.config.max_episode_length} nominal={nk}")
print("\nCONSTRUCT-SMOKE PASSED (both assemblies, both modes)")
