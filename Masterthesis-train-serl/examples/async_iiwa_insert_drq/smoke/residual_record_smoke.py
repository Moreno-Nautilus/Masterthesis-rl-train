import numpy as np, record_demo_residual as R
class StubTeleop:
    def __init__(self): self.n=0
    def get_action(self):
        self.n+=1
        return np.array([0.2,0.1,-0.3,0.0,0.0,0.1],dtype=np.float32)  # lateral+along+rot residual
    def is_success(self): return self.n>=6
    def is_failure(self): return False
    def is_stop_forward(self): return False
    def close(self): pass
R.PS4TeleopProvider = lambda *a, **k: StubTeleop()
R.record(insert=1, assembly="plumbers_block", n_demos=1, server_url="", out_dir="/tmp/rec_smoke_out", fake_env=True)
import pickle, glob
f=sorted(glob.glob("/tmp/rec_smoke_out/*.pkl"))[-1]
d=pickle.load(open(f,"rb"))
print("SAVED", len(d), "transitions; obs keys:", list(d[0]["observations"].keys()),
      "| last reward:", d[-1]["rewards"])
