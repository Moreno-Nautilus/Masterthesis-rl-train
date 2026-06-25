"""Asset configs for the cooling insertion parts (held screw + fixed base).

USD assets live in ``insertion_policy/assets`` (baked by ``scripts/convert_assets.py``):
centered-mesh origin, meters, SDF collision, rigid + mass props.
"""

import os

from isaaclab_tasks.direct.factory.factory_tasks_cfg import FixedAssetCfg, HeldAssetCfg

from isaaclab.utils import configclass

# insertion_policy/assets (this file is at insertion_policy/tasks/insertion/)
ASSET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "assets"))


@configclass
class CoolingScrew(HeldAssetCfg):
    """Held part: shoulder-bolt screw. 20mm head over a 12mm shaft, 35mm tall."""

    usd_path = f"{ASSET_DIR}/cooling_screw.usd"
    diameter = 0.020  # head diameter, used for gripper width (grip across the head)
    height = 0.035  # full part height
    mass = 0.0064  # from mesh volume @ 1 g/cm^3


@configclass
class CoolingBase(FixedAssetCfg):
    """Fixed receptacle: 120x120mm plate with a raised 80x80 platform holding 2 sockets."""

    usd_path = f"{ASSET_DIR}/cooling_base.usd"
    diameter = 0.014  # socket diameter (12mm shaft + ~2mm clearance)
    # With InsertionEnv redefining fixed_pos as the socket BOTTOM, height = socket depth makes the
    # Factory "tip" land at the socket opening (entry), and base_height=0 keeps target = bottom.
    height = 0.0175  # socket depth (bottom -> opening)
    base_height = 0.0
    # Heavy + high-friction so insertion/jamming contact forces can't slide the base on the table
    # ("glued" in practice). Kept dynamic (not kinematic) so it stays a valid articulation and is
    # still randomized into place each reset. The screw is ~6g, so 10kg is effectively immovable.
    mass = 10.0
    friction = 1.0
