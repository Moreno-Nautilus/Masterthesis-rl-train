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


# ---------------------------------------------------------------------------------------------------
# pb-assembly parts (generalization target — "across parts"). The pb assembly is multi-step (pipe->base,
# top->base alignment, then 2 screws through the top+base holes); the SCREW insertion is the first task
# we model. DESIGN (decided 2026-06-29): the pipe+top steps are taken as DONE — pb_top is pre-assembled
# flat on pb_base's top face (center offset ~+0.0455 m in z), holes aligned, and the BASE+TOP stack is the
# single immovable receptacle. The screw (Ø9.5 shank ~47mm) drops through pb_top (26mm) into pb_base's
# aligned hole (~21mm) and seats HEAD-FLUSH on the top plate. Geometry (trimesh, all-axis section scan):
#   pb_top : Ø10 through-holes at x=+/-40mm, y=0, along Z (26mm plate).
#   pb_base: Ø10 holes at x=+/-40mm, y=0 in the UPPER region (z>+5mm) -> the screw sockets, aligned with
#            pb_top; (also Ø10 at x=+/-65mm lower down = mounting, NOT the screw path).
# TODO (morning, needs GPU smoke): spawn base+top as the assembled fixed receptacle; socket_offsets_local
# at x=+/-0.040; success = head-flush (not cooling's tip-at-bottom); reward geometry adapted; viz/smoke.
# ---------------------------------------------------------------------------------------------------
@configclass
class PbScrew(HeldAssetCfg):
    """Held part: pb assembly screw. ~Ø9.5mm round shank (~47mm) under a 28.8x25mm head; 70mm tall."""

    usd_path = f"{ASSET_DIR}/pb_screw.usd"
    diameter = 0.025  # head grip width (head is 28.8x25mm; pads close across ~the 25mm faces) -- VERIFY in viz_camera
    height = 0.070  # full part height; shank (insertable) length ~0.047 -> head_height ~0.023 for the grasp formula
    mass = 0.0162  # from mesh volume @ 1 g/cm^3 (16.2 g)


@configclass
class PbTop(FixedAssetCfg):
    """Fixed receptacle (screw step): 100x40x26mm top plate with 2 Ø10mm through-holes at x=+/-40mm."""

    usd_path = f"{ASSET_DIR}/pb_top.usd"
    diameter = 0.010  # hole diameter (Ø9.5mm shank + ~0.5mm clearance -> tighter than cooling's 1mm)
    height = 0.026  # plate thickness = through-hole depth (seating model TBD: head-flush vs deeper socket)
    base_height = 0.0
    mass = 10.0  # glued/immovable like CoolingBase (real ~59g); kept dynamic + high-friction
    friction = 1.0


@configclass
class PbBase(FixedAssetCfg):
    """Fixed receptacle (FIRST-CUT screw target): 160x40x65mm bracket with Ø10mm screw holes at x=+/-40mm
    in its upper region (these align with PbTop's holes; the x=+/-65mm holes are mounting, not the screw
    path). Used alone for the first smoke; the assembled pb_top on top is the documented next step.

    ALSO the fixed receptacle for the HORIZONTAL PIPE task (pb_pipe_horiz_tasks_cfg). For that task the SAME
    bracket exposes a large THROUGH-BORE running along its local Y (the 4cm-thin axis): measured Ø ~5.6cm at
    stations near x=+/-5cm (mesh scan 2026-08-28). The `diameter`/`height` below are the SCREW-task values;
    the pipe task OVERRIDES them via PbBasePipeSocket (a different bore) so this class stays screw-valid."""

    usd_path = f"{ASSET_DIR}/pb_base.usd"
    diameter = 0.010  # Ø10 screw socket (Ø9.5 shank + ~0.5mm clearance)
    height = 0.0275  # socket depth (blind hole bottom ~z+5mm -> opening at base top z+32.5mm) -- TODO-tune at smoke
    base_height = 0.0
    mass = 10.0  # glued/immovable (real ~236g); kept dynamic + high-friction like CoolingBase
    friction = 1.0


# ---------------------------------------------------------------------------------------------------
# HORIZONTAL PIPE insertion (2nd generalization target, 2026-08-28): pb_pipe driven HORIZONTALLY into
# pb_base's through-bore. Measured from the raw meshes (scripts scan; units cm -> x0.01 = m):
#   pb_pipe: hollow flattened TUBE, LONG AXIS = local Y (80mm). Cross-section flattened -> +/-X faces are
#            the FLATS (42.7mm apart, the grasp faces), +/-Z is taller (49.5mm); OD ~49.5mm, ID ~22.4mm.
#   pb_base: THROUGH-BORE along local Y (the 40mm-thin axis), Ø ~56mm at stations near x=+/-50mm, z~center.
# The gripper grasps the pipe across its FLAT +/-X faces near ONE Y-end, from above (gripper Z down); the
# pipe's long axis is then ~90deg to gripper Z (sticks out sideways = horizontal). The base is placed with
# its long axis (local Y=bore axis... see note) oriented so the bore mouth faces the incoming pipe.
# Geometry marked TODO-tune are first-bring-up approximations, refined once the env smoke-steps + renders.
# ---------------------------------------------------------------------------------------------------
@configclass
class PbPipe(HeldAssetCfg):
    """Held part: pb assembly PIPE. Hollow flattened tube, long axis = mesh-local Y (80mm), OD ~49.5mm /
    ID ~22.4mm, +/-X faces flat (42.7mm across = the grasp width)."""

    usd_path = f"{ASSET_DIR}/pb_pipe.usd"
    # `diameter` = gripper width -> the jaws close across the FLAT +/-X faces (42.7mm). VERIFY in viz.
    diameter = 0.0427
    # `height` in the Factory grasp formula = the part length ALONG the grasped/insertion axis. For the pipe
    # the insertable length is the tube length (80mm) but we grasp near one END, so the protruding insertable
    # stub is ~half. Set to the full 80mm here; the grasp offset (franka_fingerpad_length) + hand_init trim
    # the effective TCP->tip at smoke. TODO-tune against render_grasp_grid.
    height = 0.080
    # Mesh volume ~113 cm^3 -> ~113g if solid @1g/cm^3; the real print is hollow-walled (lighter). The rigid
    # WELD carries it regardless, so this is not load-bearing for control. TODO-set from the real part mass.
    mass = 0.113


@configclass
class PbBasePipeSocket(PbBase):
    """pb_base as the HORIZONTAL-PIPE receptacle: same USD/mass/friction as PbBase, but the socket geometry
    describes the large Y through-bore (not the Ø10 screw hole). Ø ~56mm bore vs the pipe OD ~49.5mm ->
    radial clearance ~3mm (looser than cooling's ~1mm). height = the seat DEPTH along the bore. Both
    diameter + height are mesh-scan first cuts -> TODO-tune at the render/smoke pass."""

    diameter = 0.056   # bore Ø (pipe OD 49.5 + ~2x3mm clearance) -- TODO-tune at smoke
    height = 0.030     # seat depth along the bore (how far the pipe drives in) -- TODO-tune at smoke
    base_height = 0.0
