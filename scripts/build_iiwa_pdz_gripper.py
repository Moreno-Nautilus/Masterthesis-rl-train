"""Graft the NEW pdz_gripper onto the validated vendor iiwa7 arm -> one articulation USD.

Preserves the exact vendor iiwa7 arm (base_link..link7 + joint1..7) from kuka_iiwa7_y_gripper_src.usda,
REMOVES the old Y-gripper subtree, and grafts the converted pdz_gripper.usd links (base + L/R fingers +
tcp + D405 camera frames) at link7 via a near-massless ``force_sensor`` link (ForgeEnv reads the wrist
wrench from body_names.index("force_sensor")). Adds the finger prismatic joints (right mimics left) with
PD drives, sets the gripper total mass ~0.71 kg with per-link (asymmetric) inertia.

    OMNI_KIT_ACCEPT_EULA=YES python scripts/build_iiwa_pdz_gripper.py
"""
import os, argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RB = os.path.join(ROOT, "source/insertion_policy/insertion_policy/assets/robots")
SRC = os.path.join(RB, "kuka_iiwa7_y_gripper_src.usda")
PDZ = os.path.join(RB, "pdz_gripper.usd")
OUT = os.path.join(RB, "kuka_iiwa7_pdz_gripper.usd")
A = "/kuka_iiwa7_y_gripper"  # articulation root xform


def log(m):
    print(f"[merge] {m}", flush=True)


stage = Usd.Stage.Open(SRC)

# --- 0. discover joint container + link7/flange default world pose --------------------------------
# Capture joint PATHS as strings (prim handles expire after edits -> RuntimeError).
joint_paths = [p.GetPath().pathString for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
log(f"found {len(joint_paths)} joints; sample: {[p.split('/')[-1] for p in joint_paths[:6]]}")
joint_parent = joint_paths[0].rsplit("/", 1)[0] if joint_paths else A
log(f"joint container = {joint_parent}")

xf_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
gbl = stage.GetPrimAtPath(f"{A}/gripper_base_link")
flange_w = xf_cache.GetLocalToWorldTransform(gbl)  # old gripper base = the flange mount pose
log(f"flange (old gripper_base) world T translate = {flange_w.ExtractTranslation()}")

# capture the 4 REAL pdz links' LOCAL transform (relative to pdz base = flange) from the converted USD
GRAFT = ["pdz_gripper_base_link", "pdz_gripper_left_finger_link",
         "pdz_gripper_right_finger_link", "pdz_gripper_tcp"]
pdz = Usd.Stage.Open(PDZ)
pxf = UsdGeom.XformCache(Usd.TimeCode.Default())
PDZ_A = "/pdz_gripper"
base_w_inv = pxf.GetLocalToWorldTransform(pdz.GetPrimAtPath(f"{PDZ_A}/pdz_gripper_base_link")).GetInverse()
link_local = {}
for name in GRAFT:
    p = pdz.GetPrimAtPath(f"{PDZ_A}/{name}")
    link_local[name] = pxf.GetLocalToWorldTransform(p) * base_w_inv  # pose relative to pdz base
log(f"pdz links to graft: {list(link_local)}")

# --- 1. remove the OLD gripper subtree + its joints (by PATH) ------------------------------------
for name in ["gripper_base_link", "left_finger_link", "right_finger_link", "gripper_tcp"]:
    if stage.GetPrimAtPath(f"{A}/{name}"):
        stage.RemovePrim(f"{A}/{name}")
        log(f"removed old prim {name}")
for jpath in joint_paths:
    if jpath.rsplit("/", 1)[-1] in ("gripper_mount_joint", "gripper_tcp_joint", "left_finger_joint", "right_finger_joint"):
        stage.RemovePrim(jpath)
        log(f"removed old joint {jpath.rsplit('/', 1)[-1]}")

# --- 2. force_sensor link at the flange (near-massless wrist wrench body) -------------------------
fs_path = f"{A}/force_sensor"
fs = UsdGeom.Xform.Define(stage, fs_path)
fsp = fs.GetPrim()
UsdPhysics.RigidBodyAPI.Apply(fsp)
m = UsdPhysics.MassAPI.Apply(fsp)
m.CreateMassAttr(1e-4)
tr = flange_w.ExtractTranslation()
q = flange_w.ExtractRotationQuat()
fs.AddTranslateOp().Set(tr)
fs.AddOrientOp().Set(Gf.Quatf(q.GetReal(), *q.GetImaginary()))
log("added force_sensor")

# --- 3. graft pdz links as flat siblings (reference their geometry), positioned at flange ⊗ local -
# masses ~0.71kg total, asymmetric per README (base+motor heavy, fingers light, camera 0.072).
MASS = {"pdz_gripper_base_link": 0.52, "pdz_gripper_left_finger_link": 0.05,
        "pdz_gripper_right_finger_link": 0.05, "camera_link": 0.072}
INERTIA = {"pdz_gripper_base_link": (0.0016, 0.0011, 0.0019),
           "pdz_gripper_left_finger_link": (6e-5, 5e-5, 3e-5),
           "pdz_gripper_right_finger_link": (6e-5, 5e-5, 3e-5),
           "camera_link": (3.88e-3, 4.99e-4, 3.88e-3)}
for name, rel in link_local.items():
    tgt = f"{A}/{name}"
    p = UsdGeom.Xform.Define(stage, tgt).GetPrim()
    p.GetReferences().AddReference(PDZ, f"{PDZ_A}/{name}")
    w = rel * flange_w  # graft world pose = flange ⊗ (link relative to pdz base)
    xp = UsdGeom.Xformable(p)
    xp.ClearXformOpOrder()  # override the referenced quatd ops with one double matrix op (avoids precision clash)
    xp.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(w)
    if name in MASS:
        mp = UsdPhysics.MassAPI.Apply(p)
        mp.CreateMassAttr(MASS[name])
        ix, iy, iz = INERTIA[name]
        mp.CreateDiagonalInertiaAttr(Gf.Vec3f(ix, iy, iz))
log("grafted pdz links")


def _q(m):  # Gf.Matrix4d rotation -> Gf.Quatf
    q = m.ExtractRotationQuat(); im = q.GetImaginary()
    return Gf.Quatf(q.GetReal(), float(im[0]), float(im[1]), float(im[2]))


def _anchors(j, p0, p1):
    """Lock the joint at the bodies' CURRENT relative pose: frame0=identity on body0, frame1=Wp*Wc^-1 on body1."""
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    Wp = xc.GetLocalToWorldTransform(stage.GetPrimAtPath(p0))
    Wc = xc.GetLocalToWorldTransform(stage.GetPrimAtPath(p1))
    L1 = Wp * Wc.GetInverse()
    j.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0)); j.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
    t = L1.ExtractTranslation()
    j.CreateLocalPos1Attr(Gf.Vec3f(float(t[0]), float(t[1]), float(t[2]))); j.CreateLocalRot1Attr(_q(L1))


def _fixed(name, b0, b1):
    j = UsdPhysics.FixedJoint.Define(stage, f"{joint_parent}/{name}")
    j.CreateBody0Rel().SetTargets([b0]); j.CreateBody1Rel().SetTargets([b1])
    _anchors(j, b0, b1)
    return j


# Fingers: origins coincide with base at q=0, so anchors are at the base origin; the ±X direction is baked
# into the anchor rotation (right = +X identity; left = -X via 180deg about Z) since USD's axis token is unsigned.
_R180Z = Gf.Quatf(0, 0, 0, 1)


def _prismatic(name, b0, b1, rot, lo=0.0, hi=0.032):
    j = UsdPhysics.PrismaticJoint.Define(stage, f"{joint_parent}/{name}")
    j.CreateBody0Rel().SetTargets([b0]); j.CreateBody1Rel().SetTargets([b1])
    j.CreateAxisAttr("X")
    j.CreateLowerLimitAttr(lo); j.CreateUpperLimitAttr(hi)
    j.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0)); j.CreateLocalRot0Attr(rot)
    j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0)); j.CreateLocalRot1Attr(rot)
    d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
    d.CreateTypeAttr("force"); d.CreateStiffnessAttr(2000.0); d.CreateDampingAttr(100.0)
    d.CreateMaxForceAttr(200.0); d.CreateTargetPositionAttr(0.0)
    return j


# --- 4. joints: link7->force_sensor, force_sensor->pdz base, fingers (prismatic ±X), tcp ---------
_fixed("gripper_mount_joint", f"{A}/link7", fs_path)
_fixed("gripper_base_mount_joint", fs_path, f"{A}/pdz_gripper_base_link")
_fixed("pdz_gripper_tcp_joint", f"{A}/pdz_gripper_base_link", f"{A}/pdz_gripper_tcp")
_prismatic("pdz_gripper_left_finger_joint", f"{A}/pdz_gripper_base_link", f"{A}/pdz_gripper_left_finger_link", _R180Z)
_prismatic("pdz_gripper_right_finger_joint", f"{A}/pdz_gripper_base_link", f"{A}/pdz_gripper_right_finger_link", Gf.Quatf(1, 0, 0, 0))
log("added joints + finger drives")

stage.GetRootLayer().Export(OUT)
log(f"EXPORTED {OUT}")

# verify
v = Usd.Stage.Open(OUT)
links = [p.GetName() for p in v.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
joints = [p.GetName() for p in v.Traverse() if p.IsA(UsdPhysics.Joint)]
log(f"RESULT links({len(links)}): {links}")
log(f"RESULT joints({len(joints)}): {joints}")
simulation_app.close()
