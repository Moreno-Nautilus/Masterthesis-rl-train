"""Bake the shipped KUKA iiwa7 + custom-Y-gripper USD into an Isaac-Lab/Forge-ready robot asset.

The vendor USD (assets/robots/kuka_iiwa7_y_gripper_src.usda) is already a valid articulation
(base_link = ArticulationRoot, 7 iiwa revolutes + a mimic'd parallel Y-gripper, gripper_tcp frame,
convex-hull collisions, and even the real D405 camera prim). It is missing ONE thing Forge's env
hardcodes: a rigid body literally named ``force_sensor`` (ForgeEnv reads the wrist wrench from
``body_names.index("force_sensor")`` -> ``get_link_incoming_joint_force``). The Franka asset carries a
dedicated massless wrist link for this; we inject the equivalent here.

We insert ``force_sensor`` INTO the load path between link7 and gripper_base_link:

    link7 --[gripper_mount_joint (fixed)]--> force_sensor --[gripper_base_mount_joint (fixed)]--> gripper_base_link

so ``force_sensor``'s incoming joint force = the wrench through the flange = everything below it
(gripper + fingers + grasped screw + contacts) = the wrist F/T the policy senses. It is near-massless
so it adds no dynamics. Exports a binary ``kuka_iiwa7_y_gripper.usd`` and prints the body/joint tree.

    OMNI_KIT_ACCEPT_EULA=YES python scripts/build_iiwa_gripper_usd.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

ROBOTS = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "source",
        "insertion_policy",
        "insertion_policy",
        "assets",
        "robots",
    )
)
SRC = os.path.join(ROBOTS, "kuka_iiwa7_y_gripper_src.usda")
DST = os.path.join(ROBOTS, "kuka_iiwa7_y_gripper.usd")

REPORT = []


def emit(s):
    REPORT.append(str(s))


def main():
    stage = Usd.Stage.Open(SRC)
    root = "/kuka_iiwa7_y_gripper"
    joints = f"{root}/joints"

    link7 = stage.GetPrimAtPath(f"{root}/link7")
    gripper_base = stage.GetPrimAtPath(f"{root}/gripper_base_link")
    mount = stage.GetPrimAtPath(f"{joints}/gripper_mount_joint")
    assert link7 and gripper_base and mount, "expected vendor prim tree not found"

    # Drop the vendor's world_fixed_base_joint: a PhysicsFixedJoint with an EMPTY body0 (fixed to the
    # world at the global origin). Isaac Lab's env cloner can't re-anchor it per-env ("cloning joints
    # without a body rel ... localPose won't be updated" + "disjointed body transforms"), so in cloned
    # envs the base snaps back toward the global origin and the reset IK never reaches the socket. We fix
    # the base the Isaac Lab way instead: ArticulationRootPropertiesCfg(fix_root_link=True) in the cfg.
    wfb = stage.GetPrimAtPath(f"{joints}/world_fixed_base_joint")
    if wfb and wfb.IsValid():
        stage.RemovePrim(wfb.GetPath())
        emit("removed world_fixed_base_joint (base is fixed via fix_root_link=True in the cfg)")

    fs_path = f"{root}/force_sensor"
    if stage.GetPrimAtPath(fs_path).IsValid():
        emit("force_sensor already present -> skipping injection")
    else:
        # gripper_base sits coincident with the link7 flange (mount joint is identity/zero-offset),
        # so the force_sensor body goes at the same world pose.
        gb_xf = UsdGeom.Xformable(gripper_base)
        translate = (0.0, 0.0, 0.0)
        for op in gb_xf.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                t = op.Get()
                translate = (float(t[0]), float(t[1]), float(t[2]))

        fs = UsdGeom.Xform.Define(stage, fs_path)
        fs_prim = fs.GetPrim()
        fs.AddTranslateOp().Set(Gf.Vec3d(*translate))
        fs.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        fs.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))

        # Near-massless rigid body (a pure F/T frame): adds a joint-force readout, no real dynamics.
        UsdPhysics.RigidBodyAPI.Apply(fs_prim)
        PhysxSchema.PhysxRigidBodyAPI.Apply(fs_prim)
        mass_api = UsdPhysics.MassAPI.Apply(fs_prim)
        mass_api.CreateMassAttr(1e-4)
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-6, 1e-6, 1e-6))
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))

        # Re-parent the flange fixed joint onto force_sensor (was -> gripper_base_link).
        mount.GetRelationship("physics:body1").SetTargets([fs_path])

        # New fixed joint force_sensor -> gripper_base_link (keeps the gripper at its original pose).
        gbj = UsdPhysics.FixedJoint.Define(stage, f"{joints}/gripper_base_mount_joint")
        gbj.CreateBody0Rel().SetTargets([fs_path])
        gbj.CreateBody1Rel().SetTargets([f"{root}/gripper_base_link"])
        gbj.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        gbj.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        gbj.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        gbj.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        emit(f"injected force_sensor at translate={translate} + re-routed gripper_mount_joint")

    # Rebuild BOTH finger joints to Factory's convention: joint=0 -> CLOSED (jaws at centre), joint=0.04
    # -> OPEN, and the SAME joint value moves both jaws symmetrically (Forge drives both finger DOFs to
    # the same target). Mesh scan: each finger's gripping face sits at y=+/-0.042 at joint 0 (retracted)
    # and only reaches centre at joint 0.04 -> the vendor convention is INVERTED (0=open), so Forge's
    # "drive both to 0 = close" actually OPENED the gripper and the screw was never gripped. We bake the
    # closed pose as joint 0 via localPos0, and flip the LEFT finger's slide to -Y (axis token "Y" + a
    # 180deg-about-X rotation on BOTH joint frames -> slides along world -Y while leaving the finger
    # UNROTATED). Also drop the broken mimic and give each a force drive (target 0 = closed).
    FLIP_X = Gf.Quatf(0.0, 1.0, 0.0, 0.0)   # 180 deg about X
    IDENT = Gf.Quatf(1.0, 0.0, 0.0, 0.0)

    def _fix_finger(name, local_pos0_y, flip):
        j = stage.GetPrimAtPath(f"{joints}/{name}")
        if not (j and j.IsValid()):
            return
        pj = UsdPhysics.PrismaticJoint(j)
        pj.CreateAxisAttr("Y")
        pj.CreateLowerLimitAttr(0.0)
        pj.CreateUpperLimitAttr(0.04)
        pj.CreateLocalPos0Attr(Gf.Vec3f(0.0, float(local_pos0_y), 0.0))
        pj.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        rot = FLIP_X if flip else IDENT
        pj.CreateLocalRot0Attr(rot)
        pj.CreateLocalRot1Attr(rot)
        if j.HasAPI(PhysxSchema.PhysxMimicJointAPI, "rotY"):
            j.RemoveAPI(PhysxSchema.PhysxMimicJointAPI, "rotY")
        for attr in ("physxMimicJoint:rotY:gearing", "physxMimicJoint:rotY:offset",
                     "physxMimicJoint:rotY:referenceJoint"):
            if j.HasProperty(attr):
                j.RemoveProperty(attr)
        drive = UsdPhysics.DriveAPI.Apply(j, "linear")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(5000.0)
        drive.CreateDampingAttr(200.0)
        drive.CreateTargetPositionAttr(0.0)
        st = UsdPhysics.PrismaticJoint(j).GetPrim().GetAttribute("state:linear:physics:position")
        if st and st.IsValid():
            st.Set(0.0)

    _fix_finger("left_finger_joint", local_pos0_y=+0.04, flip=True)    # jaw on -Y; slide -Y to open
    _fix_finger("right_finger_joint", local_pos0_y=-0.04, flip=False)  # jaw on +Y; slide +Y to open
    emit("rebuilt both finger joints: joint0=CLOSED, 0.04=OPEN, symmetric (Factory convention)")

    stage.Export(DST)
    emit(f"exported -> {DST}")

    # Report the rigid-body + joint tree of the exported asset.
    out = Usd.Stage.Open(DST)
    bodies, jts = [], []
    for prim in out.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            bodies.append(prim.GetName())
        if prim.IsA(UsdPhysics.Joint):
            jt = prim.GetTypeName()
            b0 = prim.GetRelationship("physics:body0").GetTargets()
            b1 = prim.GetRelationship("physics:body1").GetTargets()
            jts.append(f"{prim.GetName():24s} {str(jt):22s} {[p.name for p in b0]} -> {[p.name for p in b1]}")
    emit(f"\nrigid bodies ({len(bodies)}): {bodies}")
    emit("joints:")
    for j in jts:
        emit(f"  {j}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        emit("FAILED\n" + traceback.format_exc())
    finally:
        with open("/tmp/build_iiwa_report.txt", "w") as f:
            f.write("\n".join(REPORT) + "\n")
        simulation_app.close()
