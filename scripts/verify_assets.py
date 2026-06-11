"""Sanity-check the converted USD part assets: world-space extent (meters), collision API, mass."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify converted USD part assets.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

ASSETS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "source", "insertion_policy", "insertion_policy", "assets")
)


REPORT_LINES: list[str] = []


def emit(line: str) -> None:
    print(line)
    REPORT_LINES.append(line)


def check(name: str) -> None:
    path = os.path.join(ASSETS_DIR, f"{name}.usd")
    stage = Usd.Stage.Open(path)

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    has_sdf = has_rigid = has_mass = False
    extent_m = None
    mass_val = None

    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            rng = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
            size = rng.GetSize()
            extent_m = (round(size[0], 4), round(size[1], 4), round(size[2], 4))
            if PhysxSchema.PhysxSDFMeshCollisionAPI(prim):
                has_sdf = bool(prim.HasAPI(PhysxSchema.PhysxSDFMeshCollisionAPI))
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            has_rigid = True
        if prim.HasAPI(UsdPhysics.MassAPI):
            has_mass = True
            m = UsdPhysics.MassAPI(prim).GetMassAttr().Get()
            if m:
                mass_val = round(float(m), 5)

    emit(
        f"{name:16s} extent(m)={extent_m}  sdf={has_sdf}  rigid={has_rigid}  "
        f"mass={mass_val}kg"
    )


def main() -> None:
    names = sorted(f[:-4] for f in os.listdir(ASSETS_DIR) if f.endswith(".usd"))
    emit("-" * 80)
    for n in names:
        check(n)
    emit("-" * 80)
    with open("/tmp/asset_report.txt", "w") as f:
        f.write("\n".join(REPORT_LINES) + "\n")


if __name__ == "__main__":
    main()
    simulation_app.close()
