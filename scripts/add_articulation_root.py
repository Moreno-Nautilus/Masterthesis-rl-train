"""Bake UsdPhysics.ArticulationRootAPI into the converted part USDs.

Forge/Factory wrap the held/fixed assets as single-link Articulations, which requires the USD
to carry an ArticulationRootAPI (the MeshConverter only adds RigidBodyAPI). This applies the
articulation root (+ PhysxArticulationAPI) to each part's default prim, in place.

Idempotent: skips parts that already have the API. Writes a report to /tmp/artroot_report.txt.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--parts", type=str, nargs="*", default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

from pxr import PhysxSchema, Usd, UsdPhysics

ASSETS = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "source", "insertion_policy", "insertion_policy", "assets")
)

REPORT = []


def fix(name: str) -> None:
    path = os.path.join(ASSETS, f"{name}.usd")
    stage = Usd.Stage.Open(path)
    default_prim = stage.GetDefaultPrim()

    # locate the rigid-body prim (for the report / sanity)
    rb_prim = None
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rb_prim = prim
            break

    already = bool(UsdPhysics.ArticulationRootAPI(default_prim))
    if not already:
        UsdPhysics.ArticulationRootAPI.Apply(default_prim)
        PhysxSchema.PhysxArticulationAPI.Apply(default_prim)
        stage.GetRootLayer().Save()

    REPORT.append(
        f"{name:16s} default_prim={default_prim.GetPath()} rigid_body={rb_prim.GetPath() if rb_prim else None} "
        f"art_root={'already' if already else 'ADDED'}"
    )


def main() -> None:
    names = args_cli.parts or sorted(f[:-4] for f in os.listdir(ASSETS) if f.endswith(".usd"))
    for n in names:
        fix(n)
    with open("/tmp/artroot_report.txt", "w") as f:
        f.write("\n".join(REPORT) + "\n")


if __name__ == "__main__":
    main()
    simulation_app.close()
