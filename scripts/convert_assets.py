"""Convert the cooling/pb part meshes (OBJ, centimeters) into Isaac Lab USD assets.

Source meshes live in ``source/insertion_policy/insertion_policy/assets/meshes`` and are
authored in **centimeters**, so we apply a uniform ``scale=0.01`` to bake them into
**meters** (Isaac Lab/USD convention).

Each part is converted with:
  * SDF mesh collision (resolution 256) - required because the receptacle parts are
    concave (insertion holes); convex hull/decomposition would fill the socket.
  * Rigid-body + mass + collision properties so the USD can be dropped straight into a
    RigidObjectCfg/ArticulationCfg. Mass is estimated from mesh volume assuming an
    effective density of 1.0 g/cm^3 (PLA print, near-solid small parts). Fixed parts
    ignore mass in-sim; held parts can be tuned in the env config.

Run (headless, from the rl/ project root)::

    ~/isaaclab/isaaclab.sh -p scripts/convert_assets.py
    # or with the conda env active:
    python scripts/convert_assets.py
"""

import argparse

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# CLI / app launch (AppLauncher must be created before importing pxr/sim code)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Convert thesis part meshes (cm OBJ) into USD assets.")
parser.add_argument(
    "--sdf-resolution",
    type=int,
    default=256,
    help="SDF grid resolution. Higher = more accurate contact, more memory. (default: 256)",
)
parser.add_argument(
    "--density",
    type=float,
    default=1.0,
    help="Effective density in g/cm^3 used to estimate part mass from mesh volume. (default: 1.0)",
)
parser.add_argument(
    "--parts",
    type=str,
    nargs="*",
    default=None,
    help="Subset of part names to convert (without extension). Default: all meshes found.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# force headless: this is an offline asset bake, no GUI needed
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Everything below runs inside the Isaac Sim runtime.
# ---------------------------------------------------------------------------
import os

import trimesh

from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg

# Source meshes are in cm -> scale to meters.
CM_TO_M = 0.01

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "source", "insertion_policy", "insertion_policy", "assets"))
MESH_DIR = os.path.join(ASSETS_DIR, "meshes")


def estimate_mass_kg(obj_path: str, density_g_per_cm3: float) -> float:
    """Estimate part mass (kg) from mesh volume (authored in cm) and an effective density."""
    mesh = trimesh.load(obj_path, process=False)
    volume_cm3 = float(mesh.volume)  # mesh units are cm -> volume is cm^3
    return volume_cm3 * density_g_per_cm3 * 1e-3  # grams -> kg


def convert_part(name: str) -> None:
    src = os.path.join(MESH_DIR, f"{name}.obj")
    if not os.path.isfile(src):
        raise FileNotFoundError(f"Mesh not found: {src}")

    mass_kg = estimate_mass_kg(src, args_cli.density)

    cfg = MeshConverterCfg(
        asset_path=src,
        usd_dir=ASSETS_DIR,
        usd_file_name=f"{name}.usd",
        force_usd_conversion=True,
        make_instanceable=False,  # keep per-prim so visual/texture domain randomization stays possible
        scale=(CM_TO_M, CM_TO_M, CM_TO_M),
        mass_props=schemas_cfg.MassPropertiesCfg(mass=mass_kg),
        rigid_props=schemas_cfg.RigidBodyPropertiesCfg(),
        collision_props=schemas_cfg.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.005,
            rest_offset=0.0,
        ),
        mesh_collision_props=schemas_cfg.SDFMeshPropertiesCfg(sdf_resolution=args_cli.sdf_resolution),
    )

    converter = MeshConverter(cfg)
    print(f"[OK] {name:16s} mass={mass_kg * 1000:7.1f}g  ->  {converter.usd_path}")


def main() -> None:
    if args_cli.parts:
        names = args_cli.parts
    else:
        names = sorted(f[:-4] for f in os.listdir(MESH_DIR) if f.endswith(".obj"))

    print("-" * 80)
    print(f"Converting {len(names)} part(s) | scale={CM_TO_M} (cm->m) | "
          f"SDF res={args_cli.sdf_resolution} | density={args_cli.density} g/cm^3")
    print(f"Output dir: {ASSETS_DIR}")
    print("-" * 80)

    for name in names:
        convert_part(name)

    print("-" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
    simulation_app.close()
