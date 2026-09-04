"""Keep Conda third-party packages behind the system Python standard library.

The SERL robot runtime intentionally uses ROS 2's system Python while exposing a
Conda environment's site-packages through PYTHONPATH. That environment contains
the obsolete PyPI ``typing`` backport, which must not shadow Python 3.10's stdlib
module. Python imports this file automatically during startup.
"""

import sys


_serl_sites = [
    path
    for path in sys.path
    if "/envs/serl/" in path and path.endswith("/site-packages")
]
if _serl_sites:
    _stdlib = f"{sys.base_prefix}/lib/python{sys.version_info.major}.{sys.version_info.minor}"
    for _path in _serl_sites:
        sys.path.remove(_path)
    _insert_at = max(
        (
            index + 1
            for index, path in enumerate(sys.path)
            if path == _stdlib or path.startswith(_stdlib + "/")
        ),
        default=len(sys.path),
    )
    for _path in reversed(_serl_sites):
        sys.path.insert(_insert_at, _path)
