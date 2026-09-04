CONFIG_MAPPING = {}

# Upstream Franka example configs. Imported defensively: they pull franka_env /
# spacemouse deps that are not installed in our KUKA serl env, so a missing dep
# must NOT break our own experiments below.
try:
    from experiments.ram_insertion.config import TrainConfig as RAMInsertionTrainConfig
    from experiments.usb_pickup_insertion.config import TrainConfig as USBPickupInsertionTrainConfig
    from experiments.object_handover.config import TrainConfig as ObjectHandoverTrainConfig
    from experiments.egg_flip.config import TrainConfig as EggFlipTrainConfig

    CONFIG_MAPPING.update({
        "ram_insertion": RAMInsertionTrainConfig,
        "usb_pickup_insertion": USBPickupInsertionTrainConfig,
        "object_handover": ObjectHandoverTrainConfig,
        "egg_flip": EggFlipTrainConfig,
    })
except Exception as _exc:  # noqa: BLE001 - upstream franka deps absent in our env
    print(f"[mappings] upstream Franka configs unavailable ({_exc!r}); KUKA configs still load.")

# --- KUKA per-insert HIL-SERL configs (one model per insert, both assemblies) ---
from experiments.iiwa_plumbers.config import (
    TrainConfigInsert0 as _Plumbers0,
    TrainConfigInsert1 as _Plumbers1,
    TrainConfigInsert2 as _Plumbers2,
    TrainConfigInsert3 as _Plumbers3,
)
from experiments.iiwa_cooling.config import (
    TrainConfigInsert0 as _Cooling0,
    TrainConfigInsert1 as _Cooling1,
    TrainConfigInsert2 as _Cooling2,
    TrainConfigInsert3 as _Cooling3,
    TrainConfigInsert4 as _Cooling4,
    TrainConfigInsert5 as _Cooling5,
)

CONFIG_MAPPING.update({
    "iiwa_plumbers_insert0": _Plumbers0,
    "iiwa_plumbers_insert1": _Plumbers1,
    "iiwa_plumbers_insert2": _Plumbers2,
    "iiwa_plumbers_insert3": _Plumbers3,
    "iiwa_cooling_insert0": _Cooling0,
    "iiwa_cooling_insert1": _Cooling1,
    "iiwa_cooling_insert2": _Cooling2,
    "iiwa_cooling_insert3": _Cooling3,
    "iiwa_cooling_insert4": _Cooling4,
    "iiwa_cooling_insert5": _Cooling5,
})
