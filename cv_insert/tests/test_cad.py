"""Tests for the OBJ loader + feature priors. CAD tests skip if meshes are absent."""

import os

import numpy as np
import pytest

from cv_insert import cad


def test_priors_present_and_metric():
    for name in ("pb_top", "pb_screw", "pb_pipe"):
        p = cad.get_prior(name)
        assert 0.0 < p.feature_diameter_m < 0.1  # a few cm, in metres
        assert 0.0 <= p.vision_weight <= 1.0
        assert len(p.part_bbox_m) == 3


def test_pb_top_is_lowest_vision_weight():
    # pb_top is the HARD one -> must trust vision least.
    w = {n: cad.get_prior(n).vision_weight for n in ("pb_top", "pb_screw", "pb_pipe")}
    assert w["pb_top"] == min(w.values())


def test_unknown_feature_raises():
    with pytest.raises(KeyError):
        cad.get_prior("nope")


@pytest.mark.skipif(
    not os.path.isdir(cad.CAD_DIR), reason="vision CAD dir not present"
)
def test_load_obj_and_bbox_matches_prior():
    for name in ("pb_top", "pb_screw", "pb_pipe"):
        path = cad.part_cad_path(name)
        if not os.path.exists(path):
            pytest.skip(f"{path} missing")
        v = cad.load_obj_vertices(path)
        assert v.ndim == 2 and v.shape[1] == 3 and len(v) > 10
        bbox_cm = v.max(0) - v.min(0)
        prior_bbox_cm = np.array(cad.get_prior(name).part_bbox_m) * 100.0
        # within 0.5 cm of what we hard-coded
        assert np.allclose(np.sort(bbox_cm), np.sort(prior_bbox_cm), atol=0.5)
