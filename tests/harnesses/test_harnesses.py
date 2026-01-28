# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import tempfile
from types import SimpleNamespace

import pytest
import importlib

import data
from garak import _plugins, _config, evaluators

import garak.harnesses.base

HARNESSES = [
    classname for (classname, active) in _plugins.enumerate_plugins("harnesses")
]


@pytest.mark.parametrize("classname", HARNESSES)
def test_buff_structure(classname):
    m = importlib.import_module("garak." + ".".join(classname.split(".")[:-1]))
    c = getattr(m, classname.split(".")[-1])

    # any parameter that has a default must be supported
    unsupported_defaults = []
    if c._supported_params is not None:
        if hasattr(g, "DEFAULT_PARAMS"):
            for k, _ in c.DEFAULT_PARAMS.items():
                if k not in c._supported_params:
                    unsupported_defaults.append(k)
    assert unsupported_defaults == []


def test_harness_modality_match():
    t = {"text"}
    ti = {"text", "image"}
    tv = {"text", "vision"}
    tvi = {"text", "vision", "image"}

    # probe, generator
    assert garak.harnesses.base._modality_match(t, t, True) is True
    assert garak.harnesses.base._modality_match(ti, ti, True) is True
    assert garak.harnesses.base._modality_match(t, tv, True) is False
    assert garak.harnesses.base._modality_match(ti, t, True) is False

    # when strict is false, generator must support all probe modalities, but can also support more
    assert garak.harnesses.base._modality_match(t, t, False) is True
    assert garak.harnesses.base._modality_match(ti, t, False) is False
    assert garak.harnesses.base._modality_match(t, tvi, False) is True
    assert garak.harnesses.base._modality_match(ti, tvi, False) is True
    assert garak.harnesses.base._modality_match(t, ti, False) is True


def test_early_stop_harness():
    from garak.harnesses.earlystop import EarlyStopHarness
    import garak.data as garak_data

    # Setting up global environment in order to run our harness
    _config.load_base_config()

    # Configure Policy via _config
    # policy_data_path loads the trait typology
    # policy_points sets specific traits to False (these become the intents to test)
    _config.cas.policy_data_path = str(garak_data.path / "cas" / "trait_typology.json")
    _config.cas.policy_points = {"T999": False,
                                 "T999test": False}

    # Configure probes
    _config.plugins.probes = {
        "tap": {
            "TAPIntent": {
                # Setting up a local model, you have to set the environment variable OPENAICOMPATIBLE_API_KEY
                "attack_model_type": "openai.OpenAICompatible",
                "attack_model_name": "TheBloke/vicuna-13B-v1.5-GGUF",
                "attack_model_config": {
                    "uri": "http://localhost:1234/v1/",
                    "max_tokens": 500,
                },
                "evaluator_model_type": "openai.OpenAICompatible",
                "evaluator_model_name": "TheBloke/vicuna-13B-v1.5-GGUF",
                "evaluator_model_config": {
                    "uri": "http://localhost:1234/v1/",
                    "max_tokens": 10,
                    "temperature": 0.0,
                },
                # Speed up generation, no need to run a full evaluation
                "attack_max_attempts": 2,
                "width": 2,
                "depth": 1,
                "branching_factor": 2,
                "pruning": False,  # We are not going to produce good results
            }}
    }

    # Setup report file
    temp_report_file = tempfile.NamedTemporaryFile(
        mode="w+", delete=False, encoding="utf-8"
    )
    _config.transient.reportfile = temp_report_file
    _config.transient.report_filename = temp_report_file.name

    # Create harness (Policy loaded from config in __init__)
    earlystop_h = EarlyStopHarness()
    assert isinstance(earlystop_h, EarlyStopHarness)

    # Verify policy loaded with correct points
    assert earlystop_h.policy is not None
    assert earlystop_h.policy.points.get("T999") is False
    assert earlystop_h.policy.points.get("T999test") is False

    # Harness inputs - PxD-style with names
    g = _plugins.load_plugin("generators.test.Blank")
    probe_names = ["probes.grandma.GrandmaIntent", "probes.tap.TAPIntent"]
    detector_names = ["detectors.always.Fail"]
    e = evaluators.ThresholdEvaluator()

    earlystop_h.run(g, probe_names, detector_names, e)

    # Verify reportfile was created and has expected content
    temp_report_file.flush()
    temp_report_file.seek(0)
    report_lines = temp_report_file.readlines()

    assert len(report_lines) >= 1, "Reportfile should contain at least one attempt"

    temp_report_file.close()
