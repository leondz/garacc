# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import tempfile

import pytest
import importlib

from garak import _plugins, _config, evaluators

import garak.harnesses.base

HARNESSES = [
    classname for (classname, active) in _plugins.enumerate_plugins("harnesses")
]


@pytest.mark.parametrize("classname", HARNESSES)
def test_harness_structure(classname):
    m = importlib.import_module("garak." + ".".join(classname.split(".")[:-1]))
    c = getattr(m, classname.split(".")[-1])

    # any parameter that has a default must be supported
    unsupported_defaults = []
    if c._supported_params is not None:
        if hasattr(c, "DEFAULT_PARAMS"):
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

    # Setting up global environment in order to run our harness
    _config.load_base_config()
    _config.cas.intent_spec = "T999"
    _config.cas.serve_detectorless_intents = True

    # Setup report file
    temp_report_file = tempfile.NamedTemporaryFile(
        mode="w+", delete=False, encoding="utf-8"
    )
    _config.transient.reportfile = temp_report_file
    _config.transient.report_filename = temp_report_file.name

    # Create harnes
    earlystop_h = EarlyStopHarness()

    # Harness inputs
    g = _plugins.load_plugin("generators.test.Blank")
    detector_names = ["detectors.always.Fail"]
    probe_names = [
        "probes.grandma.GrandmaIntent"
    ]
    e = evaluators.ThresholdEvaluator()

    earlystop_h.run(g, probe_names, detector_names, e)

    # Verify reportfile was created and has expected content
    temp_report_file.flush()
    temp_report_file.seek(0)
    report_lines = temp_report_file.readlines()

    assert len(report_lines) >= 1, "Reportfile should contain at least one entry"

    import json

    parsed = [json.loads(line) for line in report_lines]

    # --- eval entry: probe identifier, detector, scores/pass status ---
    eval_entries = [entry for entry in parsed if entry.get("entry_type") == "eval"]
    assert len(eval_entries) == 1, "Reportfile should contain exactly one eval entry"
    eval_entry = eval_entries[0]
    assert eval_entry["probe"] == "garak.harnesses.earlystop.EarlyStopHarness"
    assert eval_entry["detector"] == "EarlyStop"
    assert eval_entry["passed"] >= 0
    assert eval_entry["fails"] >= 0
    assert eval_entry["total_evaluated"] == eval_entry["passed"] + eval_entry["fails"]
    assert eval_entry["total_processed"] == eval_entry["total_evaluated"] + eval_entry["nones"]

    # --- eval_intent entry: intent metadata, score, detectors ---
    intent_entries = [entry for entry in parsed if entry.get("entry_type") == "eval_intent"]
    assert len(intent_entries) >= 1, "Reportfile should contain at least one eval_intent entry"
    intent_entry = intent_entries[0]
    assert intent_entry["probe"] == "garak.harnesses.earlystop.EarlyStopHarness"
    assert intent_entry["intent"] == "T999"
    assert intent_entry["score"] is None or 0.0 <= intent_entry["score"] <= 1.0
    assert "aggregation" in intent_entry
    assert intent_entry["n_detectors"] >= 1
    assert intent_entry["n_evaluations"] >= 1
    assert "EarlyStop" in intent_entry["detectors_used"]

    temp_report_file.close()
