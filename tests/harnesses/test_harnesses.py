# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import tempfile
from types import SimpleNamespace

import pytest
import importlib

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

    earlystop_h = EarlyStopHarness()
    assert isinstance(earlystop_h, EarlyStopHarness)

    # Setting up global environment in order to run our harness
    _config.load_base_config()
    temp_report_file = tempfile.NamedTemporaryFile(
        mode="w+", delete=False, encoding="utf-8"
    )
    _config.transient.reportfile = temp_report_file
    _config.transient.report_filename = temp_report_file.name

    # Harness inputs
    g = _plugins.load_plugin("generators.test.Blank")
    intents = ["T999", "T999test"]
    attack_methods = [
        _plugins.load_plugin("probes.grandma.GrandmaIntent", config_root=garak._config)
    ]
    detector = garak._plugins.load_plugin("detectors.always.Fail")
    e = evaluators.ThresholdEvaluator()

    # Start run
    earlystop_h.run(g, intents, attack_methods, [detector], e)

    # Verify reportfile was created and has expected content
    temp_report_file.flush()
    temp_report_file.seek(0)
    report_lines = temp_report_file.readlines()

    assert len(report_lines) >= 1, "Reportfile should contain at least one attempt"


    temp_report_file.close()
