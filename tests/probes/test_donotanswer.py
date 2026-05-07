# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for donotanswer probe JSONL data files and probe loading.

Each category has a .jsonl file in garak/data/donotanswer/ containing prompts
and per-prompt intent annotations.
"""

import json

import pytest

from garak import _plugins
from garak.data import path as data_path


DNA_CATEGORIES = [
    "discrimination_exclusion_toxicity_hateful_offensive",
    "human_chatbox",
    "information_hazard",
    "malicious_uses",
    "misinformation_harms",
]


@pytest.mark.parametrize("category", DNA_CATEGORIES)
def test_donotanswer_jsonl_exists_and_well_formed(category):
    """Each category .jsonl must exist and every line must have a prompt and intents."""
    jsonl_path = data_path / "donotanswer" / f"{category}.jsonl"
    assert jsonl_path.exists(), f"{jsonl_path} must exist"
    with open(jsonl_path, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    assert len(entries) > 0, f"{category}.jsonl must contain at least one entry"
    for i, entry in enumerate(entries):
        assert "prompt" in entry, f"line {i} in {category}.jsonl missing 'prompt'"
        assert "intents" in entry, f"line {i} in {category}.jsonl missing 'intents'"
        assert (
            len(entry["intents"]) >= 1
        ), f"line {i} in {category}.jsonl must have at least one intent"
        assert isinstance(
            entry["intents"][0], str
        ), f"line {i} in {category}.jsonl first intent must be a string"


@pytest.mark.parametrize("category", DNA_CATEGORIES)
def test_donotanswer_jsonl_intents_valid(category, loaded_intent_service):
    """Every intent specifier in the JSONL must pass validation."""
    from garak.services.intentservice import validate_intent_specifier

    jsonl_path = data_path / "donotanswer" / f"{category}.jsonl"
    with open(jsonl_path, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    for i, entry in enumerate(entries):
        for intent_code in entry["intents"]:
            assert validate_intent_specifier(
                intent_code
            ), f"line {i} in {category}.jsonl has invalid intent '{intent_code}'"


@pytest.mark.parametrize("category", DNA_CATEGORIES)
def test_donotanswer_probe_loads_prompts(category, loaded_intent_service):
    """Loading the probe class must populate prompts and matching _prompt_intents."""
    classname = category.title().replace("_", "")
    p = _plugins.load_plugin(f"probes.donotanswer.{classname}")
    assert len(p.prompts) > 0, f"donotanswer.{classname} must load prompts from JSONL"
    assert hasattr(
        p, "_prompt_intents"
    ), f"donotanswer.{classname} must populate _prompt_intents"
    assert len(p._prompt_intents) == len(
        p.prompts
    ), f"_prompt_intents length must match prompts length"


@pytest.mark.parametrize("category", DNA_CATEGORIES)
def test_donotanswer_attempt_carries_per_prompt_intent(category, loaded_intent_service):
    """Minted attempts must carry the per-prompt intent from the JSONL data."""
    classname = category.title().replace("_", "")
    p = _plugins.load_plugin(f"probes.donotanswer.{classname}")
    attempt_0 = p._mint_attempt(p.prompts[0], seq=0)
    assert (
        attempt_0.intent == p._prompt_intents[0]
    ), "first attempt must carry the first prompt's per-prompt intent"
    if len(p.prompts) > 1:
        attempt_last = p._mint_attempt(p.prompts[-1], seq=len(p.prompts) - 1)
        assert (
            attempt_last.intent == p._prompt_intents[-1]
        ), "last attempt must carry the last prompt's per-prompt intent"
