# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import langcodes
import pytest
import re

from garak import _config, _plugins
from garak.attempt import Turn, Conversation, Message, Attempt
import garak.probes
from garak.services.intentservice import validate_intent_specifier

PROBES = [classname for (classname, active) in _plugins.enumerate_plugins("probes")]

DETECTORS = [
    classname
    for (classname, active) in _plugins.enumerate_plugins(
        "detectors", skip_base_classes=False
    )
]
DETECTOR_BARE_NAMES = [".".join(d.split(".")[1:]) for d in DETECTORS]


with open(
    _config.transient.package_dir / "data" / "tags.misp.tsv",
    "r",
    encoding="utf-8",
) as misp_data:
    MISP_TAGS = [line.split("\t")[0] for line in misp_data.read().split("\n")]


def _probe_intent_may_be_none(probe_class) -> bool:
    return (
        probe_class.__module__ == "garak.probes.base"
        or garak.probes.IntentProbe in probe_class.mro()
    )


@pytest.mark.parametrize("classname", PROBES)
def test_detector_specified(classname):  # every probe should give detector(s)
    plugin_name_parts = classname.split(".")
    module_name = "garak." + ".".join(plugin_name_parts[:-1])
    class_name = plugin_name_parts[-1]
    mod = importlib.import_module(module_name)
    probe_class = getattr(mod, class_name)
    if (
        garak.probes.IntentProbe not in probe_class.mro()
    ):  # intent probes detector spec not relevant
        assert (
            isinstance(probe_class.primary_detector, str)
            or len(probe_class.extended_detectors) > 0
        ), "One primary detector (str), or a non-empty list of extended detector, must be given"


@pytest.mark.parametrize("classname", PROBES)
def test_probe_detector_exists(classname):
    plugin_name_parts = classname.split(".")
    module_name = "garak." + ".".join(plugin_name_parts[:-1])
    class_name = plugin_name_parts[-1]
    mod = importlib.import_module(module_name)
    probe_class = getattr(mod, class_name)
    probe_detectors = list(probe_class.extended_detectors)
    if probe_class.primary_detector is not None:
        probe_detectors.append(probe_class.primary_detector)
    assert set(probe_detectors).issubset(DETECTOR_BARE_NAMES)


@pytest.mark.parametrize("classname", PROBES)
def test_probe_structure(classname):

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


@pytest.mark.parametrize("classname", PROBES)
def test_probe_intent(classname, loaded_intent_service):
    plugin_name_parts = classname.split(".")
    module_name = "garak." + ".".join(plugin_name_parts[:-1])
    class_name = plugin_name_parts[-1]
    mod = importlib.import_module(module_name)
    probe_class = getattr(mod, class_name)

    assert hasattr(probe_class, "intent"), "probes must declare an intent attribute"

    if _probe_intent_may_be_none(probe_class):
        assert (
            probe_class.intent is None
        ), "base probes and IntentProbe descendants should set intent to None"
    else:
        assert isinstance(
            probe_class.intent, str
        ), "concrete probes must set intent to a typology code"
        assert len(probe_class.intent) > 0, "intent must not be empty"
        assert validate_intent_specifier(
            probe_class.intent
        ), "intent must match a valid intent typology entry"


@pytest.mark.parametrize("classname", PROBES)
def test_probe_metadata(classname, loaded_intent_service):
    try:
        p = _plugins.load_plugin(classname)
    except ModuleNotFoundError:
        pytest.skip("required deps not present")
    if not isinstance(p, garak.probes.IntentProbe):  # intent probes have flexible goal
        assert isinstance(p.goal, str), "probe goals should be a text string"
        assert len(p.goal) > 0, "probes must state their general goal"
    assert p.lang is not None and (
        p.lang == "*" or langcodes.tag_is_valid(p.lang)
    ), "lang must be either * or a BCP47 code"
    assert isinstance(
        p.doc_uri, str
    ), "probes should give a doc uri describing/citing the attack"
    if len(p.doc_uri) > 1:
        assert p.doc_uri.lower().startswith(
            "http"
        ), "doc uris should be fully-specified absolute HTTP addresses"
    assert isinstance(p.modality, dict), "probes need to describe available modalities"
    assert "in" in p.modality, "probe modalities need an in descriptor"
    assert isinstance(p.modality["in"], set), "modality descriptors must be sets"
    assert p.tier is not None, "probe tier must be specified"
    assert isinstance(p.tier, garak.probes.Tier), "probe tier must be one of type Tier'"
    if p.active:
        assert (
            p.extra_dependency_names == []
        ), "active must be False for Probes requiring external modules, so that they're not run by default"


@pytest.mark.parametrize("plugin_name", PROBES)
def test_check_docstring(plugin_name):
    plugin_name_parts = plugin_name.split(".")
    module_name = "garak." + ".".join(plugin_name_parts[:-1])
    class_name = plugin_name_parts[-1]
    mod = importlib.import_module(module_name)
    doc = getattr(getattr(mod, class_name), "__doc__")
    doc_paras = re.split(r"\s*\n\s*\n\s*", doc)
    assert (
        len(doc_paras) >= 2
    )  # probe class doc should have a summary, two newlines, then a paragraph giving more depth, then optionally more words
    assert (
        len(doc_paras[0]) > 0
    )  # the first paragraph of the probe docstring should not be empty


@pytest.mark.parametrize("classname", PROBES)
def test_tag_format(classname):
    plugin_name_parts = classname.split(".")
    module_name = "garak." + ".".join(plugin_name_parts[:-1])
    class_name = plugin_name_parts[-1]
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    assert (
        cls.tags != [] or cls.active == False
    )  # all probes should have at least one tag
    for tag in cls.tags:  # should be MISP format
        assert type(tag) == str
        for part in tag.split(":"):
            assert re.match(r"^[A-Za-z0-9_\-\&]+$", part)
        if tag.split(":")[0] != "payload":
            assert tag in MISP_TAGS


def test_probe_prune_alignment():
    p = _plugins.load_plugin("probes.glitch.Glitch")
    assert len(p.prompts) == _config.run.soft_probe_prompt_cap
    assert len(p.triggers) == _config.run.soft_probe_prompt_cap
    assert p.triggers[0] in p.prompts[0]
    assert p.triggers[-1] in p.prompts[-1]


PROMPT_EXAMPLES = [
    "test example",
    Message(text="test example"),
    Conversation([Turn(role="user", content=Message(text="test example"))]),
    Conversation(
        [
            Turn(role="system", content=Message(text="test system")),
            Turn(role="user", content=Message(text="test example")),
        ]
    ),
]


@pytest.mark.parametrize("prompt", PROMPT_EXAMPLES)
def test_mint_attempt(prompt):
    import garak.probes.base

    probe = garak.probes.base.Probe()
    attempt = probe._mint_attempt(prompt)
    assert isinstance(attempt, Attempt)
    for turn in attempt.prompt.turns:
        assert isinstance(turn, Turn)
    assert attempt.prompt.last_message().text == "test example"


@pytest.mark.parametrize("prompt", PROMPT_EXAMPLES)
def test_mint_attempt_with_run_system_prompt(prompt):
    import garak.probes.base

    expected_system_prompt = "test system prompt"
    probe = garak.probes.base.Probe()
    probe.system_prompt = expected_system_prompt

    if isinstance(prompt, Conversation):
        try:
            expected_system_prompt = prompt.last_message("system").text
        except ValueError as e:
            pass

    attempt = probe._mint_attempt(prompt)
    assert isinstance(attempt, Attempt)
    for turn in attempt.prompt.turns:
        assert isinstance(turn, Turn)
    assert attempt.prompt.last_message().text == "test example"
    assert attempt.prompt.last_message("system").text == expected_system_prompt
    system_message = [turn for turn in attempt.prompt.turns if turn.role == "system"]
    assert len(system_message) == 1


def test_mint_attempt_base_probe_intent_is_none():
    import garak.probes.base

    probe = garak.probes.base.Probe()
    attempt = probe._mint_attempt("hello")
    assert (
        attempt.intent is None
    ), "base Probe has no intent, so attempt.intent must be None"


def test_mint_attempt_propagates_probe_intent():
    import garak.probes.base

    probe = garak.probes.base.Probe()
    probe.intent = "T999test"
    attempt = probe._mint_attempt("hello")
    assert attempt.intent == "T999test", "attempt.intent must match the probe's intent"


def test_mint_attempt_intent_survives_multiple_attempts():
    import garak.probes.base

    probe = garak.probes.base.Probe()
    probe.intent = "S005"
    attempts = [probe._mint_attempt(f"prompt {i}", seq=i) for i in range(5)]
    for a in attempts:
        assert a.intent == "S005", "every minted attempt must carry the probe's intent"


def test_concrete_probe_propagates_intent(loaded_intent_service):
    p = _plugins.load_plugin("probes.test.Test")
    assert p.intent is not None, "probes.test.Test should have a non-None intent"
    attempt = p._mint_attempt("hello")
    assert (
        attempt.intent == p.intent
    ), "attempt.intent must match the concrete probe's intent"


def test_attempt_intent_in_serialised_dict(loaded_intent_service):
    p = _plugins.load_plugin("probes.test.Test")
    attempt = p._mint_attempt("hello")
    d = attempt.as_dict()
    assert "intent" in d, "serialised attempt dict must include 'intent'"
    assert d["intent"] == p.intent, "serialised intent must match probe intent"


def test_payload_intent_overrides_probe_intent():
    import garak.probes.base

    probe = garak.probes.base.Probe()
    probe.intent = "T999test"
    probe._payload_intent = "S005hate"
    attempt = probe._mint_attempt("hello")
    assert (
        attempt.intent == "S005hate"
    ), "payload intent must override probe intent when set"


def test_payload_intent_none_falls_back_to_probe_intent():
    import garak.probes.base

    probe = garak.probes.base.Probe()
    probe.intent = "T999test"
    probe._payload_intent = None
    attempt = probe._mint_attempt("hello")
    assert (
        attempt.intent == "T999test"
    ), "when payload intent is None, probe intent should be used"


def test_no_payload_intent_attr_falls_back_to_probe_intent():
    import garak.probes.base

    probe = garak.probes.base.Probe()
    probe.intent = "T999test"
    attempt = probe._mint_attempt("hello")
    assert (
        attempt.intent == "T999test"
    ), "when no _payload_intent attr exists, probe intent should be used"


def test_encoding_probe_per_prompt_intents(loaded_intent_service):
    import garak.probes.encoding

    p = _plugins.load_plugin("probes.encoding.InjectROT13")
    assert hasattr(
        p, "_prompt_intents"
    ), "encoding probes must track per-prompt intents"
    assert len(p._prompt_intents) == len(
        p.prompts
    ), "there must be one intent entry per prompt"
    for i, pi in enumerate(p._prompt_intents):
        if pi is not None:
            assert isinstance(
                pi, str
            ), f"prompt intent at index {i} must be a string or None"


def test_encoding_probe_attempt_carries_payload_intent(loaded_intent_service):
    import garak.probes.encoding

    p = _plugins.load_plugin("probes.encoding.InjectROT13")
    intents_with_values = [
        (seq, pi) for seq, pi in enumerate(p._prompt_intents) if pi is not None
    ]
    assert (
        len(intents_with_values) > 0
    ), "at least some prompts should have payload-derived intents"
    seq, expected_intent = intents_with_values[0]
    attempt = p._mint_attempt(p.prompts[seq], seq=seq)
    assert (
        attempt.intent == expected_intent
    ), "attempt intent must reflect the payload-specific intent set in _attempt_prestore_hook"


DNA_CATEGORIES = [
    "discrimination_exclusion_toxicity_hateful_offensive",
    "human_chatbox",
    "information_hazard",
    "malicious_uses",
    "misinformation_harms",
]


@pytest.mark.parametrize("category", DNA_CATEGORIES)
def test_donotanswer_jsonl_exists(category):
    import json
    from garak.data import path as data_path

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
    import json
    from garak.data import path as data_path
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
def test_donotanswer_jsonl_matches_txt(category):
    from garak.data import path as data_path
    import json

    txt_path = data_path / "donotanswer" / f"{category}.txt"
    jsonl_path = data_path / "donotanswer" / f"{category}.jsonl"
    txt_prompts = txt_path.read_text(encoding="utf-8").strip().split("\n")
    with open(jsonl_path, encoding="utf-8") as f:
        jsonl_prompts = [json.loads(line)["prompt"] for line in f if line.strip()]
    assert (
        txt_prompts == jsonl_prompts
    ), f"prompts in {category}.jsonl must match {category}.txt exactly"


@pytest.mark.parametrize("category", DNA_CATEGORIES)
def test_donotanswer_probe_loads_prompts(category, loaded_intent_service):
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
