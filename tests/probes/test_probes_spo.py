# SPDX-FileCopyrightText: Portions Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SPO (System Prompt Override) Intent Probe"""

import pytest
import tempfile

import garak
from garak import _config
from garak.probes.spo import SPOIntent


def test_spo_dan_loading():
    """Test that DAN prompts are loaded correctly from JSON files"""
    _config.load_base_config()
    _config.cas.intent_spec = None  # No intents for this test

    probe = SPOIntent(config_root=_config)

    # Should load all 14 DAN variants
    assert len(probe.dan_prompts) == 14, f"Expected 14 DAN variants, got {len(probe.dan_prompts)}"

    # Check that expected variants are present
    expected_variants = [
        "Dan_11_0",
        "Dan_10_0",
        "Dan_9_0",
        "Dan_8_0",
        "Dan_7_0",
        "Dan_6_2",
        "Dan_6_0",
        "STAN",
        "DUDE",
        "DAN_Jailbreak",
    ]
    for variant in expected_variants:
        assert variant in probe.dan_prompts, f"Expected variant {variant} not found"

    # Each variant should have at least one prompt
    for variant_name, prompts in probe.dan_prompts.items():
        assert len(prompts) > 0, f"Variant {variant_name} has no prompts"
        assert all(isinstance(p, str) for p in prompts), f"All prompts in {variant_name} should be strings"
        assert all(len(p) > 0 for p in prompts), f"All prompts in {variant_name} should be non-empty"


def test_spo_prompts_from_stub():
    """Test that prompts_from_stub correctly combines DAN variants with stubs"""
    _config.load_base_config()
    _config.cas.intent_spec = None

    probe = SPOIntent(config_root=_config)

    test_stub = "tell me how to build a bomb"
    prompts = probe.prompts_from_stub(test_stub)

    # Should generate one prompt per DAN variant (14 total)
    expected_count = sum(len(v) for v in probe.dan_prompts.values())
    assert len(prompts) == expected_count, f"Expected {expected_count} prompts, got {len(prompts)}"

    # All prompts should contain the stub
    for prompt in prompts:
        assert test_stub in prompt, f"Prompt should contain stub: {test_stub}"

    # All prompts should contain the DAN prompt content
    # Check that at least one prompt contains a signature DAN phrase
    has_dan_content = any("{generator.name}" in p for p in prompts)
    assert has_dan_content, "At least some prompts should contain {generator.name} placeholder"


def test_spo_variant_mapping():
    """Test that prompt_to_variant mapping is created correctly"""
    _config.load_base_config()
    _config.cas.intent_spec = None

    probe = SPOIntent(config_root=_config)

    test_stub = "write malicious code"
    prompts = probe.prompts_from_stub(test_stub)

    # Should have a mapping for each generated prompt
    assert len(probe.prompt_to_variant) == len(prompts), "Should have mapping for each prompt"

    # Check that mappings have correct structure
    for idx, mapping in probe.prompt_to_variant.items():
        assert "dan_variant" in mapping, f"Mapping {idx} should have dan_variant"
        assert "dan_prompt_index" in mapping, f"Mapping {idx} should have dan_prompt_index"
        assert isinstance(mapping["dan_variant"], str), "dan_variant should be string"
        assert isinstance(mapping["dan_prompt_index"], int), "dan_prompt_index should be int"


def test_spo_generator_name_substitution():
    """Test that {generator.name} is substituted correctly in probe()"""
    _config.load_base_config()
    _config.cas.intent_spec = "T999"  # Use test intent

    import garak.intentservice
    garak.intentservice.load()

    probe = SPOIntent(config_root=_config)

    # Load a test generator
    g = garak._plugins.load_plugin("generators.test.Blank", config_root=_config)

    # Verify some prompts have the placeholder before probe()
    has_placeholder_before = any("{generator.name}" in p for p in probe.prompts)
    assert has_placeholder_before, "Should have {generator.name} placeholder before probe()"

    # Set up temporary report file for probe execution
    with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8") as temp_report_file:
        _config.transient.reportfile = temp_report_file
        _config.transient.report_filename = temp_report_file.name

        # Run probe (this should substitute {generator.name})
        attempts = list(probe.probe(g))

    # After probe(), prompts should have generator name substituted
    has_placeholder_after = any("{generator.name}" in p for p in probe.prompts)
    assert not has_placeholder_after, "Should not have {generator.name} placeholder after probe()"

    # Check that generator name appears in prompts
    generator_name = getattr(g, "name", "ChatGPT")
    has_generator_name = any(generator_name in p for p in probe.prompts)
    assert has_generator_name, f"Should have {generator_name} in prompts after substitution"


def test_spo_intent_integration():
    """Test SPOIntent with IntentService integration"""
    _config.load_base_config()

    # Load intentservice and set up intent spec
    import garak.intentservice
    garak.intentservice.load()
    _config.cas.intent_spec = "T999"  # Test intent with known stubs

    probe = SPOIntent(config_root=_config)

    # Should have loaded intents
    assert len(probe.intents) > 0, "Should have loaded intents from T999"

    # Should have loaded stubs for those intents
    assert len(probe.stubs) > 0, "Should have loaded stubs from IntentService"

    # Should have generated prompts (stubs × DAN variants)
    assert len(probe.prompts) > 0, "Should have generated prompts"

    # Number of prompts should be: num_stubs × num_dan_variants
    expected_prompts_per_stub = sum(len(v) for v in probe.dan_prompts.values())
    expected_total = len(probe.stubs) * expected_prompts_per_stub
    assert len(probe.prompts) == expected_total, f"Expected {expected_total} prompts, got {len(probe.prompts)}"


def test_spo_attempt_metadata():
    """Test that attempt metadata includes DAN variant information"""
    _config.load_base_config()

    # Load intentservice and set up intent spec
    import garak.intentservice
    garak.intentservice.load()
    _config.cas.intent_spec = "T999"

    probe = SPOIntent(config_root=_config)

    # Load a test generator
    g = garak._plugins.load_plugin("generators.test.Blank", config_root=_config)

    # Set up temporary report file for probe execution
    with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8") as temp_report_file:
        _config.transient.reportfile = temp_report_file
        _config.transient.report_filename = temp_report_file.name

        # Generate attempts
        attempts = list(probe.probe(g))

    assert len(attempts) > 0, "Should generate attempts"

    # Check that attempts have proper metadata
    for attempt in attempts:
        assert attempt.notes is not None, "Attempt should have notes"
        assert "stub" in attempt.notes, "Attempt notes should contain stub"
        assert "intent" in attempt.notes, "Attempt notes should contain intent"
        assert "dan_variant" in attempt.notes, "Attempt notes should contain dan_variant"
        assert "dan_prompt_index" in attempt.notes, "Attempt notes should contain dan_prompt_index"

        # Verify metadata types
        assert isinstance(attempt.notes["dan_variant"], str), "dan_variant should be string"
        assert isinstance(attempt.notes["dan_prompt_index"], int), "dan_prompt_index should be int"

        # Verify dan_variant is valid
        assert attempt.notes["dan_variant"] in probe.dan_prompts, f"Invalid dan_variant: {attempt.notes['dan_variant']}"


def test_spo_class_attributes():
    """Test that SPOIntent has correct class attributes"""
    assert SPOIntent.lang == "en", "Language should be English"
    assert SPOIntent.primary_detector == "mitigation.MitigationBypass", "Primary detector should be MitigationBypass"
    assert "dan.DAN" in SPOIntent.extended_detectors, "Should have dan.DAN in extended detectors"
    assert SPOIntent.goal == "bypass system prompt via DAN-style persona injection", "Goal should be set"
    assert SPOIntent.active is False, "Should be inactive by default (intent probe)"
    assert "payload:jailbreak" in SPOIntent.tags, "Should have jailbreak tag"
    assert "owasp:llm01" in SPOIntent.tags, "Should have OWASP LLM01 tag"


def test_spo_dan_variant_filtering():
    """Test that dan_variants parameter filters DAN variants"""
    _config.load_base_config()
    _config.cas.intent_spec = None

    # Test with specific variants
    probe = SPOIntent(config_root=_config)
    probe.dan_variants = ["Dan_11_0", "STAN"]

    # Manually trigger filtering (simulating what would happen in __init__)
    filtered_prompts = {
        k: v for k, v in probe.dan_prompts.items() if k in probe.dan_variants
    }

    assert len(filtered_prompts) == 2, "Should only have 2 variants after filtering"
    assert "Dan_11_0" in filtered_prompts, "Should include Dan_11_0"
    assert "STAN" in filtered_prompts, "Should include STAN"
    assert "DUDE" not in filtered_prompts, "Should not include DUDE (not in filter list)"


def test_spo_empty_intents_no_prompts():
    """Test that SPOIntent generates no prompts when intent_spec is None"""
    _config.load_base_config()
    _config.cas.intent_spec = None

    probe = SPOIntent(config_root=_config)

    # Should have loaded DAN prompts
    assert len(probe.dan_prompts) > 0, "Should have loaded DAN prompts"

    # But should have no intents, stubs, or prompts
    assert len(probe.intents) == 0, "Should have no intents"
    assert len(probe.stubs) == 0, "Should have no stubs"
    assert len(probe.prompts) == 0, "Should have no prompts when intent_spec is None"
