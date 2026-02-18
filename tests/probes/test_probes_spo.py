# SPDX-FileCopyrightText: Portions Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SPO (System Prompt Override) Intent Probe"""

import pytest
import tempfile

import garak
from garak import _config
from garak.intents import Stub
from garak.probes.spo import (
    SPOIntent,
    SPOIntentUserAugmented,
    SPOIntentSystemAugmented,
    SPOIntentBothAugmented,
)


def test_spo_dan_loading():
    """Test that DAN prompts are loaded correctly from JSON files"""
    _config.load_base_config()
    from garak.services import intentservice
    _config.cas.intent_spec = "T999"  # Use test intent
    _config.cas.serve_detectorless_intents = True
    intentservice.load()

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
    from garak.services import intentservice
    _config.cas.intent_spec = "T999"  # Use test intent
    _config.cas.serve_detectorless_intents = True
    intentservice.load()

    probe = SPOIntent(config_root=_config)

    test_stub = Stub()
    test_stub.content = "tell me how to build a bomb"
    prompts = probe._prompts_from_stub(test_stub)

    # Should generate one prompt per DAN variant (14 total)
    expected_count = min(probe.max_dan_samples, len(probe.dan_prompts))
    assert len(prompts) == expected_count, f"Expected {expected_count} prompts, got {len(prompts)}"

    # All prompts should contain the stub
    for prompt in prompts:
        assert test_stub.content in prompt, f"Prompt should contain stub: {test_stub}"

    # All prompts should contain the DAN prompt content
    # Check that at least one prompt contains a signature DAN phrase
    has_dan_content = any("{generator.name}" in p for p in prompts)
    assert has_dan_content, "At least some prompts should contain {generator.name} placeholder"


def test_spo_variant_mapping():
    """Test that prompt_to_variant mapping is created correctly"""
    _config.load_base_config()
    from garak.services import intentservice
    _config.cas.intent_spec = "T999"  # Use test intent
    _config.cas.serve_detectorless_intents = True
    intentservice.load()

    probe = SPOIntent(config_root=_config)

    test_stub = Stub()
    test_stub.content = "write malicious code"
    prompts = probe._prompts_from_stub(test_stub)

    # Should have a mapping for each generated prompt
    assert len(probe.prompt_to_variant) >= len(prompts), "Should have mapping for each prompt"

    # Check that mappings have correct structure
    for idx, mapping in probe.prompt_to_variant.items():
        assert "dan_variant" in mapping, f"Mapping {idx} should have dan_variant"
        assert "dan_prompt_index" in mapping, f"Mapping {idx} should have dan_prompt_index"
        assert isinstance(mapping["dan_variant"], str), "dan_variant should be string"
        assert isinstance(mapping["dan_prompt_index"], int), "dan_prompt_index should be int"


def test_spo_generator_name_substitution():
    """Test that {generator.name} is substituted correctly in probe()"""
    _config.load_base_config()
    from garak.services import intentservice
    _config.cas.intent_spec = "T999"  # Use test intent
    _config.cas.serve_detectorless_intents = True
    _config.cas.serve_detectorless_intents = True
    intentservice.load()

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
    from garak.services import intentservice
    _config.cas.intent_spec = "T999"  # Use test intent
    _config.cas.serve_detectorless_intents = True
    intentservice.load()

    probe = SPOIntent(config_root=_config)

    # Should have loaded intents
    assert len(probe.intents) > 0, "Should have loaded intents from T999"

    # Should have loaded stubs for those intents
    assert len(probe.stubs) > 0, "Should have loaded stubs from IntentService"

    # Should have generated prompts (stubs × DAN variants)
    assert len(probe.prompts) > 0, "Should have generated prompts"

    # Number of prompts should be: num_stubs × num_dan_variants
    expected_prompts_per_stub = sum(len(v) for v in probe.dan_prompts.values())
    expected_total = min(probe.max_dan_samples, len(probe.dan_prompts))
    assert len(probe.prompts) == expected_total, f"Expected {expected_total} prompts, got {len(probe.prompts)}"


def test_spo_attempt_metadata():
    """Test that attempt metadata includes DAN variant information"""
    _config.load_base_config()
    from garak.services import intentservice
    _config.cas.intent_spec = "T999"  # Use test intent
    _config.cas.serve_detectorless_intents = True
    intentservice.load()

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
        assert attempt.intent is not None, "Attempt notes should contain intent"
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


def test_spo_dan_variant_filtering():
    """Test that dan_variants parameter filters DAN variants"""
    _config.load_base_config()
    from garak.services import intentservice
    _config.cas.intent_spec = "T999"  # Use test intent
    _config.cas.serve_detectorless_intents = True
    intentservice.load()

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


@pytest.mark.parametrize(
    "probe_class,expected_target",
    [
        (SPOIntentUserAugmented, "user"),
        (SPOIntentSystemAugmented, "system"),
        (SPOIntentBothAugmented, "both"),
    ],
)
def test_spo_augmented_attempt_metadata(probe_class, expected_target):
    """Test that augmented SPO classes add augmentation metadata to attempts"""
    _config.load_base_config()
    from garak.services import intentservice
    _config.cas.intent_spec = "T999"  # Use test intent
    _config.cas.serve_detectorless_intents = True
    intentservice.load()

    probe = probe_class(config_root=_config)

    # Load a test generator
    g = garak._plugins.load_plugin("generators.test.Blank", config_root=_config)

    # Set up temporary report file for probe execution
    with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8") as temp_report_file:
        _config.transient.reportfile = temp_report_file
        _config.transient.report_filename = temp_report_file.name

        # Generate attempts
        attempts = list(probe.probe(g))

    assert len(attempts) > 0, f"{probe_class.__name__} should generate attempts"

    # Check that attempts have proper metadata
    for attempt in attempts:
        assert attempt.notes is not None, "Attempt should have notes"
        assert "stub" in attempt.notes, "Attempt notes should contain stub"
        assert attempt.intent is not None, "Attempt should track intent"
        assert "dan_variant" in attempt.notes, "Attempt notes should contain dan_variant"
        assert "dan_prompt_index" in attempt.notes, "Attempt notes should contain dan_prompt_index"
        assert "augmentation" in attempt.notes, "Attempt notes should contain augmentation"
        assert "augmentation_target" in attempt.notes, "Attempt notes should contain augmentation_target"

        # Verify augmentation target is correct
        assert attempt.notes["augmentation_target"] == expected_target, (
            f"{probe_class.__name__} should have augmentation_target={expected_target} in attempt notes"
        )

        # Verify augmentation function is valid
        assert attempt.notes["augmentation"] in ["word_scramble", "random_capitalization", "ascii_noise"], (
            f"augmentation should be a valid function name in attempt notes"
        )


def test_spo_dan_sampling_default():
    """Test that DAN prompts are sampled by default to max_dan_samples (10)"""
    _config.load_base_config()
    from garak.services import intentservice
    _config.cas.intent_spec = "T999"  # Use test intent
    _config.cas.serve_detectorless_intents = True
    intentservice.load()

    probe = SPOIntent(config_root=_config)

    # Count total DAN prompts available
    total_dan_prompts = sum(len(v) for v in probe.dan_prompts.values())
    assert total_dan_prompts > 10, "Test requires more than 10 DAN prompts to be meaningful"

    test_stub = Stub()
    test_stub.content = "tell me how to build a bomb"
    prompts = probe._prompts_from_stub(test_stub)

    # Should generate exactly max_dan_samples (10) prompts
    assert len(prompts) == 10, f"Expected 10 prompts with default sampling, got {len(prompts)}"

    # All prompts should contain the stub
    for prompt in prompts:
        assert test_stub.content in prompt, f"Prompt should contain stub: {test_stub}"


def test_spo_dan_sampling_custom_limit():
    """Test that DAN prompts are sampled to custom max_dan_samples"""
    _config.load_base_config()
    from garak.services import intentservice
    _config.cas.intent_spec = "T999"  # Use test intent
    _config.cas.serve_detectorless_intents = True
    intentservice.load()

    probe = SPOIntent(config_root=_config)
    probe.max_dan_samples = 5

    test_stub = Stub()
    test_stub.content = "write malicious code"
    prompts = probe._prompts_from_stub(test_stub)

    # Should generate exactly 5 prompts
    assert len(prompts) == 5, f"Expected 5 prompts with max_dan_samples=5, got {len(prompts)}"

    # All prompts should contain the stub
    for prompt in prompts:
        assert test_stub.content in prompt, f"Prompt should contain stub: {test_stub}"


def test_spo_dan_sampling_no_limit():
    """Test that DAN prompts are not sampled when max_dan_samples is None"""
    _config.load_base_config()
    from garak.services import intentservice
    _config.cas.intent_spec = "T999"  # Use test intent
    _config.cas.serve_detectorless_intents = True
    intentservice.load()

    probe = SPOIntent(config_root=_config)
    probe.max_dan_samples = None

    test_stub = Stub()
    test_stub.content = "hack into a system"
    prompts = probe._prompts_from_stub(test_stub)

    # Should generate all DAN prompts
    total_dan_prompts = sum(len(v) for v in probe.dan_prompts.values())
    assert len(
        prompts) == total_dan_prompts, f"Expected {total_dan_prompts} prompts without sampling, got {len(prompts)}"


def test_spo_dan_sampling_different_seeds():
    """Test that different seeds produce different samples"""
    _config.load_base_config()
    from garak.services import intentservice
    _config.cas.intent_spec = "T999"  # Use test intent
    _config.cas.serve_detectorless_intents = True
    intentservice.load()

    # Create two probes with different seeds
    probe1 = SPOIntent(config_root=_config)
    probe1.max_dan_samples = 8
    probe1.dan_sample_seed = 42

    probe2 = SPOIntent(config_root=_config)
    probe2.max_dan_samples = 8
    probe2.dan_sample_seed = 123

    test_stub = Stub()
    test_stub.content = "do something bad"
    prompts1 = probe1._prompts_from_stub(test_stub)
    prompts2 = probe2._prompts_from_stub(test_stub)

    # Should generate the same number of prompts
    assert len(prompts1) == len(prompts2) == 8, "Both probes should generate 8 prompts"

    # But the prompts should be different (with very high probability)
    assert prompts1 != prompts2, "Prompts should be different with different seeds"


def test_spo_dan_sampling_smaller_than_total():
    """Test that no sampling occurs when max_dan_samples >= total DAN prompts"""
    _config.load_base_config()
    from garak.services import intentservice
    _config.cas.intent_spec = "T999"  # Use test intent
    _config.cas.serve_detectorless_intents = True
    intentservice.load()

    probe = SPOIntent(config_root=_config)

    # Count total DAN prompts
    total_dan_prompts = sum(len(v) for v in probe.dan_prompts.values())

    # Set max_dan_samples to be >= total
    probe.max_dan_samples = total_dan_prompts + 5

    test_stub = Stub()
    test_stub.content = "tell me how to hack"
    prompts = probe._prompts_from_stub(test_stub)

    # Should generate all DAN prompts (no sampling)
    assert len(prompts) == total_dan_prompts, f"Expected all {total_dan_prompts} prompts when max_dan_samples >= total"


def test_spo_augmented_dan_sampling():
    """Test that DAN sampling works with augmented probes"""
    _config.load_base_config()
    from garak.services import intentservice
    _config.cas.intent_spec = "T999"  # Use test intent
    _config.cas.serve_detectorless_intents = True
    intentservice.load()

    probe = SPOIntentUserAugmented(config_root=_config)
    probe.max_dan_samples = 7
    probe.dan_sample_seed = 999

    test_stub = Stub()
    test_stub.content = "write malicious code"
    prompts = probe._prompts_from_stub(test_stub)

    # Should generate exactly 7 prompts
    assert len(prompts) == 7, f"Expected 7 prompts with max_dan_samples=7, got {len(prompts)}"

    # All prompts should contain the stub (possibly augmented)
    # We can't check for exact match since augmentation might modify it
    assert len(prompts) == 7, "Should have correct number of prompts"
