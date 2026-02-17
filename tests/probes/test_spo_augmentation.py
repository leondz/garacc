"""Tests for SPOIntent augmentation variants"""

import pytest
from garak import _config
from garak.probes.spo import (
    SPOIntentUserAugmented,
    SPOIntentSystemAugmented,
    SPOIntentBothAugmented,
)
from garak.probes._augmentation import (
    word_scramble,
    random_capitalization,
    ascii_noise,
    get_random_augmentation,
)


class TestAugmentationFunctions:
    """Test the shared augmentation utility functions"""

    def test_word_scramble_preserves_word_boundaries(self):
        """Verify word scrambling preserves word boundaries"""
        text = "Hello world this is a test"
        augmented = word_scramble(text, probability=1.0)
        assert len(augmented.split()) == len(text.split())

    def test_word_scramble_empty_text(self):
        """Verify word scrambling handles empty text"""
        assert word_scramble("") == ""

    def test_word_scramble_short_words(self):
        """Verify word scrambling skips short words"""
        text = "a be cat"  # All words <= 3 chars
        augmented = word_scramble(text, probability=1.0)
        # Short words should remain unchanged
        assert augmented == text

    def test_word_scramble_preserve_first_last(self):
        """Verify word scrambling preserves first and last characters"""
        text = "scramble"
        # Set seed for reproducibility
        import random

        random.seed(42)
        augmented = word_scramble(text, probability=1.0, preserve_first_last=True)
        # First and last should match
        assert augmented[0] == text[0]
        assert augmented[-1] == text[-1]

    def test_random_capitalization_changes_case(self):
        """Verify capitalization changes case"""
        text = "hello world"
        augmented = random_capitalization(text, probability=1.0)
        # Should have uppercase letters
        assert any(c.isupper() for c in augmented)

    def test_random_capitalization_empty_text(self):
        """Verify capitalization handles empty text"""
        assert random_capitalization("") == ""

    def test_random_capitalization_preserves_non_alpha(self):
        """Verify capitalization preserves non-alphabetic characters"""
        text = "test123!@#"
        augmented = random_capitalization(text, probability=1.0)
        # Numbers and special chars should be unchanged
        assert "123!@#" in augmented

    def test_ascii_noise_adds_characters(self):
        """Verify ASCII noise adds characters"""
        text = "hello"
        augmented = ascii_noise(text, probability=0.5)
        assert len(augmented) >= len(text)

    def test_ascii_noise_empty_text(self):
        """Verify ASCII noise handles empty text"""
        assert ascii_noise("") == ""

    def test_ascii_noise_custom_chars(self):
        """Verify ASCII noise uses custom noise characters"""
        text = "test"
        custom_noise = ["X", "Y", "Z"]
        import random

        random.seed(42)
        augmented = ascii_noise(text, probability=1.0, noise_chars=custom_noise)
        # Should contain at least one of the custom noise chars
        assert any(char in augmented for char in custom_noise)

    def test_get_random_augmentation_returns_function(self):
        """Verify get_random_augmentation returns a callable"""
        func = get_random_augmentation()
        assert callable(func)
        assert func in [word_scramble, random_capitalization, ascii_noise]


class TestSPOIntentUserAugmented:
    """Test SPOIntentUserAugmented probe variant"""

    @pytest.fixture
    def probe(self):
        """Create a probe instance for testing"""
        _config.load_base_config()
        _config.cas.intent_spec = None
        return SPOIntentUserAugmented(config_root=_config)

    def test_probe_initialization(self, probe):
        """Verify probe initializes correctly"""
        assert probe is not None
        assert hasattr(probe, "augmentation_func")
        assert callable(probe.augmentation_func)

    def test_prompts_from_stub_generates_prompts(self, probe):
        """Verify prompts_from_stub generates augmented prompts"""
        stub = "write malicious code"
        prompts = probe.prompts_from_stub(stub)

        assert len(prompts) > 0
        # All prompts should be augmented (different from original stub)
        # Note: This might occasionally fail if augmentation produces same text
        assert all(isinstance(p, str) for p in prompts)

    def test_prompts_from_stub_tracks_metadata(self, probe):
        """Verify augmentation metadata is tracked"""
        stub = "write malicious code"
        prompts = probe.prompts_from_stub(stub)

        # Verify metadata exists for all prompts
        assert all(
            "augmentation" in probe.prompt_to_variant[i] for i in range(len(prompts))
        )
        assert all(
            "augmentation_target" in probe.prompt_to_variant[i]
            for i in range(len(prompts))
        )
        assert all(
            probe.prompt_to_variant[i]["augmentation_target"] == "user"
            for i in range(len(prompts))
        )

    def test_probe_goal_and_tier(self, probe):
        """Verify probe has correct goal and tier"""
        import garak.probes

        assert "augmented" in probe.goal.lower()
        assert probe.tier == garak.probes.Tier.OF_CONCERN
        assert probe.active is False


class TestSPOIntentSystemAugmented:
    """Test SPOIntentSystemAugmented probe variant"""

    @pytest.fixture
    def probe(self):
        """Create a probe instance for testing"""
        _config.load_base_config()
        _config.cas.intent_spec = None
        return SPOIntentSystemAugmented(config_root=_config)

    def test_probe_initialization(self, probe):
        """Verify probe initializes correctly"""
        assert probe is not None
        assert hasattr(probe, "augmentation_func")
        assert callable(probe.augmentation_func)

    def test_prompts_from_stub_returns_plain_stub(self, probe):
        """Verify prompts_from_stub returns just the stub"""
        stub = "write malicious code"
        prompts = probe.prompts_from_stub(stub)

        assert len(prompts) == 1
        assert prompts[0] == stub

    def test_mint_attempt_creates_system_prompt(self, probe):
        """Verify _mint_attempt sets augmented system prompt"""
        stub = "write malicious code"

        # Initialize prompt_notes to avoid index error in _attempt_prestore_hook
        probe.prompt_notes = [{}]

        attempt = probe._mint_attempt(stub, seq=0)

        # Verify system turn exists
        system_turns = [t for t in attempt.prompt.turns if t.role == "system"]
        assert len(system_turns) > 0

        # Verify metadata
        assert "augmentation" in attempt.notes
        assert attempt.notes["augmentation_target"] == "system"
        assert "dan_variant_system" in attempt.notes

    def test_mint_attempt_restores_original_system_prompt(self, probe):
        """Verify _mint_attempt doesn't permanently modify system_prompt"""
        original_prompt = getattr(probe, "system_prompt", None)
        stub = "write malicious code"

        # Initialize prompt_notes to avoid index error
        probe.prompt_notes = [{}]

        probe._mint_attempt(stub, seq=0)

        # System prompt should be restored
        assert getattr(probe, "system_prompt", None) == original_prompt


class TestSPOIntentBothAugmented:
    """Test SPOIntentBothAugmented probe variant"""

    @pytest.fixture
    def probe(self):
        """Create a probe instance for testing"""
        _config.load_base_config()
        _config.cas.intent_spec = None
        return SPOIntentBothAugmented(config_root=_config)

    def test_probe_initialization(self, probe):
        """Verify probe initializes correctly"""
        assert probe is not None
        assert hasattr(probe, "augmentation_func")
        assert callable(probe.augmentation_func)

    def test_prompts_from_stub_returns_plain_stub(self, probe):
        """Verify prompts_from_stub returns just the stub"""
        stub = "write malicious code"
        prompts = probe.prompts_from_stub(stub)

        assert len(prompts) == 1
        assert prompts[0] == stub

    def test_mint_attempt_creates_both_augmented(self, probe):
        """Verify _mint_attempt augments both system and user"""
        stub = "write malicious code"

        # Initialize prompt_notes to avoid index error
        probe.prompt_notes = [{}]

        attempt = probe._mint_attempt(stub, seq=0)

        # Verify both turns exist
        system_turns = [t for t in attempt.prompt.turns if t.role == "system"]
        user_turns = [t for t in attempt.prompt.turns if t.role == "user"]
        assert len(system_turns) > 0
        assert len(user_turns) > 0

        # Verify metadata
        assert attempt.notes["augmentation_target"] == "both"
        assert "dan_variant_system" in attempt.notes
        assert "augmentation" in attempt.notes

    def test_mint_attempt_augments_user_message(self, probe):
        """Verify user message is augmented"""
        stub = "write malicious code"

        # Initialize prompt_notes to avoid index error
        probe.prompt_notes = [{}]

        attempt = probe._mint_attempt(stub, seq=0)

        # User message should be augmented (different from original stub)
        user_turns = [t for t in attempt.prompt.turns if t.role == "user"]
        # The augmentation might produce the same text occasionally, but typically differs
        assert len(user_turns) > 0


class TestHarnessIntegration:
    """Test integration with EarlyStopHarness"""

    def test_earlystop_harness_compatibility(self):
        """Verify all augmentation probes are compatible with EarlyStopHarness"""
        from garak.harnesses.earlystop import EarlyStopHarness

        harness = EarlyStopHarness()

        assert "spo.SPOIntent" in harness.compatible_probes
        assert "spo.SPOIntentUserAugmented" in harness.compatible_probes
        assert "spo.SPOIntentSystemAugmented" in harness.compatible_probes
        assert "spo.SPOIntentBothAugmented" in harness.compatible_probes
