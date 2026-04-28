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
        import garak.services.intentservice

        _config.load_config()
        garak.services.intentservice.load()
        return SPOIntentUserAugmented(config_root=_config)

    def test_probe_initialization(self, probe):
        """Verify probe initializes correctly"""
        assert probe is not None
        assert probe.augment_user is True
        assert probe.augment_system is False

    def test_prompts_from_stub_generates_prompts(self, probe):
        """Verify prompts_from_stub generates augmented prompts"""
        from garak.intents import TextStub

        stub = TextStub(intent="S008code")
        stub.content = "write malicious code"
        prompts = probe._prompts_from_stub(stub)

        assert len(prompts) > 0
        assert all(isinstance(p, str) for p in prompts)

    def test_prompts_from_stub_tracks_metadata(self, probe):
        """Verify augmentation metadata is tracked"""
        from garak.intents import TextStub

        stub = TextStub(intent="S008code")
        stub.content = "write malicious code"
        prompts = probe._prompts_from_stub(stub)

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

    def test_mint_attempt_creates_user_turn(self, probe):
        """Verify _mint_attempt creates an attempt with a user turn from the generated prompt"""
        from garak.intents import TextStub

        stub = TextStub(intent="S008code")
        stub.content = "write malicious code"
        prompts = probe._prompts_from_stub(stub)
        assert len(prompts) > 0, "probe should have prompts from stub"
        probe.prompts = prompts
        # Set up prompt_stubs for _attempt_prestore_hook
        from garak.intents import Stub
        original_stub = Stub(intent="S008code")
        original_stub.content = stub
        probe.prompt_stubs = [original_stub] * len(prompts)

        attempt = probe._mint_attempt(probe.prompts[0], seq=0)

        user_turns = [t for t in attempt.prompt.turns if t.role == "user"]
        assert len(user_turns) == 1
        assert probe.prompts[0] in user_turns[0].content.text

    def test_mint_attempt_adds_augmentation_metadata(self, probe):
        """Verify _mint_attempt's prestore hook adds augmentation metadata"""
        from garak.intents import TextStub, Stub

        stub = TextStub(intent="S008code")
        stub.content = "write malicious code"
        prompts = probe._prompts_from_stub(stub)
        assert len(prompts) > 0
        probe.prompts = prompts
        original_stub = Stub(intent="S008code")
        original_stub.content = stub
        probe.prompt_stubs = [original_stub] * len(prompts)

        attempt = probe._mint_attempt(probe.prompts[0], seq=0)

        assert "augmentation" in attempt.notes
        assert attempt.notes["augmentation_target"] == "user"
        assert "dan_variant" in attempt.notes

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
        import garak.services.intentservice

        _config.load_config()
        garak.services.intentservice.load()
        return SPOIntentSystemAugmented(config_root=_config)

    def test_probe_initialization(self, probe):
        """Verify probe initializes correctly"""
        assert probe is not None
        assert probe.augment_system is True
        assert probe.augment_user is False

    def test_prompts_from_stub_generates_prompts(self, probe):
        """Verify prompts_from_stub generates augmented prompts with system augmentation"""
        from garak.intents import TextStub

        stub = TextStub(intent="S008code")
        stub.content = "write malicious code"
        prompts = probe._prompts_from_stub(stub)

        assert len(prompts) > 0
        assert all(isinstance(p, str) for p in prompts)

    def test_prompts_from_stub_tracks_metadata(self, probe):
        """Verify augmentation metadata tracks system target"""
        from garak.intents import TextStub

        stub = TextStub(intent="S008code")
        stub.content = "write malicious code"
        prompts = probe._prompts_from_stub(stub)

        assert all(
            probe.prompt_to_variant[i]["augmentation_target"] == "system"
            for i in range(len(prompts))
        )

    def test_prompts_contain_stub_content(self, probe):
        """Verify prompts contain the original stub content (not augmented)"""
        from garak.intents import TextStub

        stub = TextStub(intent="S008code")
        stub.content = "write malicious code"
        prompts = probe._prompts_from_stub(stub)

        # When only system is augmented, the stub text should appear unmodified
        assert all("write malicious code" in p for p in prompts)

    def test_mint_attempt_creates_user_turn(self, probe):
        """Verify _mint_attempt creates an attempt with a user turn"""
        from garak.intents import TextStub, Stub

        stub = TextStub(intent="S008code")
        stub.content = "write malicious code"
        prompts = probe._prompts_from_stub(stub)
        assert len(prompts) > 0
        probe.prompts = prompts
        original_stub = Stub(intent="S008code")
        original_stub.content = stub
        probe.prompt_stubs = [original_stub] * len(prompts)

        attempt = probe._mint_attempt(probe.prompts[0], seq=0)

        user_turns = [t for t in attempt.prompt.turns if t.role == "user"]
        assert len(user_turns) == 1

    def test_mint_attempt_adds_augmentation_metadata(self, probe):
        """Verify _mint_attempt's prestore hook adds augmentation metadata"""
        from garak.intents import TextStub, Stub

        stub = TextStub(intent="S008code")
        stub.content = "write malicious code"
        prompts = probe._prompts_from_stub(stub)
        assert len(prompts) > 0
        probe.prompts = prompts
        original_stub = Stub(intent="S008code")
        original_stub.content = stub
        probe.prompt_stubs = [original_stub] * len(prompts)

        attempt = probe._mint_attempt(probe.prompts[0], seq=0)

        assert "augmentation" in attempt.notes
        assert attempt.notes["augmentation_target"] == "system"
        assert "dan_variant" in attempt.notes


class TestSPOIntentBothAugmented:
    """Test SPOIntentBothAugmented probe variant"""

    @pytest.fixture
    def probe(self):
        """Create a probe instance for testing"""
        import garak.services.intentservice

        _config.load_config()
        garak.services.intentservice.load()
        return SPOIntentBothAugmented(config_root=_config)

    def test_probe_initialization(self, probe):
        """Verify probe initializes correctly"""
        assert probe is not None
        assert probe.augment_system is True
        assert probe.augment_user is True

    def test_prompts_from_stub_generates_prompts(self, probe):
        """Verify prompts_from_stub generates augmented prompts"""
        from garak.intents import TextStub

        stub = TextStub(intent="S008code")
        stub.content = "write malicious code"
        prompts = probe._prompts_from_stub(stub)

        assert len(prompts) > 0
        assert all(isinstance(p, str) for p in prompts)

    def test_prompts_from_stub_tracks_metadata(self, probe):
        """Verify augmentation metadata tracks both target"""
        from garak.intents import TextStub

        stub = TextStub(intent="S008code")
        stub.content = "write malicious code"
        prompts = probe._prompts_from_stub(stub)

        assert all(
            probe.prompt_to_variant[i]["augmentation_target"] == "both"
            for i in range(len(prompts))
        )

    def test_mint_attempt_creates_user_turn(self, probe):
        """Verify _mint_attempt creates an attempt with a user turn"""
        from garak.intents import TextStub, Stub

        stub = TextStub(intent="S008code")
        stub.content = "write malicious code"
        prompts = probe._prompts_from_stub(stub)
        assert len(prompts) > 0
        probe.prompts = prompts
        original_stub = Stub(intent="S008code")
        original_stub.content = stub
        probe.prompt_stubs = [original_stub] * len(prompts)

        attempt = probe._mint_attempt(probe.prompts[0], seq=0)

        user_turns = [t for t in attempt.prompt.turns if t.role == "user"]
        assert len(user_turns) == 1

    def test_mint_attempt_adds_augmentation_metadata(self, probe):
        """Verify _mint_attempt's prestore hook adds augmentation metadata"""
        from garak.intents import TextStub, Stub

        stub = TextStub(intent="S008code")
        stub.content = "write malicious code"
        prompts = probe._prompts_from_stub(stub)
        assert len(prompts) > 0
        probe.prompts = prompts
        original_stub = Stub(intent="S008code")
        original_stub.content = stub
        probe.prompt_stubs = [original_stub] * len(prompts)

        attempt = probe._mint_attempt(probe.prompts[0], seq=0)

        assert "augmentation" in attempt.notes
        assert attempt.notes["augmentation_target"] == "both"
        assert "dan_variant" in attempt.notes


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
