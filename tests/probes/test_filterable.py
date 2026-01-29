# SPDX-FileCopyrightText: Copyright (c) 2025 Red Hat
# SPDX-License-Identifier: Apache-2.0

"""Tests for FilterableIntentProbe."""

import pytest
from typing import Set, Tuple
from unittest.mock import MagicMock, patch

from garak import _config
from garak.probes.filterable import (
    FilterableIntentProbe,
    IntentStubPair,
    make_exclusion_filter,
)


@pytest.fixture(autouse=True)
def setup_config():
    """Ensure _config.cas.intent_spec is set for tests."""
    if not hasattr(_config.cas, "intent_spec"):
        _config.cas.intent_spec = None
    yield


class SimpleFilterableProbe(FilterableIntentProbe):
    """Minimal subclass for testing."""

    active = False

    def __init__(self, stubs=None, stub_intents=None, config_root=_config):
        self._test_stubs = stubs or ["stub1", "stub2", "stub3"]
        self._test_stub_intents = stub_intents or ["intent1", "intent2", "intent3"]
        super().__init__(config_root)

    def _populate_intents(self, intent_spec):
        self.intents = list(set(self._test_stub_intents))

    def _populate_stubs(self):
        self.stubs = self._test_stubs
        self.stub_intents = self._test_stub_intents

    def _apply_technique(self, stub: str):
        return [f"Attack with: {stub}"]


class TestIntentStubPair:
    """Tests for IntentStubPair dataclass."""

    def test_equality(self):
        pair1 = IntentStubPair(intent="intent1", stub="stub1")
        pair2 = IntentStubPair(intent="intent1", stub="stub1")
        pair3 = IntentStubPair(intent="intent1", stub="stub2")

        assert pair1 == pair2
        assert pair1 != pair3

    def test_hash(self):
        pair1 = IntentStubPair(intent="intent1", stub="stub1")
        pair2 = IntentStubPair(intent="intent1", stub="stub1")
        pair3 = IntentStubPair(intent="intent1", stub="stub2")

        # Same pairs should have same hash
        assert hash(pair1) == hash(pair2)
        # Can be used in sets
        pair_set = {pair1, pair2, pair3}
        assert len(pair_set) == 2

    def test_not_equal_to_non_pair(self):
        pair = IntentStubPair(intent="intent1", stub="stub1")
        assert pair != ("intent1", "stub1")
        assert pair != {"intent": "intent1", "stub": "stub1"}


class TestFilterableIntentProbe:
    def test_no_filter_all_pairs_included(self):
        probe = SimpleFilterableProbe()

        # No filter set - should have all prompts
        assert len(probe.prompts) == 3
        assert len(probe.prompt_notes) == 3

        # All active pairs returned
        active = probe.get_active_pairs()
        assert len(active) == 3

    def test_notes_contain_both_intent_and_stub(self):
        probe = SimpleFilterableProbe()

        for note in probe.prompt_notes:
            assert "intent" in note
            assert "stub" in note

        # Verify correct mapping
        assert probe.prompt_notes[0]["intent"] == "intent1"
        assert probe.prompt_notes[0]["stub"] == "stub1"

    def test_filter_excludes_pairs(self):
        probe = SimpleFilterableProbe()

        # Exclude pairs where intent is "intent2"
        def exclude_intent2(pair: IntentStubPair) -> bool:
            return pair.intent != "intent2"

        probe.set_filter(exclude_intent2)
        probe._build_prompts()

        # Should have 2 prompts (intent1 and intent3)
        assert len(probe.prompts) == 2
        intents = [note["intent"] for note in probe.prompt_notes]
        assert "intent2" not in intents

    def test_filter_with_tuple_set(self):
        probe = SimpleFilterableProbe()

        # Simulate successful pairs that should be excluded
        successful_pairs: Set[Tuple[str, str]] = {
            ("intent1", "stub1"),
            ("intent3", "stub3"),
        }

        def exclusion_filter(pair: IntentStubPair) -> bool:
            return (pair.intent, pair.stub) not in successful_pairs

        probe.set_filter(exclusion_filter)
        probe._build_prompts()

        # Only intent2/stub2 should be there
        assert len(probe.prompts) == 1
        assert probe.prompt_notes[0]["intent"] == "intent2"
        assert probe.prompt_notes[0]["stub"] == "stub2"

    def test_get_active_pairs_reflects_filter(self):
        probe = SimpleFilterableProbe()

        # Initially all pairs active
        assert len(probe.get_active_pairs()) == 3

        # Add filter
        probe.set_filter(lambda p: p.stub != "stub2")

        # Now only 2 pairs active
        active = probe.get_active_pairs()
        assert len(active) == 2
        stubs = [p.stub for p in active]
        assert "stub2" not in stubs

    def test_clear_filter(self):
        probe = SimpleFilterableProbe()

        # Set filter that excludes everything
        probe.set_filter(lambda p: False)
        assert len(probe.get_active_pairs()) == 0

        probe.clear_filter()
        assert len(probe.get_active_pairs()) == 3

    def test_filter_state_changes_reflected_in_probe(self):
        probe = SimpleFilterableProbe()

        # External state
        excluded: Set[Tuple[str, str]] = set()

        def dynamic_filter(pair: IntentStubPair) -> bool:
            return (pair.intent, pair.stub) not in excluded

        probe.set_filter(dynamic_filter)

        # Initially nothing excluded
        probe._build_prompts()
        assert len(probe.prompts) == 3

        # Exclude a pair
        excluded.add(("intent1", "stub1"))
        probe._build_prompts()
        assert len(probe.prompts) == 2

        excluded.add(("intent2", "stub2"))
        probe._build_prompts()
        assert len(probe.prompts) == 1

    def test_probe_rebuilds_prompts_with_filter(self):
        from garak.attempt import Message

        probe = SimpleFilterableProbe()

        # Set filter
        probe.set_filter(lambda p: p.intent == "intent1")

        # Set required attributes for _execute_all
        probe.parallel_attempts = False
        probe.generations = 1

        # Mock generator
        mock_generator = MagicMock()
        mock_generator.parallel_capable = False
        mock_generator.generate.return_value = [Message(text="response", lang="*")]
        mock_generator.clear_history = MagicMock()

        # Patch _config to avoid missing attributes
        with patch.object(_config, "buffmanager", MagicMock(buffs=[])), \
             patch.object(_config, "transient", MagicMock()):
            _config.transient.reportfile = MagicMock()

            # Call probe - should rebuild prompts with filter
            results = list(probe.probe(mock_generator))

        # Should only have 1 attempt (for intent1)
        assert len(results) == 1


class TestExclusionFilterFactory:

    def test_factory_creates_working_filter(self):
        successful: Set[Tuple[str, str]] = set()
        filter_fn = make_exclusion_filter(successful)

        pair = IntentStubPair(intent="test", stub="stub")
        assert filter_fn(pair) is True

        successful.add(("test", "stub"))
        assert filter_fn(pair) is False

    def test_factory_captures_set_by_reference(self):
        """Filter sees changes to set without re-registration."""
        successful: Set[Tuple[str, str]] = set()
        filter_fn = make_exclusion_filter(successful)

        probe = SimpleFilterableProbe()
        probe.set_filter(filter_fn)

        # Initially all pairs included
        assert len(probe.get_active_pairs()) == 3

        # Add exclusion - filter sees it automatically
        successful.add(("intent1", "stub1"))
        assert len(probe.get_active_pairs()) == 2

        successful.add(("intent2", "stub2"))
        assert len(probe.get_active_pairs()) == 1
