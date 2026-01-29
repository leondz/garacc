# SPDX-FileCopyrightText: Copyright (c) 2025 Red Hat
# SPDX-License-Identifier: Apache-2.0

"""Filterable intent probe module.
"""

from dataclasses import dataclass
from collections.abc import Iterable
from typing import Callable, List, Optional, Set, Tuple

from garak import _config
from garak.probes.base import IntentProbe, Probe
import garak.attempt


@dataclass
class IntentStubPair:
    """Represents an (intent, stub) tuple for filtering purposes."""

    intent: str
    stub: str

    def __hash__(self):
        return hash((self.intent, self.stub))

    def __eq__(self, other):
        if not isinstance(other, IntentStubPair):
            return False
        return self.intent == other.intent and self.stub == other.stub


IntentStubFilter = Callable[[IntentStubPair], bool]


class FilterableIntentProbe(IntentProbe):
    """IntentProbe subclass supporting callback-based filtering.

    The filter callback is invoked for each (intent, stub) pair during
    prompt generation. If the callback returns False, that pair is skipped.

    Example usage:
        # Harness tracks attacks externally
        successful_pairs: Set[Tuple[str, str]] = set()

        # Create filter that excludes successful pairs
        def exclusion_filter(pair: IntentStubPair) -> bool:
            return (pair.intent, pair.stub) not in successful_pairs

        probe.set_filter(exclusion_filter)

        # Just call probe() - filter is already set
        attempts = list(probe.probe(model))

        # Update successful_pairs based on results
        for attempt in attempts:
            if attack_succeeded(attempt):
                successful_pairs.add((attempt.notes["intent"], attempt.notes["stub"]))
    """

    def __init__(self, config_root=_config):
        self._intent_stub_filter: Optional[IntentStubFilter] = None
        super().__init__(config_root)

    def set_filter(self, filter_fn: IntentStubFilter) -> None:
        """Set the filter callback for (intent, stub) pairs.

        Args:
            filter_fn: Callable taking IntentStubPair, returning bool.
                       True = include pair, False = skip pair.
        """
        self._intent_stub_filter = filter_fn

    def clear_filter(self) -> None:
        """Remove the filter callback."""
        self._intent_stub_filter = None

    def _should_include_pair(self, intent: str, stub: str) -> bool:
        """Check if pair should be included based on callback."""
        if self._intent_stub_filter is None:
            return True
        return self._intent_stub_filter(IntentStubPair(intent=intent, stub=stub))

    def _build_prompts(self) -> None:
        """Build prompts with filtering"""
        self.prompts = []
        self.prompt_notes = []

        for i, stub in enumerate(self.stubs):
            intent = self.stub_intents[i]

            if not self._should_include_pair(intent, stub):
                continue

            prompts = self._apply_technique(stub)
            self.prompts.extend(prompts)
            # Store BOTH intent AND stub in notes
            self.prompt_notes.extend([{"intent": intent, "stub": stub}] * len(prompts))

    def get_active_pairs(self) -> List[IntentStubPair]:
        """Return list of (intent, stub) pairs that will be probed.
        """
        active = []
        for i, stub in enumerate(self.stubs):
            intent = self.stub_intents[i]
            if self._should_include_pair(intent, stub):
                active.append(IntentStubPair(intent=intent, stub=stub))
        return active

    def probe(self, generator) -> Iterable[garak.attempt.Attempt]:
        """Probe with filtered (intent, stub) pairs.

        Rebuilds prompts with current filter state and calls Probe.probe()
        """
        self._build_prompts()  # Rebuild with current filter
        return Probe.probe(self, generator)  # Skip attack_target()


def make_exclusion_filter(
        successful_pairs: Set[Tuple[str, str]]
) -> IntentStubFilter:
    """Factory that creates a filter capturing the successful_pairs set.

    The filter should capture the set by reference, so updates to the set are automatically
    reflected in subsequent filter calls

    Args:
        successful_pairs: Set of (intent, stub) tuples to exclude.

    Returns:
        A filter function that excludes pairs in the successful_pairs set.

    Example:
        successful_pairs: Set[Tuple[str, str]] = set()
        filter_fn = make_exclusion_filter(successful_pairs)
        probe.set_filter(filter_fn)

        # As attacks succeed, add to the set
        successful_pairs.add(("intent1", "stub1"))
        # Filter automatically excludes this pair in next probe() call
    """

    def filter_fn(pair: IntentStubPair) -> bool:
        return (pair.intent, pair.stub) not in successful_pairs

    return filter_fn
