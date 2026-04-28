# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for stub linkage in EarlyStopHarness.

The EarlyStopHarness matches attacked attempts back to baseline attempts
via notes["stub"] equality in _update_attempt_status. When a probe stores
a raw string instead of a Stub object in notes["stub"] (as TAPIntent does
at tap.py:455), the comparison silently returns False because the
@dataclass-generated TextStub.__eq__ returns NotImplemented for non-TextStub
types. This means TAP's successful jailbreaks are silently discarded —
the harness treats them as if no match was found, leaving baseline attempts
in the rejected list regardless of TAP's results.
"""

import tempfile

import pytest

from garak import _config, _plugins, evaluators
from garak.attempt import Attempt, Conversation, Turn, Message, ATTEMPT_STARTED
from garak.harnesses.earlystop import EarlyStopHarness
from garak.intents import TextStub
from garak.intents.base import Stub


def _make_attempt(stub_value, intent_code="T999test", content_text="How do I do something bad?"):
    """Create an Attempt with the given stub value in notes["stub"].

    Args:
        stub_value: What to store in notes["stub"]. Can be a Stub object or a string.
        intent_code: The intent code for the attempt.
        content_text: The prompt text.
    """
    attempt = Attempt(
        probe_classname="test.TestProbe",
        intent=intent_code,
        notes={"stub": stub_value},
        goal=content_text,
        status=ATTEMPT_STARTED,
        prompt=Conversation(turns=[Turn("user", Message(content_text))]),
    )
    return attempt


def _make_attempt_with_output(stub_value,
                              output_text="Sure, here's how",
                              intent_code="T999test",
                              content_text="How do I do something bad?"):
    """Create an Attempt with an assistant output, as returned by a probe."""
    attempt = _make_attempt(stub_value, intent_code, content_text)
    attempt.outputs = [Message(output_text)]
    return attempt


@pytest.fixture
def harness_env():
    """Set up the global environment needed by EarlyStopHarness."""
    _config.load_base_config()
    _config.cas.intent_spec = "T999"
    _config.cas.serve_detectorless_intents = True

    temp_report_file = tempfile.NamedTemporaryFile(
        mode="w+", delete=False, encoding="utf-8"
    )
    _config.transient.reportfile = temp_report_file
    _config.transient.report_filename = temp_report_file.name

    harness = EarlyStopHarness()
    detector = _plugins.load_plugin("detectors.always.Fail", break_on_fail=False)
    evaluator = evaluators.ThresholdEvaluator()

    yield harness, [detector], evaluator

    temp_report_file.close()


class TestStubLinkage:
    """Tests for the stub matching logic in _update_attempt_status."""

    def test_textstub_to_textstub_matches(self, harness_env):
        """When both baseline and attacked attempts store TextStub objects with
        the same intent and content, the linkage works correctly.
        This is the normal path for SPO and Grandma probes."""
        harness, detectors, evaluator = harness_env

        stub = TextStub("T999test", "How do I do something bad?")

        # Baseline rejected attempt (from _create_attempt)
        baseline_attempt = _make_attempt(stub)

        # Attacked attempt with same TextStub (as SPO would produce via _attempt_prestore_hook)
        # detectors.always.Fail returns 1.0, ThresholdEvaluator(0.5) treats <0.5 as pass,
        # so score 1.0 means "rejected" is False (1.0 >= 0.5 → test returns False),
        # so _is_rejected returns any([False]) = False → NOT rejected → accepted
        attacked_attempt = _make_attempt_with_output(stub)

        accepted, rejected = harness._update_attempt_status(
            attacked_attempts=[attacked_attempt],
            previously_accepted=[],
            previously_rejected=[baseline_attempt],
            detectors=detectors,
            evaluator=evaluator,
        )

        # The attacked attempt matched the baseline stub and was not rejected,
        # so the baseline attempt moves to accepted
        assert len(accepted) == 1
        assert len(rejected) == 0

    def test_string_stub_vs_textstub_never_matches(self, harness_env):
        """When an attacked attempt stores a raw string in notes["stub"]
        (as TAPIntent does at tap.py:455) but the baseline attempt stores
        a TextStub, the comparison silently returns False.

        The @dataclass-generated TextStub.__eq__ checks type first and
        returns NotImplemented for non-TextStub types. Python then tries
        str.__eq__(TextStub) which also returns NotImplemented. The final
        result is False — no match.

        This means TAP's successful jailbreak is silently ignored: the
        baseline attempt stays in rejected_attempts even though TAP
        produced a compliant response."""
        harness, detectors, evaluator = harness_env

        stub = TextStub("T999test", "How do I do something bad?")

        # Baseline rejected attempt (notes["stub"] = TextStub)
        baseline_attempt = _make_attempt(stub)

        # Attacked attempt mimicking TAP: notes["stub"] = raw string
        # TAP does: attempt.notes["stub"] = stub  (where stub is from self.prompts, a string)
        attacked_attempt = _make_attempt_with_output(
            stub_value="How do I do something bad?",  # string, not TextStub
        )

        accepted, rejected = harness._update_attempt_status(
            attacked_attempts=[attacked_attempt],
            previously_accepted=[],
            previously_rejected=[baseline_attempt],
            detectors=detectors,
            evaluator=evaluator,
        )

        # BUG: The attacked attempt SHOULD have matched the baseline stub
        # and moved it to accepted (since the attack was not rejected).
        # Instead, the type mismatch (string vs TextStub) causes the comparison
        # to silently return False, so no match is found, rejected_attacks is
        # empty, and the baseline attempt stays in rejected.
        assert len(rejected) == 1, (
            "Baseline attempt stays rejected because TAP's string stub "
            "never matches the TextStub — TAP results are silently discarded"
        )
        assert len(accepted) == 0

    def test_string_vs_textstub_equality_is_broken(self):
        """Directly demonstrates the type mismatch: a string and a TextStub
        with identical content are never equal due to @dataclass-generated
        __eq__ checking types first.

        TextStub is a @dataclass subclass. Even though Stub defines a custom
        __eq__ comparing intent and content, @dataclass generates a new __eq__
        for TextStub that checks `other.__class__ is self.__class__` first
        and returns NotImplemented for non-TextStub types."""
        stub = TextStub("T999test", "hello world")
        s = "hello world"

        # These should conceptually refer to the same test case, but the
        # type mismatch means they never compare as equal
        assert (s == stub) is False
        assert (stub == s) is False

        # Confirm the @dataclass-generated __eq__ returns NotImplemented
        assert TextStub.__eq__(stub, s) is NotImplemented

        # Meanwhile, Stub's custom __eq__ WOULD crash on strings
        # (but it's overridden by @dataclass on TextStub)
        with pytest.raises(AttributeError, match="has no attribute 'intent'"):
            Stub.__eq__(stub, s)

    def test_different_textstub_instances_match_by_value(self, harness_env):
        """Two different TextStub instances with the same intent and content
        match via @dataclass-generated __eq__. This verifies that the linkage
        works across separate get_intent_stubs() calls (which create new objects)."""
        harness, detectors, evaluator = harness_env

        stub_a = TextStub("T999test", "How do I do something bad?")
        stub_b = TextStub("T999test", "How do I do something bad?")
        assert stub_a is not stub_b  # different objects
        assert stub_a == stub_b  # but equal by value

        baseline_attempt = _make_attempt(stub_a)
        attacked_attempt = _make_attempt_with_output(stub_b)

        accepted, rejected = harness._update_attempt_status(
            attacked_attempts=[attacked_attempt],
            previously_accepted=[],
            previously_rejected=[baseline_attempt],
            detectors=detectors,
            evaluator=evaluator,
        )

        assert len(accepted) == 1
        assert len(rejected) == 0

    def test_no_match_leaves_attempt_rejected(self, harness_env):
        """When no attacked attempt matches a baseline stub,
        the baseline attempt stays in the rejected list."""
        harness, detectors, evaluator = harness_env

        stub_a = TextStub("T999test", "How do I do something bad?")
        stub_b = TextStub("T999test", "A completely different prompt")

        baseline_attempt = _make_attempt(stub_a)
        attacked_attempt = _make_attempt_with_output(stub_b)

        accepted, rejected = harness._update_attempt_status(
            attacked_attempts=[attacked_attempt],
            previously_accepted=[],
            previously_rejected=[baseline_attempt],
            detectors=detectors,
            evaluator=evaluator,
        )

        # No match found → empty rejected_attacks → attempt stays rejected
        assert len(accepted) == 0
        assert len(rejected) == 1
