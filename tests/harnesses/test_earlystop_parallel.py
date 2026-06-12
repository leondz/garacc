# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for parallel rejection checking in EarlyStopHarness."""

import tempfile
import threading
import time

import pytest

from garak import _config, _plugins, evaluators
from garak.attempt import Attempt, Conversation, Turn, Message, ATTEMPT_STARTED
from garak.harnesses.earlystop import EarlyStopHarness
from garak.intents import TextStub


def _make_attempt_with_output(stub, output_text="Sure, here's how"):
    """Create an Attempt with an assistant output."""
    attempt = Attempt(
        probe_classname="test.TestProbe",
        intent=stub.intent,
        notes={"stub": stub},
        goal=stub.content,
        status=ATTEMPT_STARTED,
        prompt=Conversation(turns=[Turn("user", Message(stub.content))]),
    )
    attempt.outputs = [Message(output_text)]
    return attempt


def _make_baseline_attempt(stub):
    """Create a baseline Attempt (no outputs set via probe execution)."""
    attempt = Attempt(
        probe_classname="test.TestProbe",
        intent=stub.intent,
        notes={"stub": stub},
        goal=stub.content,
        status=ATTEMPT_STARTED,
        prompt=Conversation(turns=[Turn("user", Message(stub.content))]),
    )
    # Don't set outputs - baseline attempts start without them
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
    # Fail detector returns 1.0 (attack succeeded), which means "not rejected"
    detector = _plugins.load_plugin("detectors.always.Fail", break_on_fail=False)
    evaluator = evaluators.ThresholdEvaluator()

    yield harness, [detector], evaluator, temp_report_file

    temp_report_file.close()


class TestParallelRejectionChecking:
    """Tests for parallel execution in _update_attempt_status."""

    def test_sequential_when_parallel_attempts_false(self, harness_env):
        """When parallel_attempts is False, behavior is sequential (original)."""
        harness, detectors, evaluator, _ = harness_env
        _config.system.parallel_attempts = False

        stub = TextStub("T999test", "prompt1")
        baseline = _make_baseline_attempt(stub)
        attacked = _make_attempt_with_output(stub, "Sure, here's how")

        accepted, rejected = harness._update_attempt_status(
            attacked_attempts=[attacked],
            previously_accepted=[],
            previously_rejected=[baseline],
            detectors=detectors,
            evaluator=evaluator,
        )

        # Fail detector returns 1.0 -> evaluator.test(1.0) returns False (unsafe)
        # -> not rejected -> baseline moves to accepted
        assert len(accepted) == 1
        assert len(rejected) == 0

    def test_sequential_when_parallel_attempts_is_one(self, harness_env):
        """When parallel_attempts is 1, behavior is sequential."""
        harness, detectors, evaluator, _ = harness_env
        _config.system.parallel_attempts = 1

        stub = TextStub("T999test", "prompt1")
        baseline = _make_baseline_attempt(stub)
        attacked = _make_attempt_with_output(stub, "Sure, here's how")

        accepted, rejected = harness._update_attempt_status(
            attacked_attempts=[attacked],
            previously_accepted=[],
            previously_rejected=[baseline],
            detectors=detectors,
            evaluator=evaluator,
        )

        assert len(accepted) == 1
        assert len(rejected) == 0

    def test_parallel_execution_with_multiple_attempts(self, harness_env):
        """When parallel_attempts > 1, multiple attacks are processed in parallel."""
        harness, detectors, evaluator, _ = harness_env
        _config.system.parallel_attempts = 4

        stub1 = TextStub("T999test", "prompt1")
        stub2 = TextStub("T999test", "prompt2")

        baseline1 = _make_baseline_attempt(stub1)
        baseline2 = _make_baseline_attempt(stub2)

        attacked1 = _make_attempt_with_output(stub1, "Sure, here's how 1")
        attacked2 = _make_attempt_with_output(stub2, "Sure, here's how 2")

        accepted, rejected = harness._update_attempt_status(
            attacked_attempts=[attacked1, attacked2],
            previously_accepted=[],
            previously_rejected=[baseline1, baseline2],
            detectors=detectors,
            evaluator=evaluator,
        )

        # Both should be accepted
        assert len(accepted) == 2
        assert len(rejected) == 0

    def test_parallel_execution_verifies_concurrency(self, harness_env):
        """Verify that parallel execution actually runs concurrently."""
        harness, _, evaluator, _ = harness_env
        _config.system.parallel_attempts = 4

        # Track concurrent execution
        call_times = []
        lock = threading.Lock()

        class SlowDetector:
            detectorname = "test.SlowDetector"
            skip = False

            def detect(self, attempt):
                start = time.time()
                time.sleep(0.1)  # Simulate slow LLM call
                end = time.time()
                with lock:
                    call_times.append((start, end))
                return [1.0]  # Return "not rejected" (attack succeeded)

        stub1 = TextStub("T999test", "prompt1")
        stub2 = TextStub("T999test", "prompt2")
        stub3 = TextStub("T999test", "prompt3")
        stub4 = TextStub("T999test", "prompt4")

        baselines = [_make_baseline_attempt(s) for s in [stub1, stub2, stub3, stub4]]
        attackeds = [_make_attempt_with_output(s, "response") for s in [stub1, stub2, stub3, stub4]]

        start_time = time.time()
        accepted, rejected = harness._update_attempt_status(
            attacked_attempts=attackeds,
            previously_accepted=[],
            previously_rejected=baselines,
            detectors=[SlowDetector()],
            evaluator=evaluator,
        )
        total_time = time.time() - start_time

        # With parallelism, 4 x 0.1s calls should take ~0.1-0.2s, not 0.4s
        # Allow some margin for thread overhead
        assert total_time < 0.35, f"Expected parallel execution but took {total_time}s"
        assert len(call_times) == 4

    def test_reportfile_thread_safety(self, harness_env):
        """Verify reportfile writes are serialized and don't corrupt."""
        harness, detectors, evaluator, report_file = harness_env
        _config.system.parallel_attempts = 8

        stubs = [TextStub("T999test", f"prompt{i}") for i in range(10)]

        baselines = [_make_baseline_attempt(s) for s in stubs]
        attackeds = [_make_attempt_with_output(s, f"response{i}") for i, s in enumerate(stubs)]

        harness._update_attempt_status(
            attacked_attempts=attackeds,
            previously_accepted=[],
            previously_rejected=baselines,
            detectors=detectors,
            evaluator=evaluator,
        )

        # Read back and verify each line is valid JSON
        report_file.seek(0)
        lines = report_file.readlines()
        assert len(lines) == 10  # One line per attacked attempt

        import json
        for line in lines:
            # Should not raise JSONDecodeError
            data = json.loads(line.strip())
            assert "uuid" in data
            assert "detector_results" in data

    def test_error_handling_treats_failure_as_rejected(self, harness_env):
        """When a detector raises an exception, treat it as rejected (safe default)."""
        harness, _, evaluator, _ = harness_env
        _config.system.parallel_attempts = 4

        class FailingDetector:
            detectorname = "test.FailingDetector"
            skip = False

            def detect(self, attempt):
                raise RuntimeError("Simulated detector failure")

        stub = TextStub("T999test", "prompt1")
        baseline = _make_baseline_attempt(stub)
        attacked = _make_attempt_with_output(stub, "response")

        accepted, rejected = harness._update_attempt_status(
            attacked_attempts=[attacked],
            previously_accepted=[],
            previously_rejected=[baseline],
            detectors=[FailingDetector()],
            evaluator=evaluator,
        )

        # Error -> treated as rejected -> baseline stays rejected
        assert len(accepted) == 0
        assert len(rejected) == 1

    def test_mixed_stubs_parallel(self, harness_env):
        """Multiple attacks per stub are correctly grouped in parallel mode."""
        harness, detectors, evaluator, _ = harness_env
        _config.system.parallel_attempts = 8

        stub1 = TextStub("T999test", "prompt1")
        stub2 = TextStub("T999test", "prompt2")

        baseline1 = _make_baseline_attempt(stub1)
        baseline2 = _make_baseline_attempt(stub2)

        # Multiple attacks for stub1, one for stub2
        attacked1a = _make_attempt_with_output(stub1, "response1a")
        attacked1b = _make_attempt_with_output(stub1, "response1b")
        attacked2 = _make_attempt_with_output(stub2, "response2")

        accepted, rejected = harness._update_attempt_status(
            attacked_attempts=[attacked1a, attacked1b, attacked2],
            previously_accepted=[],
            previously_rejected=[baseline1, baseline2],
            detectors=detectors,
            evaluator=evaluator,
        )

        # Both baselines should be accepted
        assert len(accepted) == 2
        assert len(rejected) == 0
