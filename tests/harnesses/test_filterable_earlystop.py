# SPDX-FileCopyrightText: Copyright (c) 2025 Red Hat
# SPDX-License-Identifier: Apache-2.0

"""Tests for FilterableEarlyStopHarness."""

import tempfile
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from garak import _config
from garak.attempt import Message
from garak.harnesses.filterable_earlystop import FilterableEarlyStopHarness
from garak.probes.filterable import FilterableIntentProbe


class MockFilterableProbe(FilterableIntentProbe):
    """Mock FilterableIntentProbe for testing."""

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

    def _apply_technique(self, stub: str) -> List[str]:
        return [f"Attack: {stub}"]


@pytest.fixture
def setup_config():
    _config.load_base_config()

    # Set up cas config
    if not hasattr(_config, "cas"):
        _config.cas = MagicMock()
    _config.cas.intent_spec = None
    _config.cas.policy_data_path = None
    _config.cas.policy_points = {"T999": False}

    # Set up temp report file
    temp_report_file = tempfile.NamedTemporaryFile(
        mode="w+", delete=False, encoding="utf-8"
    )
    _config.transient.reportfile = temp_report_file
    _config.transient.report_filename = temp_report_file.name

    yield temp_report_file

    temp_report_file.close()


class TestFilterableEarlyStopHarness:

    def test_filter_applied_to_probes(self, setup_config):
        """Filter is applied to each probe before probing."""
        harness = FilterableEarlyStopHarness()

        mock_generator = MagicMock()
        mock_generator.parallel_capable = False
        mock_generator.generate.return_value = [Message(text="response", lang="*")]
        mock_generator.clear_history = MagicMock()

        mock_evaluator = MagicMock()
        mock_evaluator.test.return_value = False  # No successes

        mock_probe = MockFilterableProbe()
        mock_probe.parallel_attempts = False
        mock_probe.generations = 1

        mock_detector = MagicMock()
        mock_detector.detectorname = "garak.detectors.always.Fail"
        mock_detector.skip = False
        mock_detector.detect.return_value = [0.0]

        with patch.object(harness, '_load_filterable_probes', return_value=[mock_probe]):
            with patch.object(harness, '_load_detectors', return_value=[mock_detector]):
                with patch.object(_config, "buffmanager", MagicMock(buffs=[])):
                    harness.run(mock_generator, ["probes.test"], ["detectors.test"], mock_evaluator)

        # Filter should have been set on the probe
        assert mock_probe._intent_stub_filter is not None

    def test_successful_pairs_excluded_in_subsequent_probes(self, setup_config):
        """Successful (intent, stub) pairs are excluded from later probes."""
        harness = FilterableEarlyStopHarness()

        mock_generator = MagicMock()
        mock_generator.parallel_capable = False
        mock_generator.generate.return_value = [Message(text="response", lang="*")]
        mock_generator.clear_history = MagicMock()

        # Evaluator returns True (success) for first call, False for others
        mock_evaluator = MagicMock()
        call_count = [0]

        def mock_test(score):
            call_count[0] += 1
            return call_count[0] == 1  # First detection is a success

        mock_evaluator.test.side_effect = mock_test

        # Two probes - second should see first probe's success filtered out
        probe1 = MockFilterableProbe(
            stubs=["stub1"],
            stub_intents=["intent1"]
        )
        probe1.parallel_attempts = False
        probe1.generations = 1

        probe2 = MockFilterableProbe(
            stubs=["stub1", "stub2"],
            stub_intents=["intent1", "intent2"]
        )
        probe2.parallel_attempts = False
        probe2.generations = 1

        mock_detector = MagicMock()
        mock_detector.detectorname = "garak.detectors.test.Detector"
        mock_detector.skip = False
        mock_detector.detect.return_value = [1.0]

        with patch.object(harness, '_load_filterable_probes', return_value=[probe1, probe2]):
            with patch.object(harness, '_load_detectors', return_value=[mock_detector]):
                with patch.object(_config, "buffmanager", MagicMock(buffs=[])):
                    harness.run(mock_generator, ["p1", "p2"], ["d1"], mock_evaluator)

        # After probe1, (intent1, stub1) should be in exclusion set
        # probe2 should only process stub2 (not stub1)
        # Check via the active pairs after filter is set
        active = probe2.get_active_pairs()
        assert len(active) == 1
        assert active[0].stub == "stub2"
        assert active[0].intent == "intent2"

    def test_early_stop_when_all_pairs_succeed(self, setup_config):
        """Harness stops early when all pairs have succeeded."""
        harness = FilterableEarlyStopHarness()

        mock_generator = MagicMock()
        mock_generator.parallel_capable = False
        mock_generator.generate.return_value = [Message(text="response", lang="*")]
        mock_generator.clear_history = MagicMock()

        mock_evaluator = MagicMock()
        mock_evaluator.test.return_value = True  # All succeed

        probe1 = MockFilterableProbe(
            stubs=["stub1"],
            stub_intents=["intent1"]
        )
        probe1.parallel_attempts = False
        probe1.generations = 1

        # probe2 has same pairs as probe1 - should be skipped
        probe2 = MockFilterableProbe(
            stubs=["stub1"],
            stub_intents=["intent1"]
        )
        probe2.parallel_attempts = False
        probe2.generations = 1

        mock_detector = MagicMock()
        mock_detector.detectorname = "garak.detectors.test.Detector"
        mock_detector.skip = False
        mock_detector.detect.return_value = [1.0]

        probe_calls = []

        original_probe1 = probe1.probe

        def track_probe1_call(gen):
            probe_calls.append("probe1")
            return original_probe1(gen)

        original_probe2 = probe2.probe

        def track_probe2_call(gen):
            probe_calls.append("probe2")
            return original_probe2(gen)

        probe1.probe = track_probe1_call
        probe2.probe = track_probe2_call

        with patch.object(harness, '_load_filterable_probes', return_value=[probe1, probe2]):
            with patch.object(harness, '_load_detectors', return_value=[mock_detector]):
                with patch.object(_config, "buffmanager", MagicMock(buffs=[])):
                    harness.run(mock_generator, ["p1", "p2"], ["d1"], mock_evaluator)

        # Only probe1 should have been called
        assert probe_calls == ["probe1"]

    def test_attempts_written_to_report_file(self, setup_config):
        """All attempts are written to the report file."""
        report_file = setup_config

        # Get initial line count (other tests may have written to this file)
        report_file.flush()
        report_file.seek(0)
        initial_lines = len(report_file.readlines())

        harness = FilterableEarlyStopHarness()

        mock_generator = MagicMock()
        mock_generator.parallel_capable = False
        mock_generator.generate.return_value = [Message(text="response", lang="*")]
        mock_generator.clear_history = MagicMock()

        mock_evaluator = MagicMock()
        mock_evaluator.test.return_value = False

        probe = MockFilterableProbe(
            stubs=["stub1", "stub2"],
            stub_intents=["intent1", "intent2"]
        )
        probe.parallel_attempts = False
        probe.generations = 1

        mock_detector = MagicMock()
        mock_detector.detectorname = "garak.detectors.test.Detector"
        mock_detector.skip = False
        mock_detector.detect.return_value = [0.0]

        with patch.object(harness, '_load_filterable_probes', return_value=[probe]):
            with patch.object(harness, '_load_detectors', return_value=[mock_detector]):
                with patch.object(_config, "buffmanager", MagicMock(buffs=[])):
                    harness.run(mock_generator, ["p1"], ["d1"], mock_evaluator)

        report_file.flush()
        report_file.seek(0)
        lines = report_file.readlines()[initial_lines:]

        # Should have written attempts for both stubs
        # (Note: base probe class also writes, so we check >= 2)
        assert len(lines) >= 2

        # Verify both stubs appear in written entries
        content = "".join(lines)
        assert "stub1" in content
        assert "stub2" in content

