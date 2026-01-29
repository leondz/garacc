# SPDX-FileCopyrightText: Copyright (c) 2025 Red Hat
# SPDX-License-Identifier: Apache-2.0

"""Filterable early stopping harness
"""
import json
import logging
from typing import Set, Tuple

import tqdm

from garak.cas import Policy
from garak.harnesses import Harness
from garak import _config, _plugins
from garak.attempt import ATTEMPT_COMPLETE
from garak.probes.filterable import FilterableIntentProbe, make_exclusion_filter


def _load_policy_from_config(config_root) -> Policy:
    policy_data_path = getattr(config_root.cas, 'policy_data_path', None)
    policy_points = getattr(config_root.cas, 'policy_points', {})

    policy = Policy(autoload=(policy_data_path is None and not policy_points))

    if policy_data_path:
        policy._load_trait_typology(policy_data_path)

    for trait_code, value in policy_points.items():
        policy.points[trait_code] = value

    policy.propagate_up()
    return policy


class FilterableEarlyStopHarness(Harness):
    """Early stopping harness using FilterableIntentProbe.

    Uses callback-based filtering instead of manual stub management.
    The harness tracks successful (intent, stub) pairs and automatically
    excludes them from subsequent probes.
    """

    DEFAULT_PARAMS = {
        "compatible_probes": [
            "grandma.GrandmaIntent",
            "dan_intent.Dan_11_0_Intent",
        ],
    }

    def __init__(self, config_root=_config):
        super().__init__(config_root)
        self.policy = _load_policy_from_config(config_root)

    def _load_filterable_probes(self, probe_names):
        """Load probes, filtering to FilterableIntentProbe subclasses."""
        probes = []
        for probename in probe_names:
            try:
                probe = _plugins.load_plugin(probename)
            except Exception as e:
                logging.error(f"{probename} load exception, skipping: {e}")
                continue

            if not probe:
                logging.warning(f"{probename} load failed, skipping")
                continue

            # Check FilterableIntentProbe compatibility
            if not isinstance(probe, FilterableIntentProbe):
                logging.warning(f"{probename} is not a FilterableIntentProbe - skipping")
                continue

            # Check against compatible list
            short_name = probename.replace("probes.", "")
            if self.compatible_probes and short_name not in self.compatible_probes:
                logging.warning(f"{short_name} not in compatible_probes - skipping")
                continue

            probes.append(probe)
        return probes

    def _load_detectors(self, detector_names):
        """Load detectors."""
        detectors = []
        for name in detector_names:
            detector = _plugins.load_plugin(name, break_on_fail=False)
            if detector:
                detectors.append(detector)
            else:
                logging.warning(f"detector load failed: {name}, skipping")
        return detectors

    def run(self, model, probe_names, detector_names, evaluator, buff_names=None):
        if buff_names is None:
            buff_names = []

        self._load_buffs(buff_names)

        # Load and filter probes to FilterableIntentProbe only
        attack_methods = self._load_filterable_probes(probe_names)

        if not attack_methods:
            raise ValueError("No compatible FilterableIntentProbe attack methods loaded")

        # Load detectors
        detectors = self._load_detectors(detector_names)

        if not detectors:
            raise ValueError("No detectors loaded")

        # state
        successful_pairs: Set[Tuple[str, str]] = set()
        exclusion_filter = make_exclusion_filter(successful_pairs)

        self._start_run_hook()
        all_attempts = []

        for probe in attack_methods:
            probe.set_filter(exclusion_filter)

            # Check if any pairs remain to probe
            active_pairs = probe.get_active_pairs()
            if not active_pairs:
                logging.info("All pairs succeeded, stopping early")
                break

            logging.info(f"Probing {len(active_pairs)} pairs with {probe.__class__.__name__}")

            # Run probe - filter is already set, prompts rebuilt inside probe()
            attempts = list(probe.probe(model))

            # Run detectors and track successes
            for d in detectors:
                logging.debug("harness: run detector %s", d.detectorname)
                attempt_iterator = tqdm.tqdm(attempts, leave=False)
                detector_probe_name = d.detectorname.replace("garak.detectors.", "")
                attempt_iterator.set_description("detectors." + detector_probe_name)

                for attempt in attempt_iterator:
                    if d.skip:
                        continue
                    result = d.detect(attempt)
                    attempt.detector_results[detector_probe_name] = list(result)

                    evaluations = [evaluator.test(float(score)) for score in result if score is not None]
                    # If at least one detector passes, attack succeeded
                    if any(evaluations):
                        # Add to exclusion set for next probe
                        successful_pairs.add((
                            attempt.notes["intent"],
                            attempt.notes["stub"]
                        ))

            # Mark attempts as complete and collect
            for attempt in attempts:
                attempt.status = ATTEMPT_COMPLETE
                all_attempts.append(attempt)

        # Write all attempts to report file
        for attempt in all_attempts:
            _config.transient.reportfile.write(
                json.dumps(attempt.as_dict(), ensure_ascii=False) + "\n"
            )

        self._end_run_hook()
        logging.info(f"Filterable early stopping harness completed: {len(successful_pairs)} successful pairs")
