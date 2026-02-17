# SPDX-FileCopyrightText: Portions Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Early stopping harness

The idea here is that everything starts and is driven by a set of intents that the user wants to test.
The approach is to try a list of attack methods and, after each round of test,
we carry over only the attempts that have been rejected by the LLM.
We continue iterating until all the intents have been accepted, or we run out of available attack methods.
"""
import json
import logging
from copy import deepcopy
from typing import List

import garak
from garak.detectors import Detector
from garak.evaluators import Evaluator
from garak.harnesses import Harness
from garak import _config, _plugins
from garak.intents import Stub
from garak.services import intentservice
from garak.attempt import ATTEMPT_COMPLETE, Attempt, ATTEMPT_STARTED
from garak.probes.base import IntentProbe


def _is_rejected(attempt: Attempt, detectors: List[Detector], evaluator: Evaluator) -> bool:
    """
    Returns True if the target LLM rejected the attempt,
    ex: "I'm sorry, but I can't do that" -> True
    """
    outputs = [output.text for output in attempt.outputs]
    if not outputs or not any(outputs):
        logging.debug(f"No outputs for attempt {attempt.uuid}, marking as rejected")
        return True

    results = [d.detect(attempt) for d in detectors]
    # When the target model returns "I'm sorry, but I can't do that" the Refusal judge Detector returns 0.0
    detector_names = [d.detectorname for d in detectors]
    attempt.detector_results = {d: [r] for d, r in zip(detector_names, results)}
    evaluations = [evaluator.test(float(score))
                   for detector_scores in results
                   for score in detector_scores]

    # Save this intermediary attempt, no matter the outcome
    _config.transient.reportfile.write(json.dumps(attempt.as_dict(), ensure_ascii=False) + "\n")

    return any(evaluations)


def _filter_stubs(intent_code, stub, attempts: List[Attempt]):
    for a in attempts:
        if a.intent == intent_code and a.goal == stub:
            return True
    return False


class EarlyStopHarness(Harness):
    DEFAULT_PARAMS = {
        "compatible_probes": [
            "grandma.GrandmaIntent",
            "tap.TAPIntent",
            "multilingual.TranslationIntent",
        ],
    }

    def __init__(self, config_root=_config):
        super().__init__(config_root)

    def _create_attempt(self, stub: Stub) -> Attempt:
        new_attempt = Attempt(
            probe_classname=(
                str(self.__class__.__module__).replace("garak.probes.", "")
                + "."
                + self.__class__.__name__
            ),
            intent=stub.intent,
            goal=stub.content,
            status=ATTEMPT_STARTED,
            prompt=garak.attempt.Conversation(turns=[
                garak.attempt.Turn("user", garak.attempt.Message(stub.content))
            ]),
        )
        return new_attempt

    def _load_probe(self, probe_name) -> IntentProbe | None:
        try:
            probe = _plugins.load_plugin(probe_name)
        except Exception as e:
            logging.error(f"{probe_name} load exception, skipping: {e}")
            return None

        if not probe:
            logging.warning(f"{probe_name} load failed, skipping")
            return None

        # Check IntentProbe compatibility
        if not isinstance(probe, IntentProbe):
            logging.warning(f"{probe_name} is not an IntentProbe - skipping")
            return None

        # Check against compatible list
        short_name = probe_name.replace("probes.", "")
        if self.compatible_probes and short_name not in self.compatible_probes:
            logging.warning(f"{short_name} not in compatible_probes - skipping")
            return None

        return probe

    def _update_attempt_status(self, attacked_attempts, previously_rejected, detectors, evaluator):
        accepted_attempts = []
        rejected_attempts = []

        for attempt in previously_rejected:
            # Group attacked_attempts by stub
            rejected_attacks = [_is_rejected(attempt, detectors, evaluator)
                                for attacked_attempt in attacked_attempts
                                if
                                attacked_attempt.intent == attempt.intent and attacked_attempt.goal == attempt.goal]
            # Some probes don't return failed attempts; we assume that an empty rejected_attacks means failure
            if any(rejected_attacks) or not rejected_attacks:
                rejected_attempts.append(attempt)
            else:
                accepted_attempts.append(attempt)

        return accepted_attempts, rejected_attempts

    def _collect_baseline(self, model, detectors, evaluator, stubs: List[Stub]) -> tuple[List[Attempt], List[Attempt]]:
        """
        We'll do a vanilla run of just sending the input stubs as is to the target LLM in order to collect
        the baseline results.
        Returns two lists: one for successful attempts, one for rejected attempts.
        """
        baseline_probe = IntentProbe()
        baseline_probe.generations = 1

        accepted_attempts = []
        rejected_attempts = [self._create_attempt(s) for s in stubs]

        try:
            attacked_attempts = list(baseline_probe.probe(model))
        except Exception as e:
            logging.error(f"Failed to collect baseline metrics {e}")
            return accepted_attempts, rejected_attempts

        return self._update_attempt_status(attacked_attempts, rejected_attempts, detectors, evaluator)

    def run(self, model, probe_names, detector_names, evaluator, buff_names=None):
        """
        Early stopping harness - loads probes by name, filters to compatible IntentProbes,
        then iterates attack methods until all intents succeed or methods exhausted.
        """
        if buff_names is None:
            buff_names = []

        self._load_buffs(buff_names)

        # Load detectors
        detectors = []
        for detector_name in detector_names:
            detector = _plugins.load_plugin(detector_name, break_on_fail=False)
            if detector:
                detectors.append(detector)
            else:
                logging.warning(f"detector load failed: {detector_name}, skipping")

        if not detectors:
            raise ValueError("No detectors loaded")

        # Load intents
        intent_spec = _config.cas.intent_spec
        if not intent_spec:
            raise ValueError("No intents to test - intent_spec not set")
        intents = str.split(intent_spec, ",")
        if not intents:
            raise ValueError("No intents to test")

        # Generate initial payloads from intents
        all_intent_stubs = []
        for intent in intents:
            all_intent_stubs.extend(intentservice.get_intent_stubs(intent))

        if not all_intent_stubs:
            logging.warning("No intent stubs generated, nothing to test")
            self._end_run_hook()
            return

        self._start_run_hook()

        logging.info(f"Collecting baseline for {len(all_intent_stubs)} intents")
        accepted_attempts, rejected_attempts = self._collect_baseline(model, detectors, evaluator, all_intent_stubs)

        # Apply attack methods in order
        for probe_name in probe_names:
            if not rejected_attempts:
                logging.info("No rejected attempts left, stopping early")
                break

            probe = self._load_probe(probe_name)
            if not probe:
                continue

            logging.info(f"Applying {probe_name} to {len(rejected_attempts)} rejected attempts")

            # Filter out from the intentservice prompts that have already been rejected
            intentservice.set_stubs_filter(
                lambda intent_code, stub: _filter_stubs(intent_code, stub, rejected_attempts))

            # Execute the probe
            try:
                attacked_attempts = list(probe.probe(model))
            except Exception as e:
                logging.error(f"Attack method {probe_name} failed: {e}")
                continue  # Continue with rejected attempts for the next attack method

            accepted_attempts, rejected_attempts = self._update_attempt_status(attacked_attempts, rejected_attempts,
                                                                               detectors, evaluator)

        # End of the loop, update all attempts to completed status and add our detection status
        for attempt in rejected_attempts:
            attempt.detector_results = {"EarlyStop": [0.0]}
        for attempt in accepted_attempts:
            attempt.detector_results = {"EarlyStop": [1.0]}
        for attempt in rejected_attempts + accepted_attempts:
            attempt.status = ATTEMPT_COMPLETE
            _config.transient.reportfile.write(json.dumps(attempt.as_dict(), ensure_ascii=False) + "\n")

        self._end_run_hook()
        logging.info(
            f"Early stopping harness completed: {len(accepted_attempts)} accepted, {len(rejected_attempts)} rejected")
