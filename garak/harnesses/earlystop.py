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

import tqdm

from garak.harnesses import Harness
from garak import _config, _plugins, intentservice
from garak.attempt import ATTEMPT_COMPLETE, Attempt, ATTEMPT_STARTED
from garak.probes.base import IntentProbe


class EarlyStopHarness(Harness):
    DEFAULT_PARAMS = {
        "compatible_probes": [
            "grandma.GrandmaIntent",
            "tap.TAPIntent",
            "dan_intent.Dan_11_0_Intent",
        ],
    }

    def __init__(self, config_root=_config):
        super().__init__(config_root)

    def _create_attempt(self, stub) -> Attempt:
        new_attempt = Attempt(
            probe_classname=(
                    str(self.__class__.__module__).replace("garak.probes.", "")
                    + "."
                    + self.__class__.__name__
            ),
            notes={"stub": stub},
            goal=stub,
            status=ATTEMPT_STARTED,
            prompt=stub,
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

    def run(self, model, probe_names, detector_names, evaluator, buff_names=None):
        """
        Early stopping harness - loads probes by name, filters to compatible IntentProbes,
        then iterates attack methods until all intents succeed or methods exhausted.
        """
        if buff_names is None:
            buff_names = []

        self._load_buffs(buff_names)

        # Load detectors (like PxD)
        detectors = []
        for detector_name in detector_names:
            detector = _plugins.load_plugin(detector_name, break_on_fail=False)
            if detector:
                detectors.append(detector)
            else:
                logging.warning(f"detector load failed: {detector_name}, skipping")

        if not detectors:
            raise ValueError("No detectors loaded")

        intent_spec = _config.cas.intent_spec
        if not intent_spec:
            raise ValueError("No intents to test - intent_spec not set")
        intents = str.split(intent_spec, ",")
        if not intents:
            raise ValueError("No intents to test - no policy points set to False")

        self._start_run_hook()

        # Generate initial payloads from intents
        all_intent_stubs = []
        for intent in intents:
            all_intent_stubs.extend(intentservice.get_intent_stubs(intent))

        if not all_intent_stubs:
            logging.warning("No intent stubs generated, nothing to test")
            self._end_run_hook()
            return

        # Convert conversations to attempts for the first round
        accepted_attempts = []
        rejected_attempts = [self._create_attempt(stub) for stub in all_intent_stubs]

        # Apply attack methods in sequence
        for probe_name in probe_names:
            if not rejected_attempts:
                logging.info("No rejected attempts left, stopping early")
                break

            probe = self._load_probe(probe_name)
            if not probe:
                continue

            logging.info(f"Applying {probe_name} to {len(rejected_attempts)} rejected attempts")

            try:
                # Filter out from the intentservice prompts that have already been rejected
                prompts = [attempt.notes["stub"] for attempt in rejected_attempts]
                intentservice.set_stubs_filter(
                    lambda intent_code, stub: stub in prompts)  # TODO: should we check intent_code?

                attacked_attempts = list(probe.probe(model))
                # TODO: some methods might not return the internal attempts.
                #       For example: TAP returns only prompts that have been classified as jailbroken
                rejected_attempts = []

                # Process the results
                for d in detectors:
                    logging.debug("harness: run detector %s", d.detectorname)
                    attempt_iterator = tqdm.tqdm(attacked_attempts, leave=False)
                    detector_probe_name = d.detectorname.replace("garak.detectors.", "")
                    attempt_iterator.set_description("detectors." + detector_probe_name)
                    # TODO: group by attempt.notes["stub"] (we might have multiple attempts for a given input stub)
                    for attempt in attempt_iterator:
                        if d.skip:
                            continue
                        result = d.detect(attempt)
                        attempt.detector_results[detector_probe_name] = list(result)

                        evaluations = [evaluator.test(float(score)) for score in result if score is not None]
                        # If at least one detector passes, we accept the attempt
                        if any(evaluations):
                            accepted_attempts.append(attempt)
                        else:
                            rejected_attempts.append(attempt)

            except Exception as e:
                logging.error(f"Attack method {probe_name} failed: {e}")
                # Continue with rejected attempts for next attack method
                continue

        # Update all attempts to completed status
        for attempt in accepted_attempts + rejected_attempts:
            attempt.status = ATTEMPT_COMPLETE
            _config.transient.reportfile.write(json.dumps(attempt.as_dict(), ensure_ascii=False) + "\n")

        self._end_run_hook()
        logging.info(
            f"Early stopping harness completed: {len(accepted_attempts)} accepted, {len(rejected_attempts)} rejected")
