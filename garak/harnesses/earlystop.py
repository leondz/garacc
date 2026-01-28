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

from garak.cas import Policy
from garak.harnesses import Harness
from garak import _config, _plugins, intentservice
from garak.attempt import ATTEMPT_COMPLETE, Attempt, ATTEMPT_STARTED
from garak.probes.base import IntentProbe


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
        self.policy = _load_policy_from_config(config_root)

    def create_attempt(self, stub) -> Attempt:
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

    def run(self, model, probe_names, detector_names, evaluator, buff_names=None):
        """
        Early stopping harness - loads probes by name, filters to compatible IntentProbes,
        then iterates attack methods until all intents succeed or methods exhausted.
        """
        if buff_names is None:
            buff_names = []

        self._load_buffs(buff_names)

        # Load and filter probes (like PxD but with compatibility check)
        attack_methods = []
        for probename in probe_names:
            try:
                probe = _plugins.load_plugin(probename)
            except Exception as e:
                logging.error(f"{probename} load exception, skipping: {e}")
                continue

            if not probe:
                logging.warning(f"{probename} load failed, skipping")
                continue

            # Check IntentProbe compatibility
            if not isinstance(probe, IntentProbe):
                logging.warning(f"{probename} is not an IntentProbe - skipping")
                continue

            # Check against compatible list
            short_name = probename.replace("probes.", "")
            if self.compatible_probes and short_name not in self.compatible_probes:
                logging.warning(f"{short_name} not in compatible_probes - skipping")
                continue

            attack_methods.append(probe)

        if not attack_methods:
            raise ValueError("No compatible IntentProbe attack methods loaded")

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

        # Derive intents from policy - test traits that are set to False (not permitted)
        intents = [
            trait_code for trait_code, permitted in self.policy.points.items()
            if permitted is False
        ]

        if not intents:
            raise ValueError("No intents to test - no policy points set to False")

        self._start_run_hook()

        # Generate initial payloads from intents

        # this is how we can override the location of the stub files for now
        # The path should point to a file structure as follows:
        # ├── intent_skip.json
        # └── intent_stubs
        #     ├── T999.txt
        #     └── T999test.txt
        # the sub files MUST end with .txt

        # intentservice.cas_data_path = Path("/Users/hjrnunes/Downloads/cas")

        all_intent_stubs = []
        for intent in intents:
            all_intent_stubs.extend(intentservice.get_intent_stubs(intent))

        if not all_intent_stubs:
            logging.warning("No intent stubs generated, nothing to test")
            self._end_run_hook()
            return

        # Convert conversations to attempts for the first round
        accepted_attempts = []
        rejected_attempts = [self.create_attempt(stub) for stub in all_intent_stubs]

        # Apply attack methods in sequence
        for attack_method in attack_methods:
            if not rejected_attempts:
                logging.info("No rejected attempts left, stopping early")
                break

            logging.info(f"Applying {attack_method.__class__.__name__} to {len(rejected_attempts)} rejected attempts")

            try:
                # Apply attack method to rejected attempts
                prompts = [attempt.notes["stub"] for attempt in rejected_attempts]
                attacked_attempts = list(attack_method.attack_target(prompts, model))
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
                logging.error(f"Attack method {attack_method.__class__.__name__} failed: {e}")
                # Continue with rejected attempts for next attack method
                continue

        # Update all attempts to completed status
        for attempt in accepted_attempts + rejected_attempts:
            attempt.status = ATTEMPT_COMPLETE
            _config.transient.reportfile.write(json.dumps(attempt.as_dict(), ensure_ascii=False) + "\n")

        self._end_run_hook()
        logging.info(
            f"Early stopping harness completed: {len(accepted_attempts)} accepted, {len(rejected_attempts)} rejected")
