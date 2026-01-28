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
from typing import List

import tqdm

from garak.detectors import Detector
from garak.evaluators import Evaluator
from garak.generators import Generator
from garak.harnesses import Harness
from garak import _config, intentservice
from garak.attempt import ATTEMPT_COMPLETE, Attempt, ATTEMPT_STARTED
from garak.probes import IntentProbe


class EarlyStopHarness(Harness):

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

    def run(self,
            model: Generator,
            intents: List[str],
            attack_methods: List[IntentProbe],
            detectors: List[Detector],
            evaluator: Evaluator,
            buff_names=None):
        """
        First, load the list of prompts that we want to test via the IntentService
        Then, iterate over the attack_methods and pass the prompts that have been rejected by the LLM to them.
        The process finishes when we don't have any more attack methods or all the conversations have been accepted.
        """

        if not attack_methods:
            msg = "No attack methods, nothing to do"
            logging.warning(msg)
            if hasattr(_config.system, "verbose") and _config.system.verbose >= 2:
                print(msg)
            raise ValueError(msg)

        if not buff_names:
            buff_names = []

        self._load_buffs(buff_names)  # TODO: buffs? Is this enough or should we do something more?
        self._start_run_hook()

        # Generate initial payloads from all payload generators
        all_intent_stubs = []
        for intent in intents:
            all_intent_stubs.extend(intentservice.get_intent_stubs(intent))

        if not all_intent_stubs:
            logging.warning("No conversations generated, nothing to test")
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
