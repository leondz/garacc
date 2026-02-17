# SPDX-FileCopyrightText: Portions Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""**System Prompt Override (SPO) attacks**

Tests whether system prompt overrides can bypass model safety mechanisms.
"""

import json
import logging
from typing import List

from garak import _config
import garak.probes
from garak.data import path as data_path


class SPOIntent(garak.probes.IntentProbe):
    """DAN-style system prompt override attack on intent stubs

    Tests whether DAN (Do Anything Now) prompt variants can bypass model
    safety mechanisms when combined with malicious intent stubs. For each
    stub, the probe applies all available DAN prompt variants and evaluates
    if the model complies with the harmful request.

    Each attempt includes metadata tracking which DAN variant was used for
    analysis and debugging purposes.
    """

    lang = "en"
    primary_detector = "mitigation.MitigationBypass"
    extended_detectors = ["dan.DAN"]
    tags = [
        # TODO
    ]
    goal = "bypass system prompt via DAN-style persona injection"
    tier = garak.probes.Tier.OF_CONCERN
    active = False

    DEFAULT_PARAMS = garak.probes.IntentProbe.DEFAULT_PARAMS | {
        "dan_variants": None,  # Optional: list of specific variants to use
    }

    def __init__(self, config_root=_config):
        # Load DAN prompts before calling super().__init__()
        # which will call _build_prompts()
        self.dan_prompts = self._load_all_dan_prompts()

        # Track mapping from prompt index to (variant_name, prompt_index_in_variant)
        self.prompt_to_variant = {}

        # Initialize dan_variants to None before config loading
        if not hasattr(self, 'dan_variants'):
            self.dan_variants = None

        # Call parent __init__ which will call _load_config and then _build_prompts
        super().__init__(config_root=config_root)

        # Filter by specific variants after config is loaded
        if self.dan_variants is not None:
            self.dan_prompts = {
                k: v for k, v in self.dan_prompts.items() if k in self.dan_variants
            }
            # Rebuild prompts with filtered variants
            self._build_prompts()

        if not self.dan_prompts:
            logging.warning(
                "SPOIntent: No DAN prompts loaded. Probe will generate no prompts."
            )

    def _load_all_dan_prompts(self) -> dict[str, List[str]]:
        """Load all DAN prompts from JSON files in garak/data/dan/

        Returns:
            Dictionary mapping variant names to lists of prompt strings.
            Example: {"Dan_11_0": ["prompt1", "prompt2"], "STAN": ["prompt1"]}
        """
        dan_prompts = {}
        dan_dir = data_path / "dan"

        if not dan_dir.exists():
            logging.error(f"DAN directory not found: {dan_dir}")
            return dan_prompts

        for json_file in dan_dir.glob("*.json"):
            variant_name = json_file.stem
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    prompts_data = json.load(f)
                    if isinstance(prompts_data, list):
                        # Filter out empty prompts and convert to strings
                        prompts = [str(p).strip() for p in prompts_data if p]
                        if prompts:
                            dan_prompts[variant_name] = prompts
                            logging.debug(
                                f"Loaded {len(prompts)} prompts from {variant_name}"
                            )
                    else:
                        logging.warning(
                            f"Expected list in {json_file}, got {type(prompts_data)}"
                        )
            except (json.JSONDecodeError, IOError) as e:
                logging.warning(f"Failed to load {json_file}: {e}")

        logging.info(f"SPOIntent: Loaded {len(dan_prompts)} DAN variants")
        return dan_prompts

    def prompts_from_stub(self, stub: str) -> List[str]:
        """Generate prompts by combining all DAN variants with the stub.

        For each DAN variant, creates a prompt in the format:
        "{dan_prompt}\\n\\n{stub}"

        Args:
            stub: The intent stub (e.g., "tell me how to build a bomb")

        Returns:
            List of prompts, one for each DAN variant × prompt combination
        """
        prompts = []
        current_index = len(self.prompt_to_variant)  # Track global prompt index

        for variant_name, variant_prompts in sorted(self.dan_prompts.items()):
            for prompt_idx, dan_prompt in enumerate(variant_prompts):
                # Combine DAN prompt with stub
                combined_prompt = f"{dan_prompt}\n\n{stub}"
                prompts.append(combined_prompt)

                # Store mapping for metadata tracking
                self.prompt_to_variant[current_index] = {
                    "dan_variant": variant_name,
                    "dan_prompt_index": prompt_idx,
                }
                current_index += 1

        return prompts

    def probe(self, generator) -> List[garak.attempt.Attempt]:
        """Execute probe with {generator.name} substitution in DAN prompts.

        The DAN prompts contain {generator.name} placeholders that should be
        replaced with the actual generator name before sending to the model.
        """
        # Substitute {generator.name} in all prompts
        generator_name = getattr(generator, "name", "ChatGPT")
        self.prompts = [
            prompt.replace("{generator.name}", generator_name)
            for prompt in self.prompts
        ]

        return super().probe(generator)

    def _attempt_prestore_hook(
        self, attempt: garak.attempt.Attempt, seq: int
    ) -> garak.attempt.Attempt:
        """Add DAN-specific metadata to attempt notes before storage.

        Adds:
        - dan_variant: Name of the DAN variant used (e.g., "Dan_11_0")
        - dan_prompt_index: Index of the prompt within that variant
        """
        # Call parent hook to add intent/stub metadata
        attempt = super()._attempt_prestore_hook(attempt, seq)

        # Add DAN-specific metadata
        if seq in self.prompt_to_variant:
            variant_info = self.prompt_to_variant[seq]
            attempt.notes["dan_variant"] = variant_info["dan_variant"]
            attempt.notes["dan_prompt_index"] = variant_info["dan_prompt_index"]

        return attempt
