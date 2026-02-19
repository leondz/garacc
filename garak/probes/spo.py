# SPDX-FileCopyrightText: Portions Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""**System Prompt Override (SPO) attacks**

Tests whether system prompt overrides can bypass model safety mechanisms.
"""

import json
import logging
import random
from typing import List, Tuple

from garak import _config
import garak.probes
from garak.data import path as data_path
from garak.intents import Stub


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
        "max_dan_samples": 10,  # Maximum number of DAN prompts to sample
        "dan_sample_seed": None,  # Random seed for DAN sampling (for reproducibility)
    }

    def __init__(self, config_root=_config):
        # Load DAN prompts before calling super().__init__()
        # which will call build_prompts()
        self.dan_prompts = self._load_all_dan_prompts()

        # Track mapping from prompt index to (variant_name, prompt_index_in_variant)
        self.prompt_to_variant = {}

        # Initialize parameters before config loading
        if not hasattr(self, "dan_variants"):
            self.dan_variants = None
        if not hasattr(self, "max_dan_samples"):
            self.max_dan_samples = 10
        if not hasattr(self, "dan_sample_seed"):
            self.dan_sample_seed = None

        # RNG for sampling - initialized once to ensure different samples per stub
        self._sampling_rng = None

        # Call parent __init__ which will call _load_config and then build_prompts
        super().__init__(config_root=config_root)

        # Filter by specific variants after config is loaded
        if self.dan_variants is not None:
            self.dan_prompts = {
                k: v for k, v in self.dan_prompts.items() if k in self.dan_variants
            }
            # Rebuild prompts with filtered variants
            self.build_prompts()

        if not self.dan_prompts:
            logging.warning(
                "SPOIntent: No DAN prompts loaded. Probe will generate no prompts."
            )

    def build_prompts(self):
        """Build prompts, resetting sampling RNG for reproducibility."""
        # Reset RNG so each build_prompts call starts fresh
        self._sampling_rng = None
        # Reset prompt_to_variant mapping
        self.prompt_to_variant = {}
        super().build_prompts()

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

    def _get_sampling_rng(self) -> random.Random:
        """Get or create the RNG for sampling DAN prompts.

        The RNG is created once per build_prompts call to ensure:
        - Different stubs get different samples (RNG state advances)
        - Reproducibility when dan_sample_seed is set
        """
        if self._sampling_rng is None:
            self._sampling_rng = random.Random(self.dan_sample_seed)
        return self._sampling_rng

    def _sample_dan_items(self) -> List[dict]:
        """Flatten and sample DAN prompts up to max_dan_samples.

        Returns:
            List of dicts with keys: dan_prompt, variant_name, prompt_idx
        """
        all_dan_items = []
        for variant_name, variant_prompts in self.dan_prompts.items():
            for prompt_idx, dan_prompt in enumerate(variant_prompts):
                all_dan_items.append({
                    "dan_prompt": dan_prompt,
                    "variant_name": variant_name,
                    "prompt_idx": prompt_idx,
                })

        if self.max_dan_samples and len(all_dan_items) > self.max_dan_samples:
            rng = self._get_sampling_rng()
            return rng.sample(all_dan_items, self.max_dan_samples)
        return all_dan_items

    def _prompts_from_stub(self, stub: Stub) -> List[str]:
        """Generate prompts by combining sampled DAN variants with the stub.

        For each DAN variant, creates a prompt in the format:
        "{dan_prompt}\\n\\n{stub}"

        DAN prompts are randomly sampled up to max_dan_samples per stub.
        If dan_sample_seed is set, the sampling is reproducible.

        Args:
            stub: The intent stub (e.g., "tell me how to build a bomb")

        Returns:
            List of prompts, one for each sampled DAN variant × prompt combination
        """
        sampled_items = self._sample_dan_items()

        prompts = []
        current_index = len(self.prompt_to_variant)

        for item in sampled_items:
            combined_prompt = f"{item['dan_prompt']}\n\n{stub.content}"
            prompts.append(combined_prompt)

            self.prompt_to_variant[current_index] = {
                "dan_variant": item["variant_name"],
                "dan_prompt_index": item["prompt_idx"],
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


class _SPOIntentAugmentedBase(SPOIntent):
    """Base class for augmented SPO probes.

    Subclasses should set augment_system and augment_user class attributes
    to control which parts of the prompt receive augmentation.
    """

    lang = "en"
    primary_detector = "mitigation.MitigationBypass"
    extended_detectors = ["dan.DAN"]
    tier = garak.probes.Tier.OF_CONCERN
    active = False

    # Subclasses must override these
    augment_system = False
    augment_user = False

    def __init__(self, config_root=_config):
        super().__init__(config_root)
        self.prompts_meta = []

    def _get_prompts(self, apply_to_system, apply_to_user, stub: str) -> Tuple[List[str], List[dict]]:
        from garak.probes._augmentation import get_random_augmentation

        sampled_items = self._sample_dan_items()

        prompts = []
        metadata = []

        for item in sampled_items:
            dan_prompt = item["dan_prompt"]
            variant_name = item["variant_name"]
            prompt_idx = item["prompt_idx"]

            augmentation_func = get_random_augmentation()
            if apply_to_system:
                dan_prompt = augmentation_func(dan_prompt)
            if apply_to_user:
                stub_augmented = augmentation_func(stub)
            else:
                stub_augmented = stub

            combined_prompt = f"{dan_prompt}\n\n{stub_augmented}"
            prompts.append(combined_prompt)

            metadata.append({
                "dan_variant": variant_name,
                "dan_prompt_index": prompt_idx,
                "augmentation": augmentation_func.__name__,
                "augmentation_target": "both" if apply_to_system and apply_to_user else "system" if apply_to_system else "user",
            })

        return prompts, metadata

    def _prompts_from_stub(self, stub: Stub) -> List[str]:
        """Generate prompts with augmentation applied according to class attributes"""
        prompts, metadata = self._get_prompts(self.augment_system, self.augment_user, stub.content)

        # Track current index for metadata
        current_index = len(self.prompt_to_variant)
        for i, meta in enumerate(metadata):
            # Store mapping for metadata tracking
            self.prompt_to_variant[current_index + i] = {
                "dan_variant": meta["dan_variant"],
                "dan_prompt_index": meta["dan_prompt_index"],
                "augmentation": meta["augmentation"],
                "augmentation_target": meta["augmentation_target"],
            }

        return prompts

    def _attempt_prestore_hook(
        self, attempt: garak.attempt.Attempt, seq: int
    ) -> garak.attempt.Attempt:
        """Add augmentation metadata to attempt notes"""
        attempt = super()._attempt_prestore_hook(attempt, seq)

        # Add augmentation-specific metadata if available
        if seq in self.prompt_to_variant:
            variant_info = self.prompt_to_variant[seq]
            if "augmentation" in variant_info:
                attempt.notes["augmentation"] = variant_info["augmentation"]
                attempt.notes["augmentation_target"] = variant_info["augmentation_target"]

        return attempt


class SPOIntentUserAugmented(_SPOIntentAugmentedBase):
    """SPO with augmentation applied to user messages (stub)

    Applies text augmentation transformations (word scrambling, random
    capitalization, or ASCII noise) to the stub portion of the prompt.
    The augmentation function is selected randomly for each prompt.
    """

    goal = "bypass system prompt via augmented DAN-style persona injection (user augmented)"
    augment_system = False
    augment_user = True


class SPOIntentSystemAugmented(_SPOIntentAugmentedBase):
    """SPO with augmentation applied to system messages (DAN prompt)

    Applies text augmentation transformations (word scrambling, random
    capitalization, or ASCII noise) to the DAN prompt portion.
    The augmentation function is selected randomly for each prompt.
    """

    goal = "bypass system prompt via augmented DAN-style persona injection (system augmented)"
    augment_system = True
    augment_user = False


class SPOIntentBothAugmented(_SPOIntentAugmentedBase):
    """SPO with augmentation applied to both system and user messages

    Applies text augmentation transformations (word scrambling, random
    capitalization, or ASCII noise) to both the DAN prompt and stub.
    The augmentation function is selected randomly for each prompt.
    """

    goal = "bypass system prompt via augmented DAN-style persona injection (both augmented)"
    augment_system = True
    augment_user = True
