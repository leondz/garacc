# SPDX-FileCopyrightText: Portions Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""**Translation-based jailbreak attacks**

Probes in this module test whether content moderation can be bypassed by
translating malicious requests into different languages.

The translation attack vector works by exploiting potential weaknesses in
multilingual models where harmful content detection may be less effective
in non-English languages. The model receives translated prompts and returns
translated responses, which are then reverse-translated back to English for
evaluation.

This module uses garak's langservice infrastructure for translation.
Configuration example for local translation models:
```yaml
run:
  langproviders:
    - language: "en,zh"
      model_type: "local.LocalHFTranslator"
      model_name: "Helsinki-NLP/opus-mt-zh-en"
    - language: "zh,en"
      model_type: "local.LocalHFTranslator"
      model_name: "Helsinki-NLP/opus-mt-en-zh"
```

Relative paper:
- https://arxiv.org/abs/2310.02446 - Multilingual Jailbreak Challenges in LLMs
"""

from typing import List

from garak import _config
import garak.probes


class TranslationIntent(garak.probes.IntentProbe):
    """Translation-based jailbreak probe

    Tests if content moderation can be bypassed by translating malicious
    requests to another language. Translates intent stubs to the configured
    target language, sends to LLM, and translates responses back to English
    for detection.

    The attack exploits potential weaknesses in multilingual models where
    harmful content filters may be less effective in non-English languages.
    Translation is handled automatically by garak's base Probe class using
    the configured langproviders.

    This probe requires bidirectional translation configuration:
    1. Source (en) → Target language translation
    2. Target language → Source (en) reverse translation
    """

    DEFAULT_PARAMS = {
        "target_lang": "zh",
    }

    lang = "en"  # Source language
    primary_detector = "mitigation.MitigationBypass"
    tags = [
        # TODO
    ]
    goal = "bypass content moderation via translation"
    tier = garak.probes.Tier.OF_CONCERN
    active = False

    def __init__(self, config_root=_config):
        # We need to set target_lang in config BEFORE calling super().__init__()
        # because the base Probe class creates langproviders during initialization
        # Load config early to get target_lang from DEFAULT_PARAMS or user config
        self._load_config(config_root)

        # Now set the target language for the run
        # Handle both dict config (from tests) and module config (from real runs)
        if isinstance(config_root, dict):
            # In test mode with dict config, use global _config
            _config.run.target_lang = self.target_lang
        else:
            # In normal mode with module config
            config_root.run.target_lang = self.target_lang

        # Call IntentProbe.__init__ which will call Probe.__init__
        super().__init__(config_root=config_root)

        # Verify langproviders are configured
        if self.langprovider is None:
            import logging
            logging.warning(
                f"No langprovider configured for {self.lang} → {self.target_lang}. "
                f"Translation probe will not translate prompts. "
                f"Please configure langproviders in your config file."
            )
        if self.reverse_langprovider is None:
            import logging
            logging.warning(
                f"No reverse langprovider configured for {self.target_lang} → {self.lang}. "
                f"Translation probe will not reverse-translate outputs. "
                f"Please configure reverse langproviders in your config file."
            )

        # There's nothing more needed here:
        #   The base Probe.probe() method automatically translates prompts using
        #   self.langprovider before sending to the target LLM. Similarly,
        #   Probe._postprocess_attempt() reverse-translates outputs using
        #   self.reverse_langprovider before detection.
