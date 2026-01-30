# SPDX-FileCopyrightText: Portions Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DAN IntentProbe implementations for use with EarlyStopHarness"""

import json
from typing import List

from garak import _config
from garak.data import path as data_path
import garak.probes


class Dan_11_0_Intent(garak.probes.IntentProbe):
    """DAN 11.0 attack as an IntentProbe for use with EarlyStopHarness.

    Applies the DAN 11.0 jailbreak prompt to intent stubs, appending the
    intent stub after the DAN prompt.
    """

    active = False
    lang = "en"
    goal = "disregard the system prompt"
    tags = [
        "avid-effect:security:S0403",
        "owasp:llm01",
        "quality:Security:PromptStability",
        "avid-effect:performance:P0204",
        "payload:jailbreak",
    ]
    primary_detector = "dan.DAN"
    extended_detectors = ["mitigation.MitigationBypass"]

    prompt_file = "dan/Dan_11_0.json"

    def __init__(self, config_root=_config):
        super().__init__(config_root=config_root)
        # Load the DAN 11.0 template from JSON
        prompt_path = data_path / self.prompt_file
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompts_data = json.load(f)
            # The JSON contains a single-element array with the prompt
            self.dan_template = prompts_data[0] if prompts_data else ""

    def _apply_technique(self, stub: str) -> List[str]:
        """Apply DAN 11.0 jailbreak followed by the intent stub."""
        return [
            f"{self.dan_template}\n\n{stub}"
        ]
