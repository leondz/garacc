# SPDX-FileCopyrightText: Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Migrate the deprecated selection keys onto the unified ``run.spec`` grammar."""

import copy

from garak.resources.fixer import Migration
from garak._spec import legacy_selection_spec


class MapLegacySelectionToSpec(Migration):
    def apply(config_dict: dict) -> dict:
        """Convert ``plugins.probe_spec``/``buff_spec`` and ``run.probe_tags`` into ``run.spec``."""
        cfg = copy.deepcopy(config_dict)
        plugins = cfg.get("plugins", {})
        run = cfg.get("run", {})
        spec = legacy_selection_spec(
            plugins.get("probe_spec"),
            plugins.get("buff_spec"),
            run.get("probe_tags"),
        )
        if spec is None:
            return cfg
        plugins.pop("probe_spec", None)
        plugins.pop("buff_spec", None)
        run.pop("probe_tags", None)
        # an explicitly-set run.spec wins; otherwise adopt the converted spec
        if not run.get("spec"):
            run["spec"] = spec
        cfg["run"] = run
        return cfg
