# SPDX-FileCopyrightText: Portions Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified ``run.spec`` selection grammar.

A single internal :class:`Spec` (a list of :class:`Selector` with explicit
polarity) is produced by two transports that parse to the same object:

* CLI string (comma separated), via :func:`parse_spec_string`
* config file form (YAML/JSON ``include``/``exclude`` lists), via
  :func:`parse_spec_file`

:meth:`Spec.resolve` orchestrates probe and buff selection plus the ``tier``
and ``tag`` filters. The single plugin-path resolution core is
:func:`_resolve_plugin_paths` (category generic), shared by :meth:`Spec.resolve`
and by the ``parse_plugin_spec`` adapter used for detectors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Plugin categories selectable via run.spec. Detectors are not selectable here
# yet; they keep their own legacy spec surface (parse_plugin_spec).
_CATEGORIES = ("probes", "buffs")

# Tier assigned to probes that do not declare one (Tier.UNLISTED).
_DEFAULT_TIER = 9


@dataclass(frozen=True)
class Selector:
    """A single selection clause.

    ``kind`` is one of ``"plugin_path"``, ``"tag"`` or ``"tier"``. For
    ``plugin_path`` selectors ``value`` carries the category-prefixed path
    (e.g. ``"probes.dan"``) and ``category`` is set; for ``tag``/``tier`` the
    filter applies to probes and ``category`` is ``None``.
    """

    kind: str
    value: str
    include: bool = True
    category: Optional[str] = None


@dataclass
class Spec:
    """Internal representation of a unified selection spec."""

    include: List[Selector] = field(default_factory=list)
    exclude: List[Selector] = field(default_factory=list)

    def to_file_dict(self) -> dict:
        """Serialise to the config-file form (``include``/``exclude`` lists)."""

        def _token(selector: Selector):
            if selector.kind == "plugin_path":
                return selector.value
            return {selector.kind: selector.value}

        return {
            "include": [_token(s) for s in self.include],
            "exclude": [_token(s) for s in self.exclude],
        }

    def resolve(self, skip_unknown: bool = False) -> "Resolution":
        """Resolve the spec to concrete probe and buff name lists."""
        return _resolve_spec(self, skip_unknown)


@dataclass
class Resolution:
    """Resolved selection. ``probes``/``buffs`` are canonical
    ``category.module.Class`` names; ``rejected`` lists unknown selectors;
    ``empty_reason`` is set when the spec resolves to no probes."""

    probes: List[str]
    buffs: List[str]
    rejected: List[str]
    empty_reason: Optional[str] = None


def parse_spec_string(spec: str) -> Spec:
    """Parse the CLI string transport into a :class:`Spec`.

    Selectors are comma separated; a leading ``-`` excludes, a leading ``+``
    (or no prefix) includes. Blank clauses and surrounding whitespace are
    tolerated. Deduplication is implicit at resolve time (set based).
    """
    out = Spec()
    for raw in (clause.strip() for clause in (spec or "").split(",")):
        if not raw:
            continue
        include = not raw.startswith("-")
        token = raw.lstrip("+-").strip()
        if not token:
            continue
        (out.include if include else out.exclude).append(_classify(token, include))
    return out


def parse_spec_file(node: Optional[dict]) -> Spec:
    """Parse the config-file transport (``{"include": [...], "exclude": [...]}``).

    List items are either plugin-path strings (``"probes.dan"``) or single-key
    mappings for filters (``{"tag": "owasp:llm01"}``, ``{"tier": 1}``).
    """
    out = Spec()
    if not node:
        return out
    for polarity, bucket in (("include", out.include), ("exclude", out.exclude)):
        for item in node.get(polarity) or []:
            if not isinstance(item, str):
                if not isinstance(item, dict) or len(item) != 1:
                    raise ValueError(
                        f"run.spec item must be a string or single-key mapping: {item!r}"
                    )
                (key, value), = item.items()
                item = f"{key}:{value}"
            bucket.append(_classify(item, polarity == "include"))
    return out


def _classify(token: str, include: bool) -> Selector:
    """Turn a bare token (polarity already stripped) into a :class:`Selector`."""
    if token.startswith("tag:"):
        return Selector("tag", token[4:], include, None)
    if token.startswith("tier:"):
        return Selector("tier", _normalize_tier(token[5:]), include, None)
    category = token.split(".", 1)[0]
    if category not in _CATEGORIES:
        raise ValueError(
            f"run.spec selector {token!r}: category prefix must be one of "
            f"{_CATEGORIES}"
        )
    return Selector("plugin_path", token, include, category)


def _normalize_tier(value: str) -> str:
    """Normalise a tier value (int or Tier name) to its int as a string."""
    from garak.probes._tier import Tier

    raw = value.strip()
    try:
        number = int(raw)
    except ValueError:
        try:
            return str(Tier[raw.upper()].value)
        except KeyError as exc:
            raise ValueError(
                f"unknown tier {value!r}; use an int (1..3, 9) or a Tier name"
            ) from exc
    try:
        Tier(number)
    except ValueError as exc:
        raise ValueError(
            f"invalid tier {value!r}; use an int (1..3, 9) or a Tier name"
        ) from exc
    return str(number)


def _legacy_path_selectors(spec: Optional[str], category: str) -> List[Selector]:
    """Translate a legacy spec string (unprefixed ``dan`` / ``dan.AutoDAN`` /
    ``all`` / ``auto`` / ``none``) into plugin-path :class:`Selector` objects."""
    if spec is None or str(spec).lower() in ("", "auto", "none"):
        return []
    if str(spec).lower() in ("all", "*"):
        return [Selector("plugin_path", f"{category}.*", True, category)]
    selectors = []
    for clause in str(spec).split(","):
        clause = clause.strip()
        if not clause:
            continue
        selectors.append(Selector("plugin_path", f"{category}.{clause}", True, category))
    return selectors


def _resolve_plugin_paths(
    selectors: List[Selector], category: str
) -> Tuple[set, List[str]]:
    """Single plugin-path resolution core (category generic).

    Mirrors the legacy ``parse_plugin_spec`` name resolution:
    ``<category>.*`` -> active plugins; ``<category>.<module>`` -> active
    family; ``<category>.<module>.<Class>`` -> exact match (ignores ``active``).
    Returns the resolved name set plus the list of unknown selector values.
    """
    from garak._plugins import enumerate_plugins

    enumerated = enumerate_plugins(category=category)
    names: set = set()
    rejected: List[str] = []
    prefix = f"{category}."
    for selector in selectors:
        token = selector.value
        body = token[len(prefix):] if token.startswith(prefix) else token
        if body == "*":
            names |= {p for p, active in enumerated if active is True}
        elif body.count(".") < 1:
            found = {
                p
                for p, active in enumerated
                if p.startswith(f"{category}.{body}.") and active is True
            }
            if found:
                names |= found
            else:
                rejected.append(token)
        else:
            found = {p for p, _ in enumerated if p == f"{category}.{body}"}
            if found:
                names |= found
            else:
                rejected.append(token)
    return names, rejected


def _has_any_tag(name: str, prefixes: List[str]) -> bool:
    from garak._plugins import plugin_info

    tags = plugin_info(name).get("tags") or []
    return any(tag.startswith(prefix) for tag in tags for prefix in prefixes)


def _tier_of(name: str) -> int:
    from garak._plugins import plugin_info

    return int(plugin_info(name).get("tier", _DEFAULT_TIER))


def _empty_reason(spec: "Spec") -> str:
    """Best-effort explanation of why a spec resolved to no probes."""
    tier_ceilings = [int(s.value) for s in spec.include if s.kind == "tier"]
    explicit = [
        s.value
        for s in spec.include
        if s.kind == "plugin_path"
        and s.category == "probes"
        and s.value.count(".") >= 2
    ]
    if tier_ceilings and explicit:
        ceiling = max(tier_ceilings)
        name = explicit[0]
        return (
            f"probe '{name}' is tier {_tier_of(name)} but the spec restricts to "
            f"tiers 1..{ceiling}; widen the tier filter or drop the explicit probe"
        )
    if any(s.kind in ("tag", "tier") for s in spec.include):
        return "no active probe matches the given tier/tag filters; widen the filters"
    return "every included probe was removed by an exclusion; adjust includes/excludes"


def _resolve_spec(spec: "Spec", skip_unknown: bool) -> "Resolution":
    from garak._plugins import enumerate_plugins

    rejected: List[str] = []

    # Layer 1: probe candidate set from plugin-path includes; default probes.*
    probe_includes = [
        s for s in spec.include if s.kind == "plugin_path" and s.category == "probes"
    ]
    if probe_includes:
        candidate, rej = _resolve_plugin_paths(probe_includes, "probes")
        rejected += rej
    else:
        candidate = {
            p for p, active in enumerate_plugins(category="probes") if active is True
        }

    # Layer 2: positive filters (tier log-level + tag), combined with AND
    tier_ceilings = [int(s.value) for s in spec.include if s.kind == "tier"]
    if tier_ceilings:
        ceiling = max(tier_ceilings)
        candidate = {p for p in candidate if _tier_of(p) <= ceiling}
    tag_prefixes = [s.value for s in spec.include if s.kind == "tag"]
    if tag_prefixes:
        candidate = {p for p in candidate if _has_any_tag(p, tag_prefixes)}

    # Buffs: union of buffs.* includes (no implicit default)
    buff_includes = [
        s for s in spec.include if s.kind == "plugin_path" and s.category == "buffs"
    ]
    if buff_includes:
        buffs, rej = _resolve_plugin_paths(buff_includes, "buffs")
        rejected += rej
    else:
        buffs = set()

    # Excludes applied last (exclude wins)
    for selector in spec.exclude:
        if selector.kind == "plugin_path" and selector.category == "probes":
            removed, rej = _resolve_plugin_paths([selector], "probes")
            rejected += rej
            candidate -= removed
        elif selector.kind == "plugin_path" and selector.category == "buffs":
            removed, rej = _resolve_plugin_paths([selector], "buffs")
            rejected += rej
            buffs -= removed
        elif selector.kind == "tier":
            number = int(selector.value)
            removed = {p for p in candidate if _tier_of(p) == number}
            if not removed:
                logging.debug(
                    "run.spec: no active probe of tier %s to remove", number
                )
            candidate -= removed
        elif selector.kind == "tag":
            candidate = {p for p in candidate if not _has_any_tag(p, [selector.value])}

    rejected = sorted(set(rejected))
    if rejected and not skip_unknown:
        raise ValueError(f"unknown run.spec selectors: {rejected}")

    empty_reason = None if candidate else _empty_reason(spec)
    return Resolution(sorted(candidate), sorted(buffs), rejected, empty_reason)
