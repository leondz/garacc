# SPDX-FileCopyrightText: Portions Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the unified ``run.spec`` selection grammar (garak._spec)."""

import pytest

from garak._plugins import enumerate_plugins, plugin_info
from garak._spec import parse_spec_string, parse_spec_file


def _active(category):
    return {name for name, active in enumerate_plugins(category=category) if active}


def _tier(name):
    return int(plugin_info(name).get("tier", 9))


def resolve(spec_str, **kwargs):
    return parse_spec_string(spec_str).resolve(**kwargs)


# --- T1: polarity ---------------------------------------------------------


def test_polarity_bare_plus_equivalent():
    bare = set(resolve("probes.dan").probes)
    plus = set(resolve("+probes.dan").probes)
    assert bare == plus, "bare and '+' selectors must include identically"
    assert bare, "probes.dan family should resolve to at least one active probe"


def test_polarity_minus_excludes():
    family = set(resolve("probes.dan").probes)
    minus = set(resolve("probes.dan, -probes.dan.DanInTheWild").probes)
    assert minus < family, "'-' must remove DanInTheWild from the dan family"
    assert "probes.dan.DanInTheWild" not in minus, "excluded class must be absent"


# --- T2: plugin path forms ------------------------------------------------


def test_plugin_path_glob_is_all_active():
    assert set(resolve("probes.*").probes) == _active(
        "probes"
    ), "probes.* must resolve to all active probes"


def test_plugin_path_family_and_class():
    family = set(resolve("probes.dan").probes)
    assert all(
        p.startswith("probes.dan.") for p in family
    ), "family selector must only yield dan.* probes"
    one = resolve("probes.dan.DanInTheWild").probes
    assert one == ["probes.dan.DanInTheWild"], "explicit class must resolve to itself"


# --- T3: buffs ------------------------------------------------------------


def test_buffs_selected_and_no_default():
    res = resolve("probes.dan, buffs.lowercase")
    assert "buffs.lowercase.Lowercase" in res.buffs, "buffs.lowercase must select the buff"
    assert resolve("probes.dan").buffs == [], "no buffs by default"


# --- T4: tag filter -------------------------------------------------------


def test_tag_is_filter_intersection():
    res = set(resolve("probes.grandma, tag:owasp:llm06").probes)
    family = set(resolve("probes.grandma").probes)
    assert res <= family, "tag: must narrow, not expand the family"
    assert res, "grandma should have probes tagged owasp:llm06"
    assert all(
        any(t.startswith("owasp:llm06") for t in plugin_info(p).get("tags", []))
        for p in res
    ), "every surviving probe must carry the owasp:llm06 tag"


def test_tag_multiple_is_or():
    a = set(resolve("probes.grandma, tag:owasp:llm06").probes)
    b = set(resolve("probes.grandma, tag:risk-cards").probes)
    both = set(resolve("probes.grandma, tag:owasp:llm06, tag:risk-cards").probes)
    assert both == (a | b), "multiple tags must combine as OR"


# --- T5: tier filter (log-level) ------------------------------------------


def test_tier_log_level_is_cumulative():
    t1 = set(resolve("tier:1").probes)
    t2 = set(resolve("tier:2").probes)
    t3 = set(resolve("tier:3").probes)
    assert t1 <= t2 <= t3, "tier:N must be cumulative (1..N)"
    assert all(_tier(p) <= 2 for p in t2), "tier:2 must only contain tiers 1..2"


def test_tier_multiple_takes_widest():
    assert set(resolve("tier:1, tier:3").probes) == set(
        resolve("tier:3").probes
    ), "multiple tier: selectors take the widest (max)"


def test_tier_negative_removes_exact_tier():
    base = set(resolve("tier:3").probes)
    minus = set(resolve("tier:3, -tier:2").probes)
    assert minus == {p for p in base if _tier(p) != 2}, "-tier:2 removes exactly tier 2"


def test_tier_name_equals_int():
    assert set(resolve("tier:of_concern").probes) == set(
        resolve("tier:1").probes
    ), "tier:of_concern must equal tier:1"


@pytest.mark.parametrize("bad", ["tier:99", "tier:notatier"])
def test_tier_invalid_raises(bad):
    with pytest.raises(ValueError):
        parse_spec_string(f"probes.*, {bad}")


# --- empty-result detection -----------------------------------------------


def test_tier_contradiction_empty_with_reason():
    # ansiescape.AnsiEscaped is an active tier-3 probe; tier:1 admits only tier 1
    res = resolve("probes.ansiescape.AnsiEscaped, tier:1")
    assert res.probes == [], "tier:1 must drop a tier-3 explicit class"
    assert res.empty_reason and "tier" in res.empty_reason, "reason must name the tier conflict"


def test_fully_excluded_include_empty_with_reason():
    res = resolve("probes.dan, -probes.dan")
    assert res.probes == [], "excluding the included family yields empty"
    assert res.empty_reason, "empty result must carry a reason"


# --- T8: prefix required / scope ------------------------------------------


def test_prefix_required():
    with pytest.raises(ValueError, match="category prefix"):
        parse_spec_string("dan")


@pytest.mark.parametrize("token", ["detectors.always.Pass", "intent:S"])
def test_out_of_scope_kinds_raise(token):
    with pytest.raises(ValueError):
        parse_spec_string(token)


# --- T9: unknown / skip_unknown -------------------------------------------


def test_unknown_rejected_raises_unless_skipped():
    with pytest.raises(ValueError, match="unknown run.spec"):
        resolve("probes.doesnotexist")
    res = resolve("probes.dan, probes.doesnotexist", skip_unknown=True)
    assert "probes.doesnotexist" in res.rejected, "unknown selector recorded in rejected"
    assert res.probes, "known selectors still resolve under skip_unknown"


# --- exclude wins ---------------------------------------------------------


def test_exclude_wins_over_explicit_include():
    res = resolve("probes.dan.AutoDANCached, -probes.dan")
    assert "probes.dan.AutoDANCached" not in res.probes, "exclude of family wins over explicit class"


# --- T13: implicit default ------------------------------------------------


def test_empty_string_is_probes_star():
    assert set(resolve("").probes) == _active("probes"), "empty spec -> probes.*"


def test_buff_only_keeps_default_probes():
    res = resolve("buffs.lowercase")
    assert set(res.probes) == _active("probes"), "buff-only spec keeps implicit probes.*"
    assert "buffs.lowercase.Lowercase" in res.buffs


def test_tag_only_filters_default_active():
    res = set(resolve("tag:owasp:llm06").probes)
    expected = {p for p in _active("probes") if any(t.startswith("owasp:llm06") for t in plugin_info(p).get("tags", []))}
    assert res == expected, "tag-only spec filters the default-active universe"


# --- T19: buff subtractive ------------------------------------------------


def test_buff_subtractive_all_minus_one():
    res = resolve("probes.lmrc.Bullying, buffs.*, -buffs.paraphrase")
    assert res.buffs, "buffs.* selects active buffs"
    assert not any(b.startswith("buffs.paraphrase.") for b in res.buffs), "-buffs.paraphrase removed"
    assert len(res.probes) == 1, "single probe keeps attempts low"


def test_negative_buff_alone_is_noop():
    res = resolve("-buffs.encoding")
    assert res.buffs == [], "-buffs.Y without a positive buff include is a no-op"


# --- T22: dedup / parsing robustness --------------------------------------


def test_dedup_and_whitespace():
    a = set(resolve("probes.dan, probes.dan,  , probes.dan").probes)
    b = set(resolve("probes.dan").probes)
    assert a == b, "duplicate selectors and blank clauses are tolerated/deduplicated"


# --- T23: tag/tier do not affect buffs ------------------------------------


def test_tag_tier_do_not_touch_buffs():
    res = resolve("buffs.*, tag:owasp:llm01, tier:1")
    assert set(res.buffs) == _active("buffs"), "tag/tier filters must not remove buffs"


# --- T7: CLI <-> file semantic parity -------------------------------------


@pytest.mark.parametrize(
    "spec_str",
    [
        "probes.*",
        "probes.dan, -probes.dan.DanInTheWild, buffs.encoding.Base64",
        "tier:2, tag:owasp:llm01",
        "buffs.lowercase",
        "+probes.*, +tier:3, -tier:2",
    ],
)
def test_cli_file_semantic_parity(spec_str):
    from_cli = parse_spec_string(spec_str)
    round_trip = parse_spec_file(from_cli.to_file_dict())
    cli_res, rt_res = from_cli.resolve(), round_trip.resolve()
    assert cli_res.probes == rt_res.probes, f"probe set differs for {spec_str!r}"
    assert cli_res.buffs == rt_res.buffs, f"buff set differs for {spec_str!r}"


def test_file_mapping_form():
    spec = parse_spec_file(
        {"include": ["probes.dan", {"tag": "owasp:llm01"}], "exclude": [{"tier": 3}]}
    )
    assert any(s.kind == "tag" and s.value == "owasp:llm01" for s in spec.include)
    assert any(s.kind == "tier" and s.value == "3" for s in spec.exclude)
