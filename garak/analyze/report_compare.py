#!/usr/bin/env python3

# SPDX-FileCopyrightText: Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# compare two garak report digests, note statistically significant differences

import argparse
from dataclasses import dataclass
import sys

from scipy import stats

import garak
import garak.analyze.report_digest


@dataclass
class ProbeDetectorScore:
    """class representing one probe:detector score"""

    absolute_score: float = 0.0
    absolute_defcon: int = 0
    relative_score: float = 0.0
    relative_defcon: int = 0
    n: int = 0
    nones: int = 0


def compare_score_significant(
    target_score, base_score, target_n, base_n, target_nones, base_nones, p=0.05
):
    target_hits = int(target_score * (target_n - target_nones))
    target_passes = target_n - target_nones - target_hits
    target_distr = [1] * target_hits + [0] * target_passes

    base_hits = int(base_score * (base_n - base_nones))
    base_passes = base_n - base_nones - base_hits
    base_distr = [1] * base_hits + [0] * base_passes

    r = stats.ttest_ind(target_distr, base_distr)
    return float(r.pvalue < p)


def compare_reports(target, base):
    target_probe_groups = set(target["eval"].keys())
    base_probe_families = set(base["eval"].keys())

    common_probe_families = target_probe_groups.intersection(base_probe_families)

    target_soft_probe_cap = target["meta"]["setup"]["run.soft_probe_prompt_cap"]
    target_generations = target["meta"]["setup"]["run.generations"]
    base_soft_probe_cap = base["meta"]["setup"]["run.soft_probe_prompt_cap"]
    base_generations = base["meta"]["setup"]["run.generations"]

    for common_probe_family in common_probe_families:
        target_probes = set(target["eval"][common_probe_family].keys())
        target_probes.remove("_summary")
        base_probes = set(base["eval"][common_probe_family].keys())
        base_probes.remove("_summary")

        common_probes = target_probes.intersection(base_probes)

        for common_probe in common_probes:
            target_probe_detectors = set(
                target["eval"][common_probe_family][common_probe]
            )
            target_probe_detectors.remove("_summary")
            base_probe_detectors = set(base["eval"][common_probe_family][common_probe])
            base_probe_detectors.remove("_summary")

            common_detectors = target_probe_detectors.intersection(base_probe_detectors)

            for common_detector in common_detectors:
                target_result = ProbeDetectorScore()
                target_result.absolute_score = target["eval"][common_probe_family][
                    common_probe
                ][common_detector]["absolute_score"]
                target_result.absolute_defcon = target["eval"][common_probe_family][
                    common_probe
                ][common_detector]["absolute_defcon"]
                target_result.relative_score = target["eval"][common_probe_family][
                    common_probe
                ][common_detector]["relative_score"]
                target_result.relative_defcon = target["eval"][common_probe_family][
                    common_probe
                ][common_detector]["relative_defcon"]
                target_result.n = target_generations * target_soft_probe_cap
                target_result.nones = 0

                base_result = ProbeDetectorScore()
                base_result.absolute_score = base["eval"][common_probe_family][
                    common_probe
                ][common_detector]["absolute_score"]
                base_result.absolute_defcon = base["eval"][common_probe_family][
                    common_probe
                ][common_detector]["absolute_defcon"]
                base_result.relative_score = base["eval"][common_probe_family][
                    common_probe
                ][common_detector]["relative_score"]
                base_result.relative_defcon = base["eval"][common_probe_family][
                    common_probe
                ][common_detector]["relative_defcon"]
                base_result.n = base_generations * base_soft_probe_cap
                base_result.nones = 0

                print(common_probe, common_detector)
                if compare_score_significant(
                    target_result.absolute_score,
                    base_result.absolute_score,
                    target_result.n,
                    base_result.n,
                    target_result.nones,
                    base_result.nones,
                ):
                    compare_scores(target_result, base_result)

                    if target_result.absolute_score < base_result.absolute_score:
                        print(
                            f"⬇️ score deteriorated, {target_result.absolute_score} < {base_result.absolute_score}"
                        )
                    else:
                        print(
                            f"💖 score improved, {target_result.absolute_score} > {base_result.absolute_score}"
                        )

                if (
                    target_result.relative_defcon is not None
                    and base_result.relative_defcon is not None
                ):
                    if target_result.relative_defcon < base_result.relative_defcon:
                        print("⬇️ relative defcon deteriorated")
                    else:
                        print("💖 relative defcon improved")


def main(argv=None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    garak._config.load_config()
    print(
        f"garak {garak.__description__} v{garak._config.version} ( https://github.com/NVIDIA/garak )"
    )

    parser = argparse.ArgumentParser(
        prog="python -m garak.analyze.report_compare",
        description="Compare two garak report digests, and describe how a report deviates from a base report",
        epilog="See https://github.com/NVIDIA/garak",
        allow_abbrev=False,
    )
    parser.add_argument(
        "-r",
        "--report_path",
        required=True,
        help="Path to the garak JSONL report under analysis",
    )
    parser.add_argument(
        "-b",
        "--base_report_path",
        required=True,
        help="Path to the base garak JSONL report used as reference",
    )
    parser.add_argument(
        "-p",
        required=False,
        type=float,
        default=0.05,
        help="p-value threshold for statistical test",
    )
    args = parser.parse_args(argv)

    sys.stdout.reconfigure(encoding="utf-8")
    target = garak.analyze.report_digest._get_report_digest(args.report_path)
    base = garak.analyze.report_digest._get_report_digest(args.report_path)
    compare_reports(target, base)


if __name__ == "__main__":
    main()
