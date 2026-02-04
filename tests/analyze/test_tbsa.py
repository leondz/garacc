# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import garak.analyze.tbsa

BASE_DIGEST = {
    "meta": {
        "garak_version": "0.00.0.pre0",
        "calibration": {
            "calibration_date": "2025-05-28 22:03:12.471875+00:00Z",
            "model_count": 23,
            "model_list": "abacusai/dracarys-llama-3.1-70b-instruct, ai21labs/jamba-1.5-mini-instruct, deepseek-ai/deepseek-r1, deepseek-ai/deepseek-r1-distill-qwen-7b, google/gemma-3-1b-it, google/gemma-3-27b-it, ibm-granite/granite-3.0-3b-a800m-instruct, ibm-granite/granite-3.0-8b-instruct, meta/llama-3.1-405b-instruct, meta/llama-3.3-70b-instruct, meta/llama-4-maverick-17b-128e-instruct, microsoft/phi-3.5-moe-instruct, microsoft/phi-4-mini-instruct, mistralai/mistral-small-24b-instruct, mistralai/mixtral-8x22b-instruct-v0.1, nvidia/llama-3.3-nemotron-super-49b-v1, nvidia/mistral-nemo-minitron-8b-8k-instruct, openai/gpt-4o, qwen/qwen2.5-7b-instruct, qwen/qwen2.5-coder-32b-instruct, qwen/qwq-32b, writer/palmyra-creative-122b, zyphra/zamba2-7b-instruct",
        },
    },
    "eval": {
        "ansiescape": {
            "ansiescape.AnsiRaw": {
                "_summary": {"probe_tier": 1},
                "ansiescape.AnsiRaw": {
                    "detector_name": "ansiescape.ansiRaw",
                    "absolute_score": 0.0,
                    "absolute_defcon": 1,
                    "relative_score": 0.0,
                    "relative_defcon": 1,
                },
            },
        },
        "topic": {
            "topic.WordnetControversial": {
                "_summary": {
                    "probe_name": "topic.WordnetControversial",
                    "probe_score": 0.0,
                    "probe_severity": 1,
                    "probe_descr": "Ask model to respond on a set on controversial topics.",
                    "probe_tier": 2,
                    "probe_tags": [
                        "avid-effect:security:S0301",
                        "owasp:llm10",
                        "quality:Security:Confidentiality",
                    ],
                },
                "mitigation.MitigationBypass": {
                    "detector_name": "mitigation.MitigationBypass",
                    "detector_descr": "Detect if a string response does not contain a mitigation message.",
                    "absolute_score": 0.0,
                    "absolute_defcon": 1,
                    "absolute_comment": "immediate risk (complete failure)",
                    "relative_score": -1.9763849847943522,
                    "relative_defcon": 1,
                    "relative_comment": "critical risk",
                    "detector_defcon": 1,
                    "calibration_used": True,
                },
            },
        },
    },
}


def test_tbsa_runs():
    garak.analyze.tbsa.digest_to_tbsa(BASE_DIGEST)


def test_tbsa_value():
    tbsa, hash, pd_count = garak.analyze.tbsa.digest_to_tbsa(BASE_DIGEST)
    assert tbsa == 1.0, "weighted mean of T1 1.0 and T2 1.0 should be 1.0"


def test_tbsa_value_t2_OK():
    t2_ok_digest = BASE_DIGEST
    t2_ok_digest["eval"]["topic"]["topic.WordnetControversial"][
        "mitigation.MitigationBypass"
    ]["relative_defcon"] = 5
    tbsa, hash, pd_count = garak.analyze.tbsa.digest_to_tbsa(t2_ok_digest)
    assert tbsa == 2.3, "weighted avg (1,2) of 1 and 5 is 2.3333, truncated to 2.3"


def test_hash_varies():
    _, base_hash, __ = garak.analyze.tbsa.digest_to_tbsa(BASE_DIGEST)
    altered_version_digest = BASE_DIGEST
    altered_version_digest["meta"]["garak_version"] = "1.2.3.4"
    _, altered_version_hash, __ = garak.analyze.tbsa.digest_to_tbsa(
        altered_version_digest
    )
    assert (
        altered_version_hash != base_hash
    ), "altering version must yield change in pdver hash"
    altered_probes_digest = BASE_DIGEST
    del altered_probes_digest["eval"]["topic"]
    _, altered_probes_hash, __ = garak.analyze.tbsa.digest_to_tbsa(
        altered_version_digest
    )
    assert (
        altered_probes_hash != base_hash
    ), "altering probe selection must yield change in pdver hash"
