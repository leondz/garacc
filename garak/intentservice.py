# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Retrieval of intents and intent stubs."""

import json
from typing import List

import garak.data

intents = {}


def start_msg() -> str:
    """return a start message, assumes enabled"""
    return "🎯", "loading intent service"


def enabled() -> bool:
    """are requirements met for intent service to be enabled"""
    return True


def _load_intent_typology(intents_path=None) -> None:
    global intents
    if intents_path is None:
        intents_path = garak.data.path / "cas" / "trait_typology.json"
    with open(intents_path, "r", encoding="utf-8") as intents_file:
        intents = json.load(intents_file)


def load():
    """load the service"""
    _load_intent_typology()


def get_intent_stubs(intent_code: str) -> List[str]:
    """retrieve a list of intent strings given an intent code"""

    intent_codes_to_lookup = set()
    intent_codes_to_lookup.add(intent_code)

    # expand intent codes
    if len(intent_code) <= 4:
        for code in intents.keys():
            if code.startswith(intent_code):
                intent_codes_to_lookup.add(code)

    stubs = set()
    # retrieve intent stubs
    for candidate_code in intent_codes_to_lookup:
        intent_details = intents.get(candidate_code, {})
        stub = intent_details.get("descr")
        if stub:
            stubs.add(stub)
            continue
        stub = intent_details.get("name")
        if stub:
            stubs.add(stub)

    # return stubs
    return stubs
