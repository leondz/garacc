# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Retrieval of intents and intent stubs."""

import json
import pathlib
import re
from typing import List, Set

import garak.data

intents = {}
cas_data_path = garak.data.path / "cas"


def start_msg() -> str:
    """return a start message, assumes enabled"""
    return "🎯", "loading intent service"


def enabled() -> bool:
    """are requirements met for intent service to be enabled"""
    return True


def _load_intent_typology(intents_path=None) -> None:
    global intents
    if intents_path is None:
        intents_path = cas_data_path / "trait_typology.json"
    with open(intents_path, "r", encoding="utf-8") as intents_file:
        intents = json.load(intents_file)


def load():
    """load the service"""
    _load_intent_typology()


def _get_stubs_typology(intent_code: str) -> Set[str]:
    # return the descr of a given typoilogy point, or if empty/absent, the name
    intent_details = intents.get(intent_code, {})

    stub = intent_details.get("descr")
    if not stub:
        stub = intent_details.get("name")

    stubs = (
        set(
            [
                stub,
            ]
        )
        if stub
        else set()
    )  # careful string doens't get char'd
    return stubs


def _get_stubs_file(intent_code: str) -> Set[str]:

    # search path: cas/intent_text/xxx.txt, cas/intent_text/xxx_extra.txt
    # _extra suffix is for augmenting instead of overriding in user dirs,
    # and so must not be present in core
    core_filepath = cas_data_path / "intent_stubs" / f"{intent_code}.txt"
    extra_filepath = cas_data_path / "intent_stubs" / f"{intent_code}_extra.txt"

    stubs = set()
    for stub_file_path in (core_filepath, extra_filepath):
        if stub_file_path.exists():
            with open(stub_file_path, "r", encoding="utf-8") as sf:
                for line in sf:
                    stubs.add(line)

    return stubs


def _get_stubs_code(intent_code: str) -> Set[str]:
    intent_module_path = pathlib.Path
    return set()


def get_intent_stubs(intent_specifier: str) -> List[str]:
    """retrieve a list of intent strings given an intent code"""

    if not re.fullmatch("[CTMS]([0-9]{3}([a-z]+)?)?", intent_specifier):
        raise ValueError("Not a valid intent code: " + intent_specifier)

    intent_codes_to_lookup = set()
    intent_codes_to_lookup.add(intent_specifier)

    # expand intent codes
    if len(intent_specifier) <= 4:
        for code in intents.keys():
            if code.startswith(intent_specifier):
                intent_codes_to_lookup.add(code)

    stubs = set()
    # retrieve intent stubs
    for candidate_code in intent_codes_to_lookup:
        stubs.update(_get_stubs_typology(candidate_code))
        stubs.update(_get_stubs_file(candidate_code))
        stubs.update(_get_stubs_code(candidate_code))

    # return stubs
    return stubs
