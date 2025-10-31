# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Retrieval of intents and intent stubs."""


def start_msg() -> str:
    """return a start message, assumes enabled"""
    return "🎯", "loading intent service"


def get_intents() -> str:
    pass