# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import garak._config
import garak.services.intentservice

@pytest.fixture()
def loaded_intent_service(request):
    garak._config.load_config()
    garak.services.intentservice.load()