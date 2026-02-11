# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import garak.attempt
import garak.intents

TEST_INTENT = "S009deep"
TEST_CONTENT = "9 2 1 8 BLACK"
TEST_CONV = garak.attempt.Conversation([garak.attempt.Message(TEST_CONTENT)])


def test_stub_basic():
    s = garak.intents.Stub()
    assert s.intent is None, "Stub intent should be None on raw stub"
    assert s.content is None, "Stub content should be None on raw stub"


def test_stub_getset():
    s = garak.intents.Stub()
    s.intent = TEST_INTENT
    s.content = TEST_CONTENT
    assert s.intent == TEST_INTENT, "Stub intent should as per string set in .intent"
    assert (
        s.content == TEST_CONTENT
    ), "Stub content should as per string set in .content"


def test_stub_construct():
    s = garak.intents.Stub(TEST_INTENT, TEST_CONTENT)
    assert (
        s.intent == TEST_INTENT
    ), "Stub intent should as per string set in constructor"
    assert (
        s.content == TEST_CONTENT
    ), "Stub content should as per string set in constructor"


def test_convstub_basic():
    c = garak.intents.ConversationStub()
    assert c.intent is None, "ConversationStub intent should be None on raw stub"
    assert c.content is None, "ConversationStub content should be None on raw stub"


def test_convstub_getset_intent():
    c = garak.intents.ConversationStub()
    c.intent = TEST_INTENT
    assert (
        c.intent == TEST_INTENT
    ), "ConversationStub intent should be as set in .intent"


def test_convstub_getset_str():
    c = garak.intents.ConversationStub()
    c.content = TEST_CONTENT
    assert (
        c.content == TEST_CONV
    ), "ConversationStub content should be Conversation created from str sent to .content"


def test_convstub_getset_conv():
    c = garak.intents.ConversationStub()
    c.content = TEST_CONV
    assert (
        c.content == TEST_CONV
    ), "ConversationStub content should be Conversation as set in .content"


def test_convstub_construct_intent():
    c = garak.intents.ConversationStub(TEST_INTENT, "")
    assert (
        c.intent == TEST_INTENT
    ), "ConversationStub intent should as per string set in constructor"


def test_convstub_construct_str():
    c = garak.intents.ConversationStub(TEST_INTENT, TEST_CONTENT)
    assert (
        c.content == TEST_CONV
    ), "ConversationStub content should be Conversation created from str sent in constructor"


def test_convstub_construct_conv():
    c = garak.intents.ConversationStub(TEST_INTENT, TEST_CONV)
    assert (
        c.content == TEST_CONV
    ), "ConversationStub content should be Conversation given in Constructor"
