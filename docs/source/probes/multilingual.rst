Translation-based jailbreak attacks
====================================

Probes in this module test whether content moderation can be bypassed by
translating malicious requests into different languages.

The translation attack vector works by exploiting potential weaknesses in
multilingual models where harmful content detection may be less effective
in non-English languages. The model receives translated prompts and returns
translated responses, which are then reverse-translated back to English for
evaluation.

This module uses garak's langservice infrastructure for translation.

Relative paper:
- https://arxiv.org/abs/2310.02446 - Multilingual Jailbreak Challenges in LLMs
