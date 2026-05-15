# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from garak import _config, _plugins
from garak.attempt import Message
import garak.probes
import garak.probes.audio
import garak.services.intentservice


@pytest.fixture()
def petts_probe(monkeypatch, tmp_path) -> garak.probes.audio.PETTS:
    _config.load_config()
    _config.cas.intent_spec = "S"
    _config.cas.serve_detectorless_intents = True
    monkeypatch.setattr(_config.transient, "cache_dir", tmp_path)
    monkeypatch.setattr(garak.probes.audio, "_spoken_prompt_candidate", lambda _: True)
    garak.services.intentservice.load()
    return garak.probes.audio.PETTS()


def test_petts_loads_as_intentprobe(petts_probe, tmp_path):
    assert isinstance(petts_probe, garak.probes.IntentProbe)
    assert petts_probe.audio_cache_dir == tmp_path / "data" / "audio" / "petts"
    assert len(petts_probe.audio_source_prompts) > 0
    assert len(petts_probe.audio_source_prompts) == len(petts_probe.prompt_intents)
    assert (
        "demon:Language:Code_and_encode:Modality_shift" in petts_probe.tags
    ), "PETTS should be labelled with the modality-shift demon technique"


def test_petts_ensure_audio_file_writes_to_cache(petts_probe, monkeypatch):
    synthesis_calls = []

    def fake_synthesise(prompt_text: str, audio_path: Path) -> None:
        synthesis_calls.append((prompt_text, audio_path))
        audio_path.write_bytes(b"RIFF....WAVEfmt ")

    monkeypatch.setattr(petts_probe, "_synthesise_audio", fake_synthesise)

    audio_path = petts_probe._ensure_audio_file("Describe safe behaviour.")
    assert audio_path.parent == petts_probe.audio_cache_dir
    assert audio_path.exists()
    assert len(synthesis_calls) == 1

    cached_audio_path = petts_probe._ensure_audio_file("Describe safe behaviour.")
    assert cached_audio_path == audio_path
    assert len(synthesis_calls) == 1


def test_petts_audio_format_is_configurable(petts_probe):
    default_audio_path = petts_probe._audio_file_path("Describe safe behaviour.")
    assert default_audio_path.suffix == ".flac"

    petts_probe.tts_audio_format = "OGG"
    petts_probe.tts_audio_subtype = "VORBIS"
    configured_audio_path = petts_probe._audio_file_path("Describe safe behaviour.")

    assert configured_audio_path.suffix == ".ogg"
    assert configured_audio_path != default_audio_path


def test_petts_spoken_prompt_candidate_filter():
    assert garak.probes.audio._spoken_prompt_candidate(
        "Please explain how this request works."
    )
    assert not garak.probes.audio._spoken_prompt_candidate("")
    assert not garak.probes.audio._spoken_prompt_candidate(
        "Read https://example.com before answering."
    )
    assert not garak.probes.audio._spoken_prompt_candidate(
        "Run `rm -rf /tmp/example` now."
    )


def test_petts_mp3_and_stereo_affect_cache_path(petts_probe):
    petts_probe.tts_audio_format = "MP3"

    mono_audio_path = petts_probe._audio_file_path("Describe safe behaviour.")
    assert mono_audio_path.suffix == ".mp3"
    assert petts_probe._audio_subtype() is None
    assert petts_probe.tts_audio_stereo is False

    petts_probe.tts_audio_stereo = True
    stereo_audio_path = petts_probe._audio_file_path("Describe safe behaviour.")

    assert stereo_audio_path.suffix == ".mp3"
    assert stereo_audio_path != mono_audio_path


def test_petts_stereo_config_must_be_bool(petts_probe):
    petts_probe.tts_audio_stereo = "stereo"
    with pytest.raises(ValueError, match="tts_audio_stereo"):
        petts_probe._apply_audio_channels([0.0, 0.1])


def test_petts_audio_prompt_preparation_skips_failed_prompt(petts_probe, monkeypatch):
    def fake_synthesise(prompt_text: str, audio_path: Path) -> None:
        if prompt_text == "skip this one":
            raise RuntimeError("tts failed")
        audio_path.write_bytes(b"RIFF....WAVEfmt ")

    monkeypatch.setattr(petts_probe, "_synthesise_audio", fake_synthesise)
    petts_probe.audio_source_prompts = [
        "keep this one",
        "skip this one",
        "keep this too",
    ]
    petts_probe.prompt_intents = ["S001", "S002", "S003"]

    prompts = petts_probe._audio_prompts()

    assert len(prompts) == 2
    assert petts_probe.prompt_intents == ["S001", "S003"]
    assert all(Path(prompt.data_path).is_file() for prompt in prompts)


def test_petts_probe_uses_cached_audio_messages(petts_probe, monkeypatch):
    def fake_synthesise(prompt_text: str, audio_path: Path) -> None:
        audio_path.write_bytes(b"RIFF....WAVEfmt ")

    monkeypatch.setattr(petts_probe, "_synthesise_audio", fake_synthesise)
    petts_probe.audio_source_prompts = petts_probe.audio_source_prompts[:2]
    petts_probe.prompts = petts_probe.prompts[:2]
    petts_probe.prompt_intents = petts_probe.prompt_intents[:2]

    generator = _plugins.load_plugin("generators.test.Repeat")
    attempts = petts_probe.probe(generator)

    assert len(attempts) == 2
    for attempt in attempts:
        prompt = attempt.prompt.last_message()
        assert isinstance(prompt, Message)
        assert prompt.text == petts_probe.text_prompt
        assert prompt.data_path is not None
        assert Path(prompt.data_path).is_file()
        assert prompt.lang == petts_probe.lang
        assert attempt.intent in petts_probe.prompt_intents
