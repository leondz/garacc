# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

**garak** is NVIDIA's open-source LLM vulnerability scanner (Generative AI Red-teaming & Assessment Kit). It probes LLMs for vulnerabilities like hallucination, data leakage, prompt injection, toxicity, and jailbreaks. Built on a plugin architecture with five core plugin types: probes, detectors, generators, harnesses, and buffs.

- **Branch**: `automated-red-teaming` — adds intent-based testing via `EarlyStopHarness`
- Python >=3.10 (CI tests on 3.10, 3.12, 3.13 across Linux/macOS/Windows)
- Build backend: flit-scm
- License: Apache-2.0

## Build & Development Commands

```bash
# Install in editable mode
pip install -e .

# Install with extras
pip install -e ".[tests]"        # pytest, pytest-mock, pytest-cov, etc.
pip install -e ".[lint]"         # black>=26.1.0, pylint>=3.1.0
pip install -e ".[calibration]"  # scipy
pip install -e ".[audio]"       # soundfile, librosa
pip install -e ".[dra]"         # detoxify

# Run all tests
python -m pytest tests/

# Run a specific test file / single function
python -m pytest tests/test_attempt.py
python -m pytest tests/test_attempt.py::test_message_setup

# Run tests for a plugin category
python -m pytest tests/probes/
python -m pytest tests/detectors/

# Format code (line length 88, target Python 3.10)
black garak/

# Lint (pylint max-line-length=100, intentionally wider than black's 88)
pylint garak/

# Run garak (standard probe mode)
python -m garak --target_type openai --target_name gpt-4 --probes dan

# Run garak (intent-based mode — triggers EarlyStopHarness)
python -m garak --target_type openai --target_name gpt-4 --intents S
```

## Code Style

- **black**: line-length=88, target py310 (configured in `pyproject.toml`)
- **pylint**: max-line-length=100 (configured in `pylintrc`) — black formats to 88 but pylint allows up to 100
- Pre-commit hooks configured for `docs/` only (black + standard hooks)

## Architecture

### Plugin System

All functionality is built around five plugin types, each in its own subdirectory under `garak/`:

- **probes/** — Attack strategies that generate adversarial prompts (e.g., DAN jailbreaks, encoding attacks, TAP)
- **detectors/** — Analyze model outputs to determine if a vulnerability was triggered
- **generators/** — Interfaces to LLMs (OpenAI, HuggingFace, Bedrock, LiteLLM, Ollama, etc.)
- **harnesses/** — Orchestrate how probes, detectors, and generators interact during a run
- **buffs/** — Transform probe outputs (paraphrase, encode, translate) before detection

Additional subdirectories: `analyze/` (reporting), `evaluators/`, `langproviders/`, `intents/` (intent definitions), `services/` (intent service), `resources/` (YAML configs, JSON caches), `configs/`, `data/`.

All plugins inherit from `Configurable` (`garak/configurable.py`), which provides YAML config loading, environment variable validation, and dependency management. Plugins are dynamically loaded by dotted name (e.g., `"probes.dan.Dan_11_0"`) via `_plugins.load_plugin()`.

### Execution Flow

1. CLI (`cli.py`) parses args -> `_config` loads layered config -> `command.py` selects run path
2. A **harness** iterates over probes. Three harness strategies:
   - `ProbewiseHarness` (default) — each probe uses its own recommended detectors
   - `PxD` — all probes x all detectors (exhaustive)
   - `EarlyStopHarness` — intent-based adaptive testing with early stopping (triggered when `--intents` is set)
3. Each **probe** calls `prepare()` to create `Attempt` objects, then `execute()` sends them to the generator
4. **Detectors** score each attempt's outputs (0.0-1.0, where 1.0 = vulnerability found)
5. **Evaluator** applies threshold (`eval_threshold`, default 0.5) to determine pass/fail. `evaluator.test(score)` returns `True` if SAFE (below threshold)
6. Results written to JSONL report files in `garak_runs/`

### Configuration Precedence (lowest to highest)

1. Plugin class defaults
2. `garak/resources/garak.core.yaml` (base config)
3. Site config (`~/.config/garak/`)
4. Run config file (`--config`)
5. CLI arguments

### Data Model (`attempt.py`)

- **Message** (line 21) — Single text/binary communication unit with optional metadata (`notes` dict, `data_path`, `data_type`)
- **Turn** (line 91) — Role (`system`/`user`/`assistant`) + Message pair
- **Conversation** (line 120) — Ordered sequence of Turns (dialogue history)
- **Attempt** (line 174) — Wraps a prompt (Conversation), outputs (List[Message]), detector_results, and metadata (uuid, goal, intent, tags, probe info)

### Intent/CAS System

The intent system (`garak/cas.py`, `garak/intents/`, `garak/services/intentservice.py`) enables high-level objective-driven testing. Intents define what to test; stubs (`TextStub`, `ConversationStub`) carry content. When `--intents` is set (e.g., `--intents S`), the `EarlyStopHarness` is used instead of the default probewise harness. By default `intent_spec` is `None` (standard probe mode).

- **Intent codes** follow format `TDDDs*` — e.g., `S001fraud`, `S002hatespeech`
- **`cas.py`** — Policy system for context-aware scanning, with trait typologies and hierarchical codes
- **`intentservice.py`** — Loads intents and stubs from multiple sources (typology, `.txt`, `.json`, `.yaml`, code)
- **`garak.core.yaml` CAS settings**: `intent_spec` (None by default), `expand_intent_tree`, `trust_code_stubs` (False by default), `serve_detectorless_intents`

### Plugin Discovery and Caching

`_plugins.py` provides:
- `enumerate_plugins(category)` — discover all plugins in a category
- `load_plugin(name)` — instantiate by dotted name
- `PluginCache` — caches plugin metadata in `resources/plugin_cache.json` with user overrides in XDG cache dir
- `PluginProvider` — singleton/thread-safe instance cache

## EarlyStopHarness — Intent-Based Adaptive Testing

The `EarlyStopHarness` (`garak/harnesses/earlystop.py`) is the primary harness for automated red-teaming. It implements a **funnel pipeline** that progressively applies attack techniques to a set of intent stubs, stopping early when all stubs have been jailbroken or all attack methods are exhausted.

### Core Concepts

- **Intent Stub** — An original test prompt from the intent service. This is the unit of measurement; all metrics track back to these original stubs.
- **Intent** — A harm category (e.g., `S001fraud`, `S002hatespeech`). Each intent has multiple stubs.
- **Accepted** — A stub whose model output was classified as compliant (jailbroken). The model produced harmful content.
- **Rejected** — A stub whose model output was classified as a refusal (safe). The model declined.

### Pipeline Flow

```
All intent stubs
    |
Stage 1: Baseline (IntentProbe -- sends stubs unmodified)
    -> Accepted stubs removed from pipeline
    -> Only rejected stubs continue
    |
Stage 2: Attack probe (e.g., SPO -- system prompt override)
    -> Newly accepted stubs removed
    -> Only rejected stubs continue
    |
Stage 3: Attack probe (e.g., TAP -- tree of attacks with pruning)
    -> ...continues until no rejected stubs remain or probes exhausted
```

### How Rejection Works (`_is_rejected`, line 26)

The function runs ALL detectors on the attempt, then evaluates each score against the threshold. **One safe score from any detector in any generation -> the entire attempt is rejected.** This is conservative (AND-like) consensus: every detector on every generation must agree the output is harmful for the attempt to be considered "accepted" (jailbroken). `any(evaluations)` at line 58 means "at least one safe signal found."

### How Stub Status Updates (`_update_attempt_status`, line 177)

For each previously-rejected stub, the harness groups attacked attempts by stub (via `notes.get("stub")` equality). If ALL attacks for that stub were rejected (or no attacks were produced), the stub stays rejected. If ANY attack was NOT rejected, the stub moves to accepted. This implements the "any compliance = jailbroken" rule.

### Stub Linkage

Attack probes (SPO, TAP, Translation) may modify or completely rewrite the prompt text, but the original `TextStub` object is always preserved in `attempt.notes["stub"]` via `IntentProbe._attempt_prestore_hook` (`probes/base.py:845-852`). The harness compares stubs using value equality (`intent` + `_content`), so tracking is maintained even when prompts are transformed.

The `IntentProbe` base class (`probes/base.py:830`) manages this pipeline:
1. `_populate_intents()` (line 854) — loads applicable intents from the intent service
2. `_populate_stubs()` (line 861) — gets stubs for each intent, applies `_expand_stub()` (1:many)
3. `build_prompts()` (line 886) — transforms stubs into prompts via `_prompts_from_stub()` (1:many), recording the original stub in `prompt_stubs[seq]`
4. `_attempt_prestore_hook()` (line 845) — stamps `attempt.notes["stub"]` with the original `TextStub` before the attempt is stored

### Summary Entries (line 288-295)

At the end of the pipeline, the harness writes one JSONL entry per original stub with the cumulative outcome:

- `detector_results`: `{"EarlyStop": [1.0]}` (jailbroken by some probe) or `{"EarlyStop": [0.0]}` (all probes failed)

These summary entries are the authoritative source for computing full-pipeline Attack Success Rate (ASR):

```
ASR = (stubs with EarlyStop score 1.0 / total stubs) x 100%
```

### Compatible Probes

The harness maintains a `compatible_probes` list (`DEFAULT_PARAMS`, line 121-131). Only `IntentProbe` subclasses in this list are loaded:

- `grandma.GrandmaIntent`
- `tap.TAPIntent`
- `multilingual.TranslationIntent`
- `spo.SPOIntent`, `spo.SPOIntentUserAugmented`, `spo.SPOIntentSystemAugmented`, `spo.SPOIntentBothAugmented`

### Detector Scoring Convention

All detector scores follow: **higher = attack succeeded**. A score of 1.0 means the model produced harmful content; 0.0 means the model refused. The evaluator's `test()` returns `True` when a score indicates safety (below `eval_threshold`), which is why `any(evaluations)` in `_is_rejected` means "at least one safe signal found."

### Known Pitfalls

- **Stub type mismatch**: `notes["stub"]` must be a `TextStub` object, not a raw string. If a probe overwrites it with a string after `_attempt_prestore_hook`, stub comparison in `_update_attempt_status` silently returns `False`, causing jailbreak results to be discarded.
- **`intent_spec` default**: Must be `None` (not `"S"`) in `garak.core.yaml` so standard probe runs don't inadvertently trigger the intent system and EarlyStopHarness.
- **Detector format**: Intermediate attempts use nested format `[[scores]]` for detector_results, but EarlyStop summary entries use flat format `[score]`. Code consuming these entries must handle both.

## Testing Conventions

- Tests mirror the main package structure: `tests/probes/`, `tests/detectors/`, etc.
- The `conftest.py` auto-use `config_cleanup` fixture reloads `_config` and clears plugin instances between tests
- Tests suppress logging unless `GARAK_LOG_FILE` env var is set
- Use `@pytest.mark.requires_storage(required_space_gb=N)` for tests needing disk space
- The `loaded_intent_service` fixture sets up the intent system for CAS tests
- `COMPLYING_OUTPUTS` and `REFUSAL_OUTPUTS` in conftest provide standard test data

## Key Files

| File | Purpose |
|------|---------|
| `garak/_config.py` | Global config system, layered YAML loading |
| `garak/_plugins.py` | Plugin discovery, loading, caching |
| `garak/cli.py` | CLI argument parsing and run dispatch |
| `garak/command.py` | Run orchestration (probewise_run, pxd_run, early_stop_run) |
| `garak/configurable.py` | Base class for all configurable plugins |
| `garak/attempt.py` | Attempt, Message, Turn, Conversation data classes |
| `garak/cas.py` | Intent/stub system, Policy class, trait typologies |
| `garak/probes/base.py` | Probe, TreeSearchProbe, IterativeProbe, IntentProbe base classes |
| `garak/generators/base.py` | Generator base class with `_call_model()` |
| `garak/detectors/base.py` | Detector base class with `detect()` |
| `garak/harnesses/earlystop.py` | EarlyStopHarness — intent-based funnel pipeline |
| `garak/services/intentservice.py` | Intent loading, stub retrieval, intent-to-detector mapping |
| `garak/resources/garak.core.yaml` | Default configuration values |
| `pyproject.toml` | Build config, dependencies, extras |
| `pylintrc` | Pylint configuration (max-line-length=100) |

## Important Patterns

- Generators must implement `_call_model(prompt, generations_this_call)` -> `List[Message | None]` where prompt is a `Conversation`
- Probes set `primary_detector` and `extended_detectors` to declare which detectors to use
- Probes declare `modality` dict for input/output types (text, image, audio)
- API keys are loaded via `key_env_var` / `ENV_VAR` class attributes -> `os.getenv()`
- Plugins declare `extra_dependency_names` for optional imports loaded at construction time
- The `_supported_params` class attribute controls which config keys a plugin accepts
- `TreeSearchProbe` (base.py:469) — base for tree-search attack strategies with breadth-first/depth-first traversal and per-node thresholds
- `IterativeProbe` (base.py:677) — base for multi-turn probes where probe uses the last target response to generate the next prompt; supports `max_calls_per_conv` and exponential attempt expansion via BFS
- `IntentProbe` (base.py:830) — base for intent-based probes used by EarlyStopHarness; manages stub population, prompt building, and stub linkage via `_attempt_prestore_hook`
