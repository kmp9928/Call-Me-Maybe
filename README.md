*This project has been created as part of the 42 curriculum by kimendon.*

# call-me-maybe

## Description

`call-me-maybe` turns a plain natural-language prompt into a structured function call — a function name plus correctly-typed arguments — without relying on a large hosted model's built-in "function calling" feature.

Given a list of available functions (name, description, typed parameters) and a set of natural-language prompts, the program:

1. Picks which function a prompt is asking for.
2. Extracts each of that function's argument values, converted to the right type (string, number, integer, boolean).
3. Writes the results out as JSON.

The goal of the project is to do this reliably and deterministically using a small, local, general-purpose language model by manually implementing **constrained decoding**: instead of asking the model to freely generate text and hoping it comes back well-formed, the program inspects the model's raw next-token probabilities at every generation step and forces it down a path that can only ever produce syntactically valid, schema-conforming — and, for most parameter types, prompt-grounded — output.

## Instructions

**Requirements:** Python ≥ 3.10, [`uv`], `Qwen/Qwen3-0.6B` model, numpy and pydantic.

**Install dependencies:**

```bash
make install
```

**Run the program:**


```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

**Lint / type-check:**

```bash
make lint          # flake8 + mypy
```

## Resources

- [`argparse` docs](https://docs.python.org/3/library/argparse.html) — CLI argument parsing (`src/input_reader.py`).
- [Pydantic documentation](https://docs.pydantic.dev/) — input validation models (`src/models.py`).
- [Constrained Decoding for Structured LLM Output](https://mbrenndoerfer.com/writing/constrained-decoding-structured-llm-output#constrained-beam-search) — background on constrained decoding techniques.

**How AI was used:** Concretely, it was used to:

- Do root-cause analysis of specific extraction bugs.
- Discuss and pressure-test candidate fixes before they were implemented.
- Draft this README's technical sections from the actual codebase.

## Algorithm Explanation

Extraction happens in two stages, both built on the same core mechanism.

**Stage 1 — function name selection.** The prompt, plus every candidate function's name and description, is shown to the model, which is constrained to output exactly one of the known function names (a closed vocabulary — nothing else is a valid token sequence).

**Stage 2 — parameter extraction**, one parameter at a time. For each parameter, the model is shown the function's schema, the parameters already resolved so far and the user's prompt, and is asked to produce that one parameter's value as a JSON literal.

Both stages run the same core loop (`JSONExtractor.extract`):

1. Start from a prefix (`"` or `"/`).
2. Repeat:
   - Encode `base_prompt + output_so_far` and run **one forward pass** through the model to get raw next-token logits (no sampling, no softmax — just the raw scores).
   - For every token in the model's vocabulary, check whether appending it keeps the output valid:
     - **Shape check** — does `output_so_far + token` still match a hand-written *partial* regex for this parameter's type (e.g. "still looks like the start of a quoted string", "still looks like the start of a number")?
     - **Grounding check** (for string/number types) — does the decoded value, so far, still occur as a literal substring of the user's original prompt? This is what stops the model from *inventing* plausible-but-wrong content instead of copying it.
     - Any token failing either check has its logit set to `-inf`.
   - Pick the single highest-logit surviving token — **pure greedy argmax**, deterministic, no randomness — and append it.
   - Stop once the accumulated output fully matches the expected parameter or once a hard step-count cap is exceeded (raises `JSONExtractorTimeoutError` instead of looping forever).
3. Convert the final text into the right Python type (`str`/`int`/`float`/`bool`), with a few type-specific touches (e.g. collapsing a run of the same repeated character down to one for a "replacement" symbol parameter).

Different parameter shapes get their own extractor subclass, each with its own regex pair and finalization step:

| Extractor | Used for | Notes |
|---|---|---|
| `LiteralJSONExtractor` / `BooleanJSONExtractor` | function name, booleans | closed, fixed vocabulary |
| `StringJSONExtractor` | generic strings | quoted + grounded against the prompt |
| `ReplacementJSONExtractor` | parameters named `replacement` | narrower grammar for single symbols |
| `RegexJSONExtractor` | parameters named `regex` | generates one of a small set of canonical patterns (`\d+`, `\bword\b`, …) |
| `NumberJSONExtractor` / `IntegerJSONExtractor` | numeric parameters | grounded like strings; integers use a strict digits-only regex and parse with `int()` or `float()`|
| `PathJSONExtractor` | parameters named `path` | biases toward an absolute-path prefix |

## Design Decisions

- **Greedy decoding, no sampling.** This is structured extraction, not creative writing — for a given prompt and schema there should be exactly one deterministic, reproducible answer, so always taking the single highest-logit valid token is the right call.
- **Shape constraint + prompt-grounding constraint, not shape alone.** A regex only guarantees the output *looks like* valid JSON. Requiring every candidate value to be a literal substring of the prompt closes off outright fabrication.
- **Per-parameter-name specialization instead of one universal string extractor.** A regex parameter, a replacement symbol, and free text are fundamentally different shapes; rather than stretching one generic extractor to cover all of them, each gets its own narrow, purpose-built grammar.

## Performance Analysis

**Accuracy:** 11/11 (100%) on the example prompts in `data/input/function_calling_tests.json`, verified by running the pipeline end-to-end and checking each result.

**Reliability:** fully deterministic. Greedy argmax decoding means the same prompt, schema, and model/hardware combination always produces the same output — important for a grading harness that expects one canonical correct answer per parameter.

**Speed:** slow (under 3 minutes) relative to a hosted LLM API, by design of running everything locally. Every single generation step re-encodes `base_prompt + output_so_far` from scratch and runs a full forward pass (`LLM.get_logits`), with no KV-cache reuse between steps. Fine for short parameter values on a handful of parameters per prompt; would need real incremental decoding to scale to long prompts or long free-text values.

## Challenges Faced

1. **Truncated / boundary-ambiguous values** — several different real substrings of the prompt are all individually valid — nothing forces the decoder to prefer the longest/correct one.
2. **Empty strings are structurally unreachable** — a parameter whose correct value is genuinely `""` can never satisfy the "complete" check; it always runs out the step cap instead.

## Testing Strategy

- **A self-authored exploratory input file** (`data/input/function_calling_tests.json`), used to deliberately probe edge cases outside the official corpus that the grading harness doesn't cover — a prompt missing a required argument, a number at the classic 64-bit unsigned-integer boundary (`18446744073709551617` = 2^64 + 1), etc. — to find and reason about bugs before deciding whether/how to fix them.
- **Ad-hoc, targeted verification during development** — rather than re-running the full (slow, model-loading) pipeline for every hypothesis, individual regex and conversion behaviors were checked directly in isolation.

## Example Usage

Given a functions definition file describing available functions and their typed parameters, and a prompts file:

```json
// data/input/function_calling_tests.json
[
  { "prompt": "What is the sum of 2 and 3?" },
  { "prompt": "Reverse the string 'hello'" }
]
```

Run at the root directory:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

Produces:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 2.0,
      "b": 3.0
    }
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": {
      "s": "hello"
    }
  }
]
```
