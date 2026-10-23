# Hand-built trace-export samples

These files are **hand-built samples**, not real vendor exports. They mirror
the shape each vendor documents for its export (field names, nesting, units)
over a tiny synthetic support-ticket flow: three tickets, four LLM steps each
(`intent`, `urgency`, `extract`, `reply`), so every session id groups exactly
four records. There is no PII: the tickets, order ids and replies are invented.

| file | shape | rows |
|---|---|---|
| `langfuse_sample.jsonl` | Langfuse observation export, one observation per line; 12 `GENERATION` rows plus one `SPAN` row per trace (`type != "GENERATION"` rows are skipped by the reader) | 15 |
| `langsmith_sample.jsonl` | LangSmith run export, one run per line; 12 `run_type == "llm"` rows (the `wrap_openai` shape: `inputs.messages`, `outputs.choices`) plus one `chain` row per trace | 15 |
| `openai_sample.jsonl` | OpenAI Batch API input line joined with its output line on `custom_id` (`body` = request, `response.body` = chat completion object) | 12 |
| `generic_sample.jsonl` | arbitrary column names, read through `--source jsonl` with a column map | 12 |
| `generic_sample.csv` | the same rows flattened for `--source csv` (`prompt` is a plain string) | 12 |

The tests keep the row counts as module constants so the real exports from
the dogfood agent can replace these files without rewriting the assertions.

## Local-recorder fixtures

Trace-envelope v1 lines (`dagnam_contracts.trace`), written with the recorder's own line builder
over documented provider response shapes and a fixed clock, so the ids are `rec_*` and the times
are 2026-10-12. Read by `tests/audit/readers/test_recorder.py`; regenerate them with the recorder
once it can record real calls.

| file | shape | lines |
|---|---|---|
| `recorder_openai_chat.jsonl` | OpenAI Chat Completions: plain, cached tokens, a tool call, an omitted image (`media_omitted`), a developer message with assistant history | 5 |
| `recorder_openai_responses.jsonl` | OpenAI Responses: a message, a `function_call`, an omitted `input_image` | 3 |
| `recorder_anthropic.jsonl` | Anthropic Messages: system blocks, cache usage, a thinking block, a `tool_use`, an omitted document | 5 |
| `recorder_norm_only.jsonl` | `norm` without raw (`--no-raw`) for OpenAI and Anthropic, plus a `provider: "other"` line | 3 |
| `recorder_mixed_bad.jsonl` | three 200 lines among a 429 and a 500, which the reader skips without counting them malformed | 5 |

## Token-estimate fixtures

Written for the student token estimate's tests, not trace exports. The real counts are
those of the `Qwen/Qwen2.5-0.5B-Instruct` tokenizer, taken once and kept as literals, so no
test needs the tokenizer.

| file | what |
|---|---|
| `token_estimate_pins.json` | 48 natural texts (every kind of text the estimate prices, Chinese in both variants) with their real count and the estimate that is pinned |
| `token_estimate_rows.json` | system / user / reply rows with their real count template included, and rendered chat templates with the real count of the conversation and of each content |
| `token_estimate_scripts.json` | the pinned estimate of 45 letters of each script written without spaces, alone and after a space (not natural text, so no real count) |
| `token_estimate_attacks.json` | 213 texts built to be hard (encodings, whitespace, controls, random letters of every script, long random words): an ASCII label, the real count and the pinned estimate, in the order `tests/audit/_attacks.py` builds them from code points, repetition and seeded generators; no text is stored |
| `token_estimate_scripts_natural.json` | a short natural-language text in each of Kannada, Oriya and Tibetan (the first article of the Universal Declaration of Human Rights and two sentences about a capital), with the real count and the pinned estimate |
