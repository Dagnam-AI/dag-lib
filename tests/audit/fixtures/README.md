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

Trace-envelope v1 lines (`dagnam_contracts.trace`) exactly as `dagnam_record` wrote them while the
real `openai` and `anthropic` clients called a local mock server. They are **generated**: run
`record/tests/gen_fixtures.py` (see its docstring) and review the diff. Only `trace_id`
(`rec_<tag>_<n>`), `ts` (2026-10-12, one second per call) and `latency_ms` are rewritten, so a second
run gives the same bytes; `record/tests/test_gen_fixtures.py` fails when a file is stale. The bytes
depend on the locked `openai` (2.54.0) and `anthropic` (0.125.0): regenerate when either is bumped.
Read by `tests/audit/readers/test_recorder.py` and `test_recorder_roundtrip.py`.

| file | shape | lines |
|---|---|---|
| `recorder_openai_chat.jsonl` | Chat Completions: plain, cached tokens, a tool call, an omitted image, a developer message with assistant history, parallel tool calls, a history with a tool call and its result (arguments with non-ASCII text), the old `function_call` (answer and history), an empty system message, control characters, a `custom` tool call, JSON text as a system and a user prompt, an assistant turn with `<think>` | 14 |
| `recorder_openai_stream.jsonl` | `stream=True`: Chat text with usage, Chat with two tool calls, a Responses stream | 3 |
| `recorder_openai_parse.jsonl` | `parse()` with a pydantic model: the recorded schema (`response_format`, `text_format`) and the `parsed` fields | 2 |
| `recorder_openai_responses.jsonl` | Responses: a message, a `function_call`, an omitted `input_image`, a history with a `function_call` and its output, an empty `instructions` beside a developer message, a reasoning item, a `custom_tool_call`, a history with a custom tool call and its output and no `instructions` | 8 |
| `recorder_anthropic.jsonl` | Messages: system string and blocks, all three cache counters, thinking, `tool_use`, an omitted document, parallel tool use, a thinking-only answer, a tool-use history, a web-search answer, an empty system, control characters, a thinking history, a reply of two text parts the second opening with `<think>`, a reply that is JSON of a message | 14 |
| `recorder_anthropic_stream.jsonl` | `stream=True`: text, parallel tool use, thinking then text, a tool call cut at `max_tokens` | 4 |
| `recorder_media.jsonl` | prompts with media of each kind (`media_omitted` markers in place of the payloads) | 5 |
| `recorder_norm_only.jsonl` | `no_raw=True`: `norm` without the bodies, for OpenAI chat, Anthropic and Responses, and a chat history with a tool call | 4 |
| `recorder_mixed_bad.jsonl` | three 200 lines among a 429, a 500, a dead connection (status 599) and a 529, which the reader skips without counting them malformed | 7 |
| `recorder_other.jsonl` | two `record_call()` lines the recorder writes no norm for (`provider: "other"`, and OpenAI with endpoint `custom`) | 2 |
| `openai_roundtrip_batch.jsonl` | the 14 calls of `recorder_openai_chat.jsonl` as an OpenAI Batch export: the request as the mock server received it on the wire (media in full) and the response it sent | 13 |

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
