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
