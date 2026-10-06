# Changelog

All notable changes to the `dagnam` Python SDK are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

The changes below are `__version__` 0.16.0, not yet released: the release folds
them into a dated `## [0.16.0] - YYYY-MM-DD` heading (see `RELEASE.md`).

Needs the platform release that runs `dagnam-contracts` 0.4.1 and has
`POST /api/v1/audits/{id}/resume`, the run read's `model_version_id` and
`credits_consumed`, the `resume` publish field, a workload publish body that
accepts `response_mode` and agreement blocks that carry `min_class_recall`, and
the recipes `head-tune-text-classification@1.2` and `qlora-sft-chat@1.2`.
**Deploy the backend first.** Against an older platform the server's PII check
lacks `PII_SECRET`, so every workload stops at `pii_disagreement`; its strict
schemas refuse the new publish fields; a completed run is `no_model_version`;
and a candidate's submit names a recipe version the platform does not have.

The teardown changes below read the platform's newer receipts (`audit_status`,
`code` on every row, an `audit_id` accepted by the create routes, a repeat delete
answered with the stored receipt) and degrade as follows against an older
platform: `audit run` stops at the platform check; `audit cancel` and `audit
delete` read its receipts by status and the four legacy reasons, treat a legacy
`blocked` row as left (exit 1), make no direct call, and read its 404 on a repeat
delete as "deleted elsewhere" (exit 0, no receipt available).

### Changed

- **Breaking: the platform's cancel and delete are the only paths that touch a
  published audit's resources.** Once `state.json` holds an `audit_id`,
  `dagnam audit cancel` and `dagnam audit delete` ask the platform once (a long
  read timeout, no retry into a second walk), print its receipt as sent, record
  the decisions, remove this machine's own files, and exit. They make no
  destructive call of their own: a platform that does not answer (a 5xx, a
  timeout), refuses, or has no such audit (404) is never "finished from here".
  A failure writes an `audit blocked [not_answered]` row, removes nothing local,
  and exits 1 -- run it again. A 409 "another teardown is running"
  (`teardown_in_progress`, top level or under `detail`) and a 503 "the lock is
  unavailable" (`teardown_unavailable`) are waited out (the platform's
  `Retry-After`, any length up to two minutes in all; five seconds when it sends
  none or a value that is no number of seconds) and asked again. A 404 decides nothing: it
  says only that this key cannot see the audit (another account's key, another
  host, a deleted audit). The command says which host and masked key it asked,
  keeps every local row, the deployment keys and `state.json`, and exits 1; only
  the platform's positive answer marks the audit deleted, and `audit delete
  --already-deleted` is how a person says it is gone. The same holds for an
  unpublished audit: a walk in which every id answers not-found changes nothing
  local. `audit run` and `audit cancel` refuse a deleted directory, and a run that
  meets a 404 halts as `audit_not_found`, which is not the deleted state. Direct
  calls remain only for an audit with no platform record (`--local-only`, or a run
  that never reached `create_audit`), and then only for ids this directory
  recorded as created, plus what the platform refused to claim (below). This removes the retry loop, the "held" and "account failed" paths and
  the read-back of a refused stop from `dagnam.audit.cleanup`
  (`delete_audit(audit_dir, client)` and `cancel_audit(audit_dir, client)` now
  take no receipt arguments). A `blocked` row is never retried from here, whatever
  its code; `audit delete` with the platform answering `audit_status: halted`
  removes nothing local and exits 1, and the next delete asks again.
- **Breaking: `dagnam audit cancel` always answers with a receipt.** A local
  cancel writes `cancelled.json` too and prints (or, with `--json`, emits) the
  same `{schema, deleted_at, entries}` receipt a published cancel does. The
  receipts of an unpublished audit's delete use `entries` as well (they were
  `items`); older `deleted.json` files still read. A row that says `stopped`
  marks the local run or endpoint; `stopped` with the code `already_stopped`
  (the run had already finished, or the endpoint was paused) marks nothing, so a
  finished run stays resumable. Exit 1 when any row is `blocked`, a row's status
  is one this version cannot read, or the platform did not answer. Cancelling a
  deleted audit is refused. `dagnam.audit.cleanup.mark_cancelled` takes the ids
  that stopped and the ids found gone.
- **Delete and cancel read the platform's receipt by `status` and `code`, never
  by the wording of a reason.** The closed table of codes (`deleted`,
  `already_absent`, `stopped`, `already_stopped`, the `kept` codes
  `project_not_ours`, `project_shared`, `project_held`, `entry_shared`,
  `weights_served`, `not_in_project`, `not_created_here`, and the `blocked` codes
  `weights_not_removed`, `has_served`, `refused`, `failed`) is decided in one
  place, `dagnam.audit.receipt_rows`, and the exit status in one function,
  `exit_status`: 1 if and only if something of the audit's own is left. An
  unknown code on a known status is read by the status; an unknown status is
  shown, never acted on, and exits 1. `kept` is a decision to leave something
  that is not the audit's to take: it is recorded in `state.json` (`kept_ids`) so
  no listing offers it as the audit's, and it never fails a command. A platform
  that sends no `code` and no `audit_status` (older) is read once, in
  `receipt_rows`, from the status and the four reasons it used; a legacy
  `blocked` row is a leftover (exit 1, nothing retried). Rows this client writes
  carry `not_answered` (the platform did not answer), `not_removed` (a local
  file, or an unpublished id that would not go) and `platform_only` (a registry
  version this client cannot remove because the platform has no route for it).
- **`dagnam audit run` creates the audit before it uploads anything.** Every
  resource a published run creates (dataset, training run, endpoint) is sent with
  the audit's id, so the platform tags it at creation and the audit's delete finds
  it whatever any step later says. A `create_audit` that fails now halts the run
  (`halted: publish_failed`, exit 1) with nothing uploaded; run it again. It was a
  warning, and the rows uploaded meanwhile belonged to no audit. A body whose
  answer was never heard (a timeout) is re-sent as written; one the platform
  answered is dropped. A directory that ran `--local-only` and is then published
  hands its recorded datasets, runs, endpoints and project to the new audit
  (`POST /audits/{id}/claims`, at most 200 per request; a version follows its
  run). A claim the platform refuses is not "kept": this directory made the
  resource, so `audit delete` removes it itself once the platform has deleted the
  audit and `audit cancel` stops it (the run says so). The exception is the
  refusal code `in_use_elsewhere` (a dataset another of the owner's runs uses, or
  the run that made it): that is recorded as kept, shown, and never touched from
  here. What the platform's own receipt has a row for is never walked from here,
  and a refused id this client cannot remove is named once without holding the
  directory out of the deleted state. A claim request that fails as a whole halts
  the run (`publish_failed`), tells the platform the run stopped, says that the
  earlier run's resources still exist, and is asked again. A body the
  platform did not refuse (a timeout, a 502, the in-progress 409) is re-sent as
  written; one it refused (400, 404, 422) is dropped. A dataset an earlier
  candidate, retired ones included, already recorded is never adopted by another;
  a forced rescan drops the create body it had saved.
- **A published run adopts a lost upload only when the listing row proves it.**
  The `audit_id` filter on the dataset listing is not trusted: a platform that
  ignores it lists every dataset the caller can see, another audit's and other
  accounts' public ones included, and adopting by name would train this audit on
  the wrong rows and upload nothing. A row is adopted only when it carries this
  audit's id, belongs to the audit's owner, bears this name, and holds this file
  (the stored size, and the sample count once analysed: a dataset resource carries
  no content digest). A platform that returns no `audit_id` never adopts; the rows
  are uploaded afresh.
- **An audit an older dagnam published is offered a claim before it is deleted.**
  The platform keeps what it cannot prove the audit created; `audit delete` asks
  once, claims what the directory recorded, and then deletes. Declined, refused,
  or a claim that cannot be made at all (no such route, no answer), it says so
  and the delete goes on, naming which resources the platform kept and why.
- **An unpublished audit never deletes what someone else uses, and finishes.**
  A dataset linked into a project this directory did not create is `kept
  [in_use_elsewhere]` (the owner's projects are read to check; an answer of the
  wrong shape, a failed read, or more than 100 pages of projects, and the dataset
  is `blocked`, never guessed at; a dataset that is gone needs no check) and recorded in `state.json` so no later
  walk or cancel touches it. A purge the platform refuses with `409
  weights_served` (`status: kept`) is `kept`, not `blocked`, and the delete
  finishes around it (the client now keeps the response's parsed `detail` on the
  new `VersionKeptError`; it used to hand over only its message). A walk
  changes nothing local, and marks nothing, unless its key is shown to see the
  account: a deletion this same key and host confirmed earlier (`confirmed_gone`,
  scoped by `confirmed_by`, a digest of host and key), a delete or stop that just
  succeeded, a keep the platform's own purge answered, or a read of a recorded
  deployment or training run. A project, dataset or model version can be public,
  and a dataset can be linked into another account's project, so reading those,
  or finding a dataset in use elsewhere, proves nothing (and a dataset found in
  use by a walk that proves nothing is not recorded as kept).
  Another account's key is told "not found" for everything, so not-found is never
  proof, and neither is a walk where the rest failed (a 5xx on one id used to be
  taken as an answer and released the deployment keys). An id whose delete and
  re-read both failed, or whose delete said "not found" while it still reads back
  or cannot be read, is `blocked [not_answered]` and keeps the keys and rows;
  a delete's single "not found" is now read again before it is believed. A
  damaged `state.json` is a named error (`StateFileError`, still a `ValueError`)
  instead of "unexpected error". The teardown 409/503 error carries the backend's
  `detail` words. When it fires, the receipt and the error name
  `audit delete --already-deleted`, whose help now says it is irreversible, and
  whose note no longer says "nothing was changed" while it removes the files.
  The error line no longer counts an unanswered audit as an artifact still there.
  An unpublished run that finds its lost upload by the directory's key adopts it
  only when the dataset belongs to the owner of this directory's project and the
  listing's size and sample count say it is this run's file (a project read that
  is not a project uploads afresh and says so, instead of ending the run);
  otherwise it uploads afresh and logs which dataset it left behind.
- **Smaller teardown fixes.** `audit delete` of an audit an older dagnam
  published goes on to the delete when the claim fails (it used to stop, and a
  platform without the claim route could not be deleted from at all) and does not
  offer a claim for a directory already deleted. A second `audit cancel` of a
  finished 0.15 audit exits 0 (the endpoint's "Invalid status transition from
  paused to paused" is `already_stopped`). A `Retry-After` of `nan` no longer
  crashes the wait.
  A response body nested thousands of levels deep is treated as an unreadable
  body (a malformed response, or no marker) instead of ending the command with
  `RecursionError`. An empty-bodied 409 on a version that still reads back says
  "the platform gave no reason" on its row instead of claiming the delete was
  reported done. Platform words copied into a receipt row or a teardown wait
  error are cut at 2,048 characters with a note of the full length.
- **Rows are redacted last.** Each training row is redacted before the character
  budget cuts it (so a cut can never expose part of an identifier) and again on
  the cut row, exactly as it will be uploaded, with the contract's `redact_rows`
  (`dagnam-contracts` 0.4.1): the uploaded row scans clean, a second pass changes
  nothing, and the cap still holds. Text a cut leaves next to an identifier (digits
  run together, say) used to ship; it is now redacted and counted. Redaction scans
  each row whole, system prompt included: about two milliseconds more a record
  than 0.15's cached system prompt when every call repeats a long one (2,000
  records, 3 KB system prompt: 0.9 s before, 4.6 s now; the same either way when
  each record's prompt differs).
- **A registry version is removed through its own route, never through its
  entry.** For an unpublished audit the version's weights are purged with
  `DELETE /api/v1/model-versions/{id}` (`DagnamClient.purge_model_version`, which
  returns the platform's receipt row); deleting the whole registry entry around it,
  and with it versions the audit never made, is gone. A purge that fails after the
  row was revoked is not taken for gone (the version stays `blocked`). A platform
  without the route reports the version `blocked [platform_only]` and exits 1.
- **`audit cancel` and `audit delete` of an unpublished audit.** An endpoint that
  cannot be paused is a `blocked` row (exit 1) unless the endpoint itself says it
  is already paused or stopped (read back once; then it is `stopped
  [already_stopped]`). The project stays (`kept [project_held]`) while anything in
  it was not removed. The async client no longer has `cancel_audit` or
  `delete_audit`: a teardown is the sync CLI's.
- **Smaller changes.** An `audit_status` this client does not know (not `deleted`,
  not absent) is exit 1 and removes nothing local; a receipt row that is not an
  object is shown as `?` and fails the command; `deleted` beside a deployment row
  that is still there keeps the deployment's key; a platform without `code` that
  says a run "Cannot cancel job with status ..." is read as `already_stopped`. A
  published run whose endpoint never activates no longer pauses it itself (the
  audit's cancel does). A failed `create_audit` prints its cause and "nothing was
  uploaded or spent: run it again"; an answer that lacks a field says which one
  instead of a Python error name; `audit status` survives a failing metrics read.
  A poll that fails for thousands of attempts in a row no longer overflows its
  backoff.
- **Every create with an `Idempotency-Key` reads a body-less replay back.** When
  the platform replays a completed create whose body it did not keep (only a
  marker with `resource_id`, and a `Location` when the original had one), the
  resource is read by that id -- for audits, foundation runs, deployments,
  projects and training jobs -- and one the owner deleted since steps the create to
  its next key instead of returning it. The "in progress" 409 is recognised by
  its exact text or its `"error": "idempotency_in_progress"` marker, never a looser
  match.
- **The unpublished delete lists the owner's projects oldest first and trusts only a steady
  list.** A project touched while the walk ran used to shift between pages and could be
  missed, and its dataset deleted. The list now also has to report the same `total` on every
  page and on a last read of the first; a changed total counts as "could not check" and the
  dataset is kept.
- **A 503 "idempotent creates are unavailable" is waited out for as long as the platform
  asks, up to 60 seconds a wait.** The create still gives up after three retries. A
  `Retry-After` of `nan`, a negative number or `inf` on any retried request is ignored or
  capped instead of ending the command with a `ValueError`.
- **The publish carries exact token totals.** Each workload sends
  `completion_tokens_total` and `calls_total` (the sums its local price is
  computed from) beside the rounded means, so a platform that prices through the
  contract function reproduces the CLI's price exactly. A platform that refuses fields it
  does not know (production's strict bodies) refuses the audit create: the run halts before
  anything is uploaded, and the platform check normally stops it sooner.
- **`audit` commands resolve the directory argument once.** A directory named
  through a link (a temp or home directory) works; a link inside it is still
  refused. The files a run reads from the directory are read only if they are
  regular files.

- **Breaking: workloads without a system prompt get new ids.** They fall into
  `unstructured-json`, `unstructured-short` and `unstructured-long` by the shape
  of their answers instead of one mixed `unstructured` workload. Template
  normalization now also masks weekday and month names, clock times and
  ordinals, so a dated system prompt is one workload (and one id); a one-line
  `{...}` schema keeps its keys, so different extraction tasks stay apart.
  Workload ids are not validated as hex anywhere.
- **Breaking: `--max-credits` is a hard ceiling.** `dagnam audit run` never
  submits a candidate whose own cost could take the credits spent past it, and
  never starts a replay that could: a run is budgeted at the most its recipe can
  charge (an hour at 2 credits a minute, 120 credits, since the platform offers
  no pre-submit estimate), and a replay at one credit per holdout row plus 10%.
  The old check compared against the dearest candidate *so far*, so the first
  submit and every replay were never checked: `--max-credits 10` still submitted
  a 300-credit run. A run still going counts at no less than its ceiling; a
  replay whose cost was never read counts at its projection, and an interrupted
  one at the rows it had answered.
- **Breaking: without `--max-credits` the ceiling is the plan's own estimate**:
  what the audit already spent plus every selected candidate still to finish,
  rounded up to 100, printed in the listing you confirm. It was a flat 500.
  `dagnam.audit.orchestrate.DEFAULT_MAX_CREDITS` is gone;
  `plan_credits(audit_dir, selected, state)` computes the default, and
  `dagnam.audit.steps_train.credits_spent` returns one number.
- **A cancel or a delete on the website stops a live `dagnam audit run`.** Each
  run resumes its audit once, before it waits on anything
  (`POST /api/v1/audits/{id}/resume`), and every publish after that says
  `resume: false`. From then on the next publish after a cancel gets the 409
  "audit is halted", and the next one after a delete the 404 of an audit that
  is gone: the run stops with `halted: cancelled` or `halted: deleted`, and runs
  and publishes nothing more -- nothing is paid for after the next step. A
  delete that lands while the run waits on training or a deployment ends it as
  `deleted` too, not as an error, and a later run over a deleted audit stops
  before its first step. On a platform without the resume route the run's first
  publish resumes the audit instead, as 0.15 did. A halt the run publishes
  never overwrites the account's own `cancelled`.
- **Tool-calling workloads are audited on their tool calls.** `dagnam audit
  scan` trains and scores a tool-calling workload on its call's name and
  arguments as one `{"name", "arguments"}` JSON object -- the ordered list of
  them when the teacher made several calls in one reply -- marks it
  `response_mode: "tool_call"` (published with the workload), and says, in the
  scan's warning, the run report's switch and `dagnam audit run`'s output, in
  one sentence: "This workload answers with tool calls: the replacement returns
  the call as {"name", "arguments"} JSON (an ordered list of them when the
  teacher made several) in message.content, not in tool_calls." The contract
  scores such a call per row: a wrong tool name scores 0 whatever its arguments.
- **Breaking: a named step is its own workload.** A trace's `workload_hint`
  (`metadata.workload`), else its registered prompt's name (Langfuse
  `promptName` / `prompt_name`, a Responses request's `prompt.id`), keys its
  workload ahead of the system prompt; the
  scan report carries it as `name`, and the id is a hash of it. The name of the
  request's structured-output schema, or of the tool its `tool_choice` forces
  (OpenAI, Anthropic, LangChain's `with_structured_output`), joins the template
  key, so two extraction tasks under one system prompt stay apart; the tools a
  request offers do not, since an agent's tool set changes from call to call.
  Ids of named, schema'd or forced-tool workloads change.
- **Breaking: earlier assistant turns stay in the prompt as context**, each
  with the tool calls it made (as the JSON the student answers with), so a later
  step's tool result answers a call the student can see. Tool results sent as
  content parts (Anthropic `tool_result`, Gemini `functionResponse`, Vercel
  `tool-result`, OTel `tool_call_response`) are read instead of blanked.
- **Candidates train with `head-tune-text-classification@1.2` and
  `qlora-sft-chat@1.2`**: a label absent from train is a miss instead of a
  crash, and the SFT trains on the completion only and drops, never
  right-truncates, a row over its token budget.
- **A candidate trains on at most 5,000 rows** (`MAX_TRAIN_ROWS`), a
  proportional sample by target (a label, or a router's route) chosen by
  content, so it finishes inside its recipe's 1-hour ceiling instead of being
  killed after an hour of paid GPU. The holdout is kept whole. The scan report
  counts the rows the cap dropped (`dataset.train_rows_capped`) and warns.
- **An SFT candidate's training rows must fit the student's 2,048 tokens.**
  `qlora-sft-chat@1.2` drops a longer row at train time, after the credits are
  spent, so the scan estimates each training row's tokens and leaves out rows
  estimated over the budget: the chat template's own tokens exactly (5 a message
  and 21 for the default system turn) and an estimate of each message's text.
  The scan report counts the rows left out (`dataset.train_rows_over_budget`) and
  warns; the holdout keeps them. Under 32 training rows left, the workload is
  `too_few_samples`: "N rows exceed the student's 2,048-token context".
  The estimate needs no tokenizer and guesses no language. The student's
  tokenizer normalises its input to NFC and, being a byte-level BPE, spends on a
  piece at most as many tokens as the piece has UTF-8 bytes in that form. The
  estimate does the same normalisation, starts every character at its bytes,
  discounts it only where the student's vocabulary shows a merge exists, each
  discount with a limit, and caps every piece at its bytes: a character that is a
  token costs 1; a word that is a whole-word token costs 1, but only up to the
  longest such token, so a long word never consults the table; any other word
  costs a line in its length up to the length natural words reach, and the lowest
  price a random string of letters measured after that; han is priced by the
  longest match against the vocabulary's tokens of 2-4 characters, symbols by the
  longest match of up to 3, and a unit the tokenizer does not merge again when it
  recurs (678 of them, measured) costs its characters when seen again within 8
  units; whitespace by the period measured for each kind up to 4,096; a script
  written without spaces at the rate measured on natural text of that script; and
  what the vocabulary does not hold at its bytes. It was fitted to the real Qwen2.5
  tokenizer on a 5,251-row corpus in 77 languages. What was measured, and all that
  is claimed:
  on natural text of the measured kinds it is at least 0.95x the real count:
  0.954x at the lowest of 2,498 held-out rows of 500 or more real tokens, 1.04x at
  the median and 1.22x at the 95th percentile. The measured kinds are the Latin,
  Cyrillic, Greek, Armenian, Georgian, Hebrew and Arabic alphabets; Simplified and
  Traditional Chinese, Japanese and Korean; Thai, Lao, Khmer and Myanmar;
  Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada,
  Malayalam and Sinhala; Ethiopic and Tibetan; and code, JSON, identifiers, tables
  and emoji (Gurmukhi, Gujarati, Lao, Sinhala and Ethiopic on a handful of rows).
  Scripts with no natural text in the corpus (Syriac, Thaana, N'Ko, Mongolian,
  Cherokee, Canadian syllabics, Tifinagh, Vai, Bamum, Javanese, Balinese,
  Sundanese, Tai Le, Tai Viet, Lisu, Yi, Ol Chiki, Osmanya, Deseret, Gothic,
  Glagolitic and Coptic) are priced at the safe byte price for every letter that
  is not a token. By kind, median and 95th percentile: English 1.05x and 1.10x;
  lists of labels and terse prompts 1.17x and 1.21x; Spanish, French, German,
  Portuguese, Italian, Indonesian and the like 1.20x and 1.26x, other Latin-script
  languages 1.03x and 1.18x; Cyrillic 1.07x and 1.21x; Greek 1.03x and 1.04x;
  Arabic script 1.08x and 1.14x; Hebrew 1.04x and 1.08x; Indic scripts 1.01x and
  1.05x; Tibetan 1.02x and 1.04x; Thai, Lao, Khmer and Burmese 1.03x and 1.08x;
  Armenian, Georgian and Ethiopic 1.01x and 1.12x; Simplified Chinese 1.06x and
  1.08x; Traditional Chinese 1.05x and 1.07x; Japanese 1.03x and 1.10x; Korean
  1.05x and 1.12x; JSON 1.04x and 1.07x; code 1.12x and 1.26x; identifiers, ids and
  hashes 1.04x and 1.15x; numbers and tables 1.01x and 1.02x; emoji 1.00x and
  1.05x. On random or crafted text there is no such
  floor: it can count about half the real tokens (0.486x at the lowest on random
  Cyrillic words, 0.504x on random Thai, 0.547x to 0.62x on a repeated number sign
  or random Tibetan, 0.75x on licence keys, 0.88x to 0.90x on base32 over plausible
  text), and 0.36x on crafted alternating han units, so a row estimated to fit the
  2,048-token cap can exceed it by up to about 2.8x on crafted text. Pairs of han
  units that break each other (two units outside the fragile table, alternated:
  0.36x) and cycles of 9 or more different fragile han units (0.68x to 0.82x) are
  not seen by its 8-unit window, a known limit. It is never above the UTF-8 byte count
  of the NFC form of the text. It leans high where it cannot be right, so it can
  reject rows that fit: code with long identifiers up to 3x, a symbol repeated
  thousands of times up to 21x, a repeated fragile 4-character han unit 4x, a text
  of a few tokens rounds up (a one-word text can count two). A character that is a
  token on its own but splits beside one neighbour (one currency sign after a
  Hangul syllable, 0.50x) is not modelled. If every row only appears to fit,
  training can still fail after model startup consumes credits. Memory is flat in
  the size of the text, apart from the three vocabulary tables (about 13 MB once in
  use), the normalised copy of text that is not already NFC, and a few copies of
  one unbroken word.
  Three tables ship inside the package, each checked by length and SHA-256 at first
  use, and the estimate refuses to run on a missing or changed one:
  `dagnam/audit/student_words.z` (the 49,926 whole-word tokens of
  `Qwen/Qwen2.5-0.5B-Instruct`), `dagnam/audit/student_chars.z` (its 18,177
  single-character and space-and-letter, 16,382 han and 5,078 symbol tokens) and
  `dagnam/audit/student_fragile.z` (678 han and symbol tokens it does not merge
  again when they repeat), each an exact list of keys from the model's vocabulary
  (Apache License 2.0), sorted and compressed (191 KB together), decoded into sets
  on first use. `scripts/make_student_tables.py` rebuilds them from the model's
  `tokenizer.json`, with a small byte-level BPE of its own that matches the real
  tokenizer on the units the tables need, not on text in general (it counts
  numerals such as fractions as letters, which can only over-count).
- **A workload whose calls carry images, audio or files is `not_audited`**
  ("multimodal input") once more than 10% of its calls do: a text student
  cannot see what its answers depend on.
- **No token usage and no cost is `unknown_cost`, never a $0 teacher** (a
  stream without `include_usage`, say); calls without usage are extrapolated
  from the priced ones when some have it, within one model's calls too.
- **`dagnam audit scan --sample-rate 0.1`** (or `10%`) scales the volume and
  spend of an export that holds a sample of the traffic back up; the scan
  report's `window.sample_rate` records it.
- **Secrets are always redacted** (`PII_SECRET`, from `dagnam-contracts` 0.4.0):
  provider keys, JWTs, bearer tokens, private-key blocks and credential
  assignments, quoted keys included, from the rows, the template excerpt and
  tool-call arguments. Redaction runs before truncation and again after it, and a redacted JSON
  target stays valid JSON; the scan report counts, and warns about, training
  targets that redaction rewrote.
- **An unreliable candidate never wins.** Every candidate in `audit-report.json`
  carries `unreliable` (more than 10% of its replay calls failed). It still shows,
  with its status, but earns no winner, no switch and no savings. Each candidate
  is held to the floor its own agreement recorded.
- **`verdict.savings_usd_month` in `audit-report.json` counts a REPLACE only**,
  as the spend less the winner's serving cost less maintenance, never below zero.
  Every other workload reports 0.
- **The model version a run pushed comes from the run.** The run read's
  `model_version_id` is recorded as it is; the "newest version this account
  pushed since the run started" fallback, which could deploy another run's model
  (a Studio retrain, say), is gone. A completed run that does not name one is
  `no_model_version`.
- **A run's training cost is what the platform charged**, from the run read's
  `credits_consumed`, for a failed run as much as a completed one; the estimate
  stands only until then.
- **Pricing.** `short_span` workloads are priced as the GPU SFT student the run
  trains, and a token-priced student with no completion tokens is
  `unknown_cost` instead of $0. Each spelling of a model is priced on its own,
  and calls on an unpriced model are extrapolated from the priced ones instead
  of making the whole workload `unknown_cost`. The scan report lists workloads
  by monthly spend once they are priced.
- **`DagnamClient.create_project` and `create_audit` (and their async twins) send
  an `Idempotency-Key`**, so a transient failure retries into the platform's
  replay of the first answer instead of a second, orphaned project or audit.
  New client methods `get_audit` and `resume_audit` (sync and async).
- **One command at a time holds an audit directory.** A second `dagnam audit
  run`, a `scan` into it, or an `audit delete` under a live run fails with
  `dagnam.audit.state.AuditBusyError` instead of paying for every run twice or
  swapping rows under it. `audit cancel` under a live run cancels in the account
  (which stops the run at its next publish) and says to run it again afterwards;
  under a live `--local-only` run it says to stop the run first. A mistyped
  directory is refused, never created.
- **`dagnam audit scan` requires a new output directory when changed rows
  belong to a published audit**, including when `--force` is passed. Before
  publication, `--force` retires stale run state and replay answers, preserves
  remote resource handles for cleanup, and carries spent credits into the fresh
  run. Re-scanning an unchanged export preserves the current run state.
- **Requires `dagnam-contracts` 0.4.1.** Its `redact_rows` is the redaction
  every row goes through; its `winner_of` skips unreliable
  candidates and is order-independent, and the JSON agreement interval is
  computed over scored rows, so it is wider than before.
- **`dagnam-contracts` is capped below its next minor (`>=0.4.1,<0.5`).** The
  contract's PII class list and score shape are a wire protocol with the
  platform, so a new contract minor now reaches users only through an SDK
  release built against it. An open floor let a newly published contract into
  installs of an already released SDK whose platform still ran the old one, and
  every audit then stopped at its PII check.
- **`dagnam audit run` asks the platform which contract it runs before it
  uploads anything** (`GET /health/build`, the `contracts` key), under
  `--local-only` too: that skips the mirror into your account, and the rows are
  still uploaded to, and trained on, the platform. A platform a contract minor
  behind this SDK, or too old to say, stops the run with `platform_too_old`,
  both versions, nothing uploaded and no credits spent; so does an answer that
  is not a build document at all. A platform at or ahead of the SDK passes
  silently, except that one a *patch* ahead warns first and names
  `pip install -U dagnam-contracts` (a patch changes what is found inside
  existing classes); a request that fails outright does not stop the run. The
  check is part of `dagnam.audit.orchestrate.run_audit` and `run_audit_held`
  themselves, so the library entry point makes it too, and raises
  `dagnam.audit.preflight.PlatformTooOldError`; both also set `resume_creates`
  on the client. `dagnam.cli.audit_run.platform_too_old` is gone.
- **A PII stop names its cause and its cure, on the terminal too.** The stop is
  unchanged; `pii_disagreement` now says which of three things it is: the
  platform runs an older privacy contract (it scanned fewer classes and found
  nothing), the platform runs a newer one than the rows were redacted with (it
  found matches only in classes the scan did not look for: upgrade
  `dagnam-contracts` and scan again), or a real disagreement in a class this SDK
  redacts. When the platform's contract differs from this install's at any level
  the message names both versions and which side to move. Every one says the
  uploaded rows are still there and that `dagnam audit delete` removes them --
  and, since there is no narrower form, the whole audit with them. The message
  keeps its cause in the first 500 characters (the platform stores no more),
  lists at most three classes, and names no path of your machine (it is stored
  in your account). `dagnam audit run` prints each stopped candidate's reason under its
  workload's line, and `audit-report.json` and `audit-report.md` carry it (each
  candidate's new `error` key, `null` when nothing stopped it); it was only in
  `state.json`.
- **Breaking for scripts: `dagnam audit delete` exits 1 when something of the
  audit is left on the platform.** The receipt is written and printed first,
  exactly as before (under `--json` it is still the only thing on stdout). Then,
  if a row on the audit's own resource is `blocked` -- the platform refused it,
  or the delete failed -- the command says how many are left on stderr and exits
  1. A row the platform keeps on purpose because it is not the audit's to remove
  (a project that holds your other work, a registry entry or weights another
  deployment still serves) is `kept`, shown, never touched from here, and does
  not fail the command. The last line says which case it is: everything
  deleted, everything deleted except what was kept on purpose, or what is
  left. It exited 0 whatever was blocked. A file of the audit on this machine
  that could not be removed (a link, which is never followed) counts as left
  too.
- **`dagnam audit status` lists the candidates a forced rescan retired**, as
  `retired` rows under the workload and candidate they had been, and both it
  and `dagnam audit scan` name every retired run or endpoint that is still live
  with the command that stops them (`dagnam audit cancel <dir>`). The status
  JSON gains `audit_dir` and `retired_live`. In `state.json` a retired step now
  records its `workload_id` and `kind`; a state written before this still
  loads, and shows those rows unnamed.

### Added

- **`dagnam audit scan` reads what the tools export.** The OpenAI Responses API
  (the Agents SDK's default) through the `openai` reader and in LangSmith and
  Langfuse runs; Langfuse's blob export (`observations_v2/`, snake_case, JSON
  strings) and its UI CSV/JSON exports, whose usage buckets are summed as
  Langfuse defines them; `.json` array files; gzipped JSON and JSONL, read as
  they stream instead of inflated to disk; `--map` columns as dotted paths into
  nested records (`input.inputBodyJson.messages`, for a Bedrock invocation log);
  epoch timestamps in seconds, milliseconds, microseconds or nanoseconds, as
  numbers or in text (a basic ISO date like `20260927` stays a date); Anthropic
  `tool_use` and Gemini `functionCall` replies as
  tool calls. Bedrock `global.` / `apac.` / `jp.` / `au.` / `ca.` inference
  profiles, Vertex `@date` ids and resource paths, Gemini `-001` versions and
  Fireworks `accounts/*/models/*` ids price as their model.
- **`DagnamClient.get_platform_build()`** reads `GET /health/build`.
- **`DagnamClient.resume_creates`.** When set, the creates of a foundation run,
  a deployment and an audit send an `Idempotency-Key` derived from the request
  instead of a random one, so the same request asked again inside the
  platform's 24-hour replay window returns what the first one created. A
  refusal the platform replays is stepped past, so an ask whose cause was fixed
  is evaluated afresh. A replayed success is read back first, and one that names
  something the owner has deleted since is created afresh. An "in progress" 409
  (a rerun typed while the platform still holds the marker of a run that was
  interrupted) is waited out for up to two minutes under the same key -- only
  that answer: the `error` marker `idempotency_in_progress`, or exactly the text
  `A request with this Idempotency-Key is in progress` that older platforms send
  alone, never another 409. A replay that dropped its body and points at the
  resource it made (its `Location`, or `resource_id`) is read back, for audits,
  foundation runs and deployments. The
  audit create is keyed by a nonce saved in `state.json` with its body, so a
  rerun with another `--max-credits` or `--floor` finds the audit the first ask
  made instead of opening a second; a dataset upload, which the platform cannot
  replay, is found again by that nonce in its description and adopted instead of
  uploaded twice. Off by default; `dagnam audit run` turns it on. Keys are
  scoped by the credential on the platform, so a rerun after `dagnam login` with
  another key does not replay, and nothing replays after 24 hours.
- **A release gate.** `scripts/check_platform.py` exits non-zero unless the
  platform reports a `dagnam-contracts` version at or above the floor in
  `pyproject.toml`; the release workflow runs it, and the tagged commit's own CI,
  before it builds anything. The check names itself in its `User-Agent`, has a
  timeout, and refuses a redirect to another host.

### Fixed

- **Reasoning is never taken for the answer.** Gemini `thought` parts,
  Anthropic `thinking` blocks, Responses `reasoning` items and a leading inline
  `<think>...</think>` are dropped from replies (their tokens stay priced).
- **Chinese, Japanese and Thai replies are measured in tokens, not
  whitespace words**, so a long reply is free text instead of a short span.
- **`dagnam audit scan` streams the export.** It no longer holds every record:
  discovery reads the export once, and the rows of the workloads worth auditing
  are planned and then derived in two more passes that keep only those rows. On
  a 200 MB export the peak memory fell from 890 MB to 315 MB (692 MB to 90 MB
  when nothing is audited), and the scan got faster. Discovery keeps each
  template's masked excerpt, never its system prompt.
- **The holdout replay reuses its connections**: one keep-alive pool per replay
  instead of a new TCP and TLS connection per call, so the latency it reports is
  the model's and the gateway's, not a handshake's.
- **A text `outcome`** (`metadata.outcome: "resolved"`) is ignored instead of
  making its row malformed.
- **A Langfuse reply that is itself a JSON object** (`{"type": "refund",
  "amount": 12}`, `{"name": "Bob", "role": "admin"}`) is read as that answer, as
  JSON text, instead of as an empty message that left its workload without a
  single row; a reply is what carries one (`content`, `parts` or `tool_calls`), so
  a LangChain `AIMessage` a generation returned is read as its content.
- **A Ctrl+C during a slow publish no longer loses a paid step.** The state is
  saved before anything is published, so the rerun resumes a submitted run
  instead of paying for a second one that no `cancel` or `delete` could find.
- **The last step of a run reaches the account.** What is queued is flushed
  before the run ends, and each candidate remembers the last step the account
  acknowledged, so a patch a blip or a Ctrl+C stranded goes out on the next run
  instead of leaving the audit page stuck. A patch naming an artifact the owner
  deleted is dropped instead of blocking every later one.
- **An interrupted replay resumes where it stopped.** Each answer is written down
  as it lands (`workloads/<id>/replay-<kind>.jsonl`), so the rerun sends only the
  rows that never answered or failed, and the replay's cost still counts what
  the interrupted attempt burned.
- **A damaged replay file no longer breaks every budget check.** One whose head
  a crash cut short is not trusted: its answers still count toward the budget,
  every row is sent again, and the replay's cost is unknown (counted at its
  projection). The head is now written atomically.
- **The report says why a workload has no winner.** A candidate that failed to
  deploy, or is still training, measured nothing: "No candidate was measured in
  this audit (sft_small: deploy_failed)", not "No candidate cleared the floor".
  `dagnam audit run` prints the same reason beside `NOT YET`.
- **Prompt-cache reads are priced at the cached-input rate** (OpenAI, Anthropic,
  LangChain and Gemini spellings), and Anthropic cache counts are no longer
  double-counted beside LangChain usage. Gemini thinking tokens count as
  completion tokens.
- **The readers.** The Langfuse reader keeps an Anthropic `system` beside
  `messages` and flattens content blocks; a Gemini `model` turn is an assistant
  turn; LangSmith errored runs are skipped, not malformed; an
  Anthropic `tool_use` block's `input` is read as the call's arguments. One
  unexpected row shape, a plain prompt starting with `[`, or an implausible
  timestamp (before 2020 or in the future) is counted as a malformed row instead
  of aborting the scan or stretching its window.
- **The hosted floor finds a dated or namespaced model id** (`gpt-4o-mini-2024-07-18`,
  `openrouter:...`) through the price table's own lookup (`PriceTable.row`).
- **`secrets.json` is replaced atomically**, so a crash mid-write can no longer
  truncate it and lose every deployment key in file mode.
- **An interrupted `dagnam audit run` no longer pays for a run twice.** Ctrl+C
  after the platform accepted a training run, and before `state.json` recorded
  it, made the next `audit run` submit a second one and orphan the first. The
  re-run now finds the first run (see `resume_creates`). So does the project
  create: its body is only a title, so the key comes from a nonce saved in
  `state.json` before the create, never from the body, and two audit directories
  never share a project (`create_project(..., resume_nonce=...)`). A `state.json`
  written before this still loads. A re-run more than 24 hours later is not
  covered.
- **`dagnam audit delete` survives a deployment the platform will not delete.**
  That refusal is a `DeploymentStateError`, which was not read as a refusal: it
  ended the command before the training job was cancelled and before
  `deleted.json` was written.
- **The project links on PyPI** (repository, changelog, issues, security) point
  at this repository; they named one that does not exist.
- **Reasoning never reaches the answer of a Langfuse generation logged as a bare
  output array.** The array was read as a reply only when every item was
  `reasoning`, `message` or `function_call`. With any other item in it (a
  hosted tool's `web_search_call`, an `mcp_call`, a `custom_tool_call`) the whole
  array, reasoning summary and encrypted reasoning included, became the answer:
  the training target, and uploaded. The same held for a bare array of Anthropic
  blocks with a `thinking` block, or of Gemini parts with a `thought` part. An
  array holding reasoning, a tool call or a message is now read as the reply's
  parts whatever else it holds. With `--source jsonl`, a message object mapped
  as `response` is read as a message too, instead of becoming the answer as JSON
  text with its thinking blocks. Inline reasoning cut off before its
  `</think>` (the token limit hit mid-thought) is no answer either; it was kept
  whole.
- **A tool call a message carries twice is one call.** A LangChain `AIMessage`
  from an Anthropic model lists a call in `tool_calls` and keeps its `tool_use`
  block in `content`; read from both, one call was trained and scored as a list
  of two. Calls are matched by id, or by name and arguments when either has no
  id, and two different calls both stay.
- **A scan that is interrupted leaves nothing `dagnam audit run` can misread.**
  A forced rescan was several separate writes; stopped between them, the
  directory held the old scan report over rows that were partly new, with a
  split built for the old ones. The state is still saved first, so no retired
  run is lost to `audit cancel` or to the budget; `scan-report.json` is then
  removed before any row is rewritten and written last, atomically. Until it is
  back, `audit run` says to scan first, and scanning again finishes the job.
  Replay answers left without a candidate are removed by the next scan.
- **`dagnam audit run` holds the audit directory from the candidate listing to
  the end of the run.** It was taken only for the run itself, so a `dagnam audit
  scan` into the same directory during the confirmation prompt could replace the
  rows after they had been listed and confirmed. Both a scan and a second `audit
  run` are now refused until the first finishes or is declined.
- **Replayed hosted-tool items no longer break a row, and more tool calls are
  read.** A Responses `input` list that replays a `web_search_call`, `mcp_call`
  or another hosted tool's item made the row malformed ("not a chat message");
  above 5% of rows that aborted the scan. Those items are now skipped, like
  reasoning items, and never reach the reply. A `custom_tool_call` item (and its
  `custom_tool_call_output`), a Chat Completions `custom` tool call and the
  legacy top-level `function_call` of a message are read as tool calls by every
  reader, and a list of reply items mapped as a generic export's response is
  read as one.
- **Template normalization no longer takes seconds on a long prompt.** Its email
  rule started over from every character of a long run with no `@` after it:
  80,000 characters took 11 seconds a pass. They take milliseconds; what is
  masked is unchanged. Every rule is linear in a pass, and the number of passes
  is bounded (a prompt that is not settled after eight more keeps what is masked
  so far): one layer of nested quoted spans took a pass of its own, so 80,000
  characters of them still took ten seconds. Ordinary prompts settle within
  three.
- **A rescan removes the workloads it no longer derives, and nothing else, and
  `dagnam audit run --workloads` runs only what the current scan derived.** A
  workload folder is removed only when the directory's previous
  `scan-report.json` lists it with rows, the new scan no longer derives it, and
  its own `meta.json` says a scan wrote it; and then only the files a scan writes
  (the folder is opened once and they are unlinked through that handle). Any other
  entry under `workloads/` is left alone and named in the scan's warnings, so is a
  folder holding files a scan did not write, and an earlier report that cannot be
  read says so. A symbolic link at `workloads/`, at a workload's folder or at any
  file a scan writes, or as the audit directory itself, is refused before
  anything is written or removed, and so is a workload id that is not one plain
  name (a separator, `..`). Naming a workload for `audit run` needs its `dataset`
  in `scan-report.json` as well as its rows on disk, so a directory scanned by an
  older version is safe too.
- **Every file an audit writes goes through one writer that cannot be steered by
  a planted link.** The temp file beside `dataset.jsonl`, `scan-report.json` and
  the other files had a fixed name and was written through whatever it was, so a
  link left there made a scan overwrite a file outside the audit directory. The
  temp file now has a random name, is created exclusively without following a
  link, is synced before it replaces the file, and is removed if the write fails;
  the replay file is appended to the same way, and the stored keys and the audit
  report use it too. The folder is opened once, without following a link, and the
  temp file is created and renamed relative to that handle, so a folder swapped for
  a link after the check cannot redirect a write. Every file read back from an audit
  directory is opened without blocking and without following a link, and must be a
  regular file: a pipe (which hung a scan, holding the audit lock) or a directory
  where a file belongs is refused with its path. `--out` may be a link to a
  directory, resolved once; a link inside it is refused, naming the link.
- **Reasoning is kept out of training prompts and answers for the reply shapes
  the readers know.** `--source csv` did not decode a reply held as a JSON
  string, so a reply with a thinking block made the thinking the target of every
  row. Every source now reads its reply through one rule, whether it is an
  object, a list or JSON text (up to three layers of it). One table
  (`dagnam.audit.readers.reasoning`) lists what counts as reasoning: Anthropic
  `thinking` and `redacted_thinking`, Responses `reasoning`, `reasoning_text` and
  `summary_text` parts, Gemini `thought` parts, Bedrock `reasoningContent`,
  LangChain `reasoning_content`, AI SDK `reasoning` and `redacted-reasoning`, any
  `type` or `block_type` that names reasoning or thinking, and a message's
  `reasoning_content`, `reasoning_details`, `reasoningDetails`, `reasoning_summary`
  and similar fields (never read). Inline `<think>`, `<thinking>`, `<thought>` and
  `<reasoning>` tags are cut before the answer (any number of blocks, any case);
  without an opening tag, only the shape a chat template produces is cut: the
  reasoning, then `</think>` alone on its line, once, then the answer (or nothing:
  the call ended inside its reasoning, so it is counted and gives no row). A response
  object is read through the field that holds the answer for Chat Completions,
  Ollama, Cohere, Gemini and Bedrock Converse. An object no reader recognises is
  never written out whole when a name that is only ever reasoning sits anywhere
  in it: the call is kept without a training row. **Not covered, kept as
  written:** an object whose reasoning sits under a name a structured answer may
  use for itself (`reasoning`, `thinking`, `thoughts`, `rationale`, `scratchpad`),
  raw channel-delimited text, content parts typed `analysis`, and a complete
  reasoning tag in the middle of an answer. An answer that is JSON itself, or
  prose or code that mentions a closing tag, keeps every byte.
- **A call that ended inside the model's reasoning is still a call.** It was left
  out of the workload's calls and spend, so the scan under-stated what the
  teacher costs and could turn a verdict into `too_few_samples`, without a word.
  It now stays in the call count, the tokens and the spend, takes no part in the
  structure class or the output counts, and gives no training row; the scan
  warns with the count and the share of the workload's spend, and the CLI prints
  it. The verdict still counts training rows: a workload with few rows left says
  so in its reason.
- **One reply of tens of thousands of open brackets no longer ends the scan.** The
  JSON decode raised `RecursionError`, which nothing caught: the text is text, and
  a row nested too deeply to decode is one malformed row.
- **Any other item of a Responses `input` list is skipped, known or not.** The
  rule was a list of hosted-tool names, so a replayed `compaction`,
  `tool_search_output`, `program` or `configuration_update` item still made the
  row malformed. Only messages, tool calls and their outputs are conversation;
  every other typed item is dropped, and one with neither a role nor a type is
  still malformed.
- **A JSON array that is the model's answer keeps every byte.** An answer such as
  `[{"type": "paragraph", "content": "Hello"}]` was read as content parts and
  became `Hello`. Only an array holding reasoning, a tool call or a message is
  read as a reply's parts.
- **Two different tool calls with empty ids are two calls.** An empty-string id
  made them one; calls are matched by id only when both have a real one, and
  JSON arguments that are not objects compare by value.
- **The source distribution carries the scripts its tests load.** `tests/scripts`
  ships in the sdist and loads `scripts/*.py`, which it did not hold, so those
  tests failed from an unpacked sdist.

## [0.15.0] - 2026-09-27

### Added

- **Deploy a model version in one call.** `dagnam.deployments.deploy_model_version(
  model_version_id, *, name=None, project_id=None)` and `dagnam deployments
  deploy-version VERSION_ID [--name N] [--project-id P] [--json] [--output PATH]`
  create a deployment and its first serving revision together
  (`POST /api/v1/deployments/from-model-version`) and return the deployment with
  its `api_key`, shown this once. The name defaults to the model name and
  version, and the project to the model entry's project. The client methods are
  `DagnamClient.deploy_model_version` and `AsyncDagnamClient.deploy_model_version`;
  both send an `Idempotency-Key`, so a retried request never creates a second
  deployment. The agent guard hook treats the CLI, `dagnam.deployments` and
  client method shapes (`deploy_model_version`, `create_deployment`) as costly.
- **`DagnamClient.rotate_deployment_key(deployment_id)`** (and its async twin)
  calls `POST /api/v1/deployments/{id}/rotate-key` and returns the new key once.

### Changed

- **Breaking: `predict` sends the input as `{"input": ...}`.**
  `DagnamClient.predict`, `AsyncDagnamClient.predict`, `dagnam.inference` and
  `dagnam inference run` now send `{"input": inputs}`, the body
  `POST /api/v1/inference/{id}/predict` requires, instead of sending `inputs` as
  the whole body, which the route refused. Pass the model input itself (for
  example `{"text": "..."}`); a caller that wrapped it in `{"input": ...}` must
  stop. The `/api/v1/inference` routes `predict`, `predict/batch`, `schema` and
  `health` take the deployment's own key (the `api_key` returned when the
  deployment was created), not an account key: pass `api_key=` to the SDK, or set
  `DAGNAM_API_KEY` to it for `dagnam inference run`, `batch` and `schema`.
- **Breaking: `dagnam.deployments.update` and `dagnam deployments update` only
  rename.** `update(deployment_id, *, name)` and `--name` (now required). The
  instance, capacity and `config` fields are gone: the platform manages serving
  capacity, so they changed nothing for a model version deployment.
- **`dagnam register` says when the plan has no API keys.** API keys need a paid
  plan and a new account starts on Free, so the key step is refused.
  When the refusal body says so (`feature_gated` or `limit_exceeded`),
  `dagnam.account.register` now raises `QuotaExceededError` saying the account
  was created and how to get a key, and `dagnam register` prints that instead
  of "Registration failed". Any other refusal, such as an unverified email,
  propagates unchanged.
- **Restoring a checkpoint works with a personal API key.**
  `dagnam.restore_checkpoint`, `dagnam training restore` and
  `restore_from_checkpoint` (sync and async) work with an API key that has write
  access to training, under the same checks as restart; the platform used to
  accept only a browser session there.
- **A replayed deployment create still returns a usable key.** The platform no
  longer keeps the one-time key in its idempotency cache, so a create that is
  answered from that cache (`Idempotency-Replayed: true`) arrives with
  `api_key: null`. `create_deployment` and `deploy_model_version` (sync and
  async) then rotate the key and return the new one. The rotation revokes the
  key an earlier response carried, so a caller that replays with its own
  `idempotency_key` must store the new key.
- **`dagnam.inference_stream` and `AsyncDagnamClient.stream_predict` use a
  stream session.** The input is posted to
  `POST /api/v1/inference/{id}/predict/stream/session` and the events are read
  from `GET /api/v1/inference/{id}/predict/stream/{session_id}?token=...`, so the
  input no longer travels in the URL. The deprecated `?input=` route is no longer
  called. The events are unchanged.
- **`login_for_bootstrap` explains a two-factor challenge.** When the password
  login answers with a two-factor challenge, `DagnamClient.login_for_bootstrap`
  and its async twin raise `AuthError` pointing to the web app instead of a
  `TypeError`. `dagnam register` never meets one, since a new account has no
  two-factor authentication.

### Removed

- **Breaking: API key management.** `dagnam keys create`, `dagnam keys list` and
  `dagnam keys revoke`; `dagnam.account.create_api_key`, `list_api_keys` and
  `revoke_api_key`; `DagnamClient.list_api_keys` and `revoke_api_key`; and
  `AsyncDagnamClient.create_api_key`, `list_api_keys` and `revoke_api_key`.
  These routes accept only a browser session, so they always failed with an API
  key. Manage keys in the web app under Settings, Security; API keys need a paid
  plan.
  `dagnam.account.api_key_usage` stays. `DagnamClient.create_api_key` stays for
  `dagnam register`, which calls it with the session from its one-time login.
- **Breaking: account settings, security and data.** The `dagnam account`
  command group (`settings`, `notifications`, `profile get|set|photo`,
  `change-password`, `sessions`, `2fa`, `export` and `delete`); the matching
  `dagnam.account` functions (`get_settings`, `update_settings`,
  `reset_settings`, `notification_preferences`,
  `update_notification_preferences`, `get_profile`, `update_profile`,
  `upload_profile_photo`, `change_password`, `list_sessions`, `revoke_session`,
  `revoke_all_sessions`, `two_factor_enabled`, `enable_two_factor`,
  `verify_two_factor`, `disable_two_factor`, `export_data`, `download_export`
  and `delete_account`); and the same methods on `DagnamClient` and
  `AsyncDagnamClient` (`get_notification_prefs` and `update_notification_prefs`
  there). These routes accept only a browser session, so they always failed
  with an API key; use the web app's Settings. An API key cannot change a
  password, two-factor authentication or sessions, or delete the account.
  `dagnam profile show USERNAME` and `dagnam.account.get_public_profile` stay,
  as do `entitlements`, `storage_quota` and `api_key_usage`.
- **Breaking: deployment calls the platform no longer performs.** Serving
  capacity is managed by the platform and a different model is served by
  deploying its version, so scale, rollback and retry are refused (`409`) and
  cost estimates and on-demand metrics collection are gone (`410`). Removed:
  - `dagnam.deployments.scale`, `rollback`, `retry`, `estimate_cost` and
    `collect_metrics`;
  - `DagnamClient.scale_deployment`, `rollback_deployment`,
    `retry_deployment`, `estimate_cost` and `collect_deployment_metrics`, and
    the same five methods on `AsyncDagnamClient`;
  - the CLI commands `dagnam deployments scale`, `rollback`, `retry`,
    `estimate-cost` and `collect-metrics`.
- **Breaking: `dagnam deployments create` and
  `dagnam.deployments.create_from_training_job`.** Both made a deployment with no
  model version behind it, which never served a request. Use `dagnam deployments
  deploy-version` or `dagnam.deployments.deploy_model_version`. The Python
  `dagnam.deployments.create` stays; its deployment serves once
  `dagnam.deployments.create_revision` gives it a model version.
- **Breaking: `dagnam deployments platforms` and `dagnam deployments validate`.**
  They served the checkpoint deployment that `dagnam deployments create` made.
  The Python `dagnam.deployments.platforms` and `validate` stay beside
  `dagnam.deployments.create`.
- **Breaking: `get_system_dataset_meta` and `download_system_dataset`** on
  `DagnamClient` and `AsyncDagnamClient`. They called routes the platform does
  not have; `dagnam.load_dataset("<name>")` now finds a built-in dataset by name
  and loads it by id.

### Fixed

- **`dagnam.load_dataset("<name>")` loads a built-in dataset.** The name is
  matched, ignoring case, against the built-in catalog and the dataset is loaded
  by its id; an unknown name raises `DatasetNotFoundError`, which points to
  `dagnam.datasets.list_system()`. Names are the
  catalog's display names, such as `MNIST Handwritten Digits`.
  `dagnam.datasets.list_system()` lists them, now from
  `GET /api/v1/datasets/browse?source_type=system`.
- **Calls that used a path the platform does not serve.** `dagnam hub starred`,
  `dagnam.hub.starred` and `list_hub_starred` call
  `/api/v1/hub/models/starred`; `import_dag` and `import_dag_existing` call
  `/api/v1/projects/import-dag` and `/api/v1/projects/{id}/import-dag`; and
  `AsyncDagnamClient.stream_deployment_events` reads
  `/api/v1/deployments/{id}/stream`, like the sync client.
- **README and the bundled agent skill.** API keys start with `sk_` and need a
  paid plan; `download_checkpoint` picks the latest checkpoint unless
  `prefer_best=True`; deployment log levels are lowercase; the README states
  the Python version the tests run on, and the package no longer lists a Python
  3.13 classifier; inference examples pass the deployment's
  key; `DAGNAM_JOB_ID` and `DAGNAM_DATASET_ID` are documented; and the skill no
  longer documents `DAGNAM_CACHE_DIR`, which nothing reads (pass `cache_dir=`
  instead).

## [0.14.2] - 2026-09-14

### Fixed

- **`dagnam audit run` stops on a failed dataset upload instead of waiting out
  its timeout.** A dataset whose processing run fails is terminal on the
  platform (`analysis_status: "failed"`), and it never produces the version the
  data step was polling for -- so the run sat there for the whole task timeout
  and then reported no cause. The step now reads the dataset row alongside the
  versions and records `upload_failed: <the platform's reason>`, which halts
  that candidate and shows up in `audit-report.json` and the markdown beside
  it. A dataset still `pending` waits exactly as before.

### Changed

- **`audit-report.json`'s candidates and `winner` carry `candidate_id`.** 0.14.1
  added the key to the contract but left it `null`; a published run now names
  the account-side candidate, so the report and the audit page agree on which
  row won. A `--local-only` run opens no candidate, so it stays `null`.
- **New client method `DagnamClient.get_dataset(dataset_id)`** -- the dataset
  row (`GET /api/v1/datasets/{id}`), which unlike `get_dataset_meta` answers
  while an upload is still being processed and carries `analysis_status` and
  `analysis_error`.

## [0.14.1] - 2026-09-14

### Changed

- **Requires `dagnam-contracts` 0.3.1.** Three more things the SDK used to keep
  its own copy of now come from the contract, so the platform and the CLI
  cannot disagree about them: `dagnam.audit.steps_serve.parse_chat_prompt` is
  `dagnam_contracts.prompts.parse_chat_prompt` (the inverse of the rendering it
  sits beside -- a drifting inverse replays text the classifier never saw), the
  `deleted.json` schema id is `dagnam_contracts.audit.DELETED_SCHEMA`, and the
  serving rates are the contract's. The names, the values and the behaviour are
  unchanged.
- **A `cancelled.json` receipt is stamped `dagnam.audit.cancelled/1`.** The
  server writes it; a cancel stops artifacts where a delete removes them, so it
  was never the same document as `deleted.json` and no longer claims to be. A
  receipt written by an older server is passed through as it arrives.
- **`audit-report.json`'s `winner` block carries a fifth key, `candidate_id`.**
  It is the account-side candidate the winner was published as, and `null` for
  a `--local-only` run (and for any run of this version, which does not source
  it). Readers that assumed four keys should be updated.

### Fixed

- **A run wider than the audit page refuses to publish instead of 404ing every
  candidate.** `AuditCreate.workloads` holds 200 rows, and the rows past that
  cap never reached the account -- so a run that *selected* more than 200
  workloads opened its candidates against workloads the audit did not carry and
  collected one 404 warning per candidate for the whole run. `dagnam audit run`
  now says so once, publishes nothing, and keeps every number in the local
  `audit-report.json`. A scan that merely *found* more than 200 is unchanged:
  the ones this run took are published first, then the biggest spenders.
- **A candidate that failed on an earlier run reaches the account on resume.**
  A recorded failure is terminal, and the frontier skipped such a candidate
  before the publisher ever saw it -- so a run whose first `create_audit` failed
  left that candidate out of the audit page entirely, with no row and no reason.
  The back-fill now publishes each step the candidate finished with the status
  it earned and ends at the step it actually died on, carrying the error.
- **A paused deployment is reported as `paused`, not `deploying`.** An endpoint
  `dagnam audit cancel` had deliberately stopped read, in `audit-report.json`
  and the markdown beside it, as one still on its way up.
- **The audit's "watch it at" link matches the API host exactly.** It was chosen
  with a prefix test, so a URL that merely *starts* like a known host --
  `http://localhost:8000@evil.example/`, whose host is `evil.example` -- was
  given that host's site. It now compares the parsed host and scheme, which
  also fixes two honest cases the prefix missed: `https://api.dagnam.ai/v1` and
  a port-less `http://localhost`.
- **A back-fill no longer republishes one redundant `deploying` pulse** for a
  candidate that scored and was paused since: `wait_active`'s "already done"
  guard now mirrors the step's own, which returns early on `scored` as well as
  on `running`.

### Removed

- **`dagnam.audit.prices.SERVING_RATES`** (and the `prices/serving.json` it read)
  duplicated `dagnam_contracts.audit.serving.SERVING_RATES`, which is what every
  serving cost has actually been computed from. Read the rates from the contract
  -- `load_serving_rates()` there carries their basis and assumptions, and the
  npm package ships the same file.

## [0.14.0] - 2026-09-08

### Changed

- **The audit's scorers, thresholds, frontier rule and report blocks come from
  `dagnam-contracts` 0.3.0.** `dagnam.audit.scoring`, `thresholds`,
  `economics`, `frontier`, `report`, `derive` and `steps_serve` keep every
  public name and every number they had, but the bodies now live in
  `dagnam_contracts.audit` -- one definition of agreement, the economics
  bands, the winner and the switch block, so the SDK and the platform can
  never disagree about the same candidates.
- **A quoted label now normalizes like a bare one.** `normalize_label`
  (`dagnam.audit.normalize_label`) is the contract's, which strips surrounding
  punctuation rather than only trailing: `'"Refund"'` normalizes to `refund`,
  where it previously kept its opening quote. Scoring a holdout whose teacher
  or candidate quotes its labels now counts those rows as agreeing -- and,
  because `dagnam.audit.derive` normalizes through the same function, the
  `label` field of the derived training rows a workload uploads changes with
  it, not only the scoring.

### Added

- **`dagnam audit run` publishes the audit to your account as it runs, so you
  can watch it on the website.** The scan header and every workload it found
  (ids, structure class, masked template excerpt, calls/day, spend, verdict and
  its reason, the redaction counts, and which workloads this run took) go up
  once; then a candidate per workload x kind, and one record per step as the
  frontier walks it -- uploading, splitting, the PII check, submitting,
  training, deploying, and the scored replay with its agreement, its
  client-measured latency and the credits it burned. Derived rows, raw traces
  and deployment keys never leave the machine. New `--local-only` on
  `dagnam audit run` opts out entirely (nothing is published and the listing
  says so), `dagnam audit cancel` and `dagnam audit delete` go through the
  account for a run that published -- writing the server's own receipt as
  `cancelled.json` and `deleted.json` respectively -- and `state.json` gains
  `audit_id` so a resumed run continues the audit it already opened instead of
  starting a second one. Publishing is best effort throughout: a failed call is
  logged, queued behind the next one, and never stops a run.
- **The run says where to watch what it published.** A created audit prints
  `published: <audit-id> — watch it at <site>/audits/<audit-id>` (the site is
  derived from the configured API URL, exactly as `dagnam login`'s sign-in link
  is), and `dagnam audit status` carries the same `audit_id` and `url` in its
  table and its `--json`. A run that stopped short still publishes its halt,
  and it is published once: a resumed run that stops the same way again leaves
  the halt the account already shows, because the account resumes a halted
  audit itself on the next step this run publishes.
- **`audit cancel` writes `cancelled.json`, not `deleted.json`.** A cancel
  stops training jobs and pauses endpoints; nothing is deleted, so
  `deleted.json` now only ever means the artifacts are gone. Both cancel paths
  -- the local walk and the one the server performs for a published run --
  leave the same local state (each candidate's run `cancelled`, its deployment
  `paused`, the audit `halted`), so `audit status` reads the same after either
  and a later `audit run` no longer resumes into `wait_run` against a job that
  was cancelled.
- **`--floor` and `--max-credits` are validated before anything is uploaded.**
  A value outside `0 < floor <= 1`, or a credit ceiling that is not a
  non-negative whole number, is an argument error (exit 2) rather than a run
  whose every publish is refused by the server. Non-ASCII digit forms
  (`"²"`), which `str.isdigit` accepts and `int()` does not, are rejected too.
- **`state.json` is not backward compatible with 0.13.0.** The file gains
  `audit_id` and a per-candidate `published_candidate_id`; 0.13.0 refuses a
  step key it does not know, so an audit directory written by this release
  cannot be resumed by the previous one. Newer readers read older files fine.
- **`dagnam audit scan` prices calls from every major provider, not just
  OpenAI.** A model id is now looked up exactly, then through a normalized
  form (new `canonical_model_id`): lowercased, with the provider/gateway or
  Bedrock region namespace stripped (`anthropic/claude-sonnet-4-5`,
  `models/gemini-2.5-flash`, `openrouter:mistral/mistral-large-latest`,
  `us.anthropic.claude-...`) and a floating `-latest` dropped; a trailing
  release date or Bedrock version suffix (`gpt-4o-2024-08-06`, `-v1:0`) is
  dropped only when the exact id has no row of its own, so a dated variant the
  vendor prices separately keeps its own price. The Langfuse, LangSmith and
  OpenAI readers now read token usage through one shared helper that accepts
  every vendor spelling -- Anthropic's `input_tokens`/`output_tokens` plus its
  `cache_read_input_tokens` and `cache_creation_input_tokens`, Gemini's
  `promptTokenCount`/`candidatesTokenCount` (and their snake_case forms),
  LangSmith's `usage_metadata`, Langfuse's `usage`/`usageDetails` -- and
  LangSmith runs also name their model through
  `extra.invocation_params.model_name` and their reply through
  `outputs.content`, which is where `wrap_anthropic` and `wrap_gemini` put it
  (a string, or typed content blocks whose `text` is joined and whose non-text
  blocks are skipped); their requests read from `inputs.contents` as well, so
  an Anthropic or Gemini run no longer lands with an empty response. The
  bundled price table gains 19 rows read from the vendors' own pricing pages
  on 2026-09-07: Mistral (7), DeepSeek (2), xAI (5) and Cohere (5), each
  cited in the table's `_sources`.
  Meta Llama has no rows: neither Together's nor Groq's pricing page states
  the API model ids and per-token prices together, and the table never guesses
  a price.

## [0.13.0] - 2026-09-07

### Changed

- **`dagnam audit run --max-credits` now counts the holdout replay.** Every
  served prediction is metered, so a candidate's replay -- not its training
  run -- is most of what an audit spends; counting the training estimate
  alone under-reported a run by an order of magnitude and let an audit walk
  into its plan limit mid-flight. The replay's cost is *measured*, not
  assumed from a rate card: the SDK reads the account's credit balance either
  side of the replay (new `DagnamClient.get_credit_balance` /
  `AsyncDagnamClient.get_credit_balance`, `GET /api/v1/users/me/credits`) and
  records the difference as the candidate's `replay_cost_credits`. A balance
  read the platform refuses leaves the cost unknown and never unscores a
  candidate. `audit-report.json` keeps `training_cost_credits` and gains
  `replay_cost_credits` next to it; the Markdown candidate table's
  `training credits` column is now `credits (train + replay)`.

## [0.12.0] - 2026-09-07

### Added

- **`dagnam audit` CLI.** `audit scan <export> --source ... [--map field=column]
  [--window 30d] [--price-table P] [--out DIR] [--json]` discovers and prices
  the workloads in a trace export and derives redacted rows for the ones worth
  auditing, opening no network connection (test-enforced). `audit run DIR
  [--workloads w1,w2] [--floor F] [--max-credits N] [--yes] [--no-wait]
  [--json]` prints exactly what will be uploaded -- workload ids, row counts,
  redaction counts per class, the target project, the credit ceiling -- and
  refuses without `--yes` on a non-interactive terminal, then trains, serves
  and scores the candidates and writes `audit-report.{json,md}` (schema
  `dagnam.audit.report/1`: the scan plus candidates, the frontier's winner
  and the switch values per workload; the Markdown opens with the KEEP rows
  and shows an OpenAI-client switch snippet that names a `key_ref`, never a
  key). `audit status DIR` tables every candidate with its job, endpoint and
  the endpoint's 7-day request count; `audit cancel DIR` cancels in-flight
  jobs and pauses deployments and nothing else; `audit delete DIR [--yes]`
  deletes every recorded deployment, model version, dataset and the project,
  re-reads each id expecting not-found, writes `deleted.json`
  (`dagnam.audit.deleted/1`, `already_absent` for ids already gone), then
  removes the local rows and forgets the deployment keys. New helpers:
  `dagnam.audit.build_audit_report` / `write_audit_report` /
  `render_switch_snippet`, `dagnam.audit.delete_audit`,
  `SecretStore.forget`, `serving_cost_usd_month`, `select_workloads`; the
  scan report now carries each derived workload's `dataset` numbers and
  applies the `MIN_HOLDOUT` rule; a workload's `models` are in its JSON.

- **`dagnam.audit` workload discovery.** `discover_workloads` groups trace
  records by the blake2b hash of their normalized system-prompt template
  (`normalize_template` / `template_hash` mask ids, numbers, dates, quoted
  context and `{{placeholders}}`), classifies each group's output structure
  (`classify_outputs` -> `StructureClass`: label, short span, JSON object,
  free text) and summarizes it as a `Workload`. Pure and deterministic: the
  same records in any order give the same workloads in the same order.

- **`dagnam.audit` economics and price tables.** `replaceability` applies the
  break-even rule (teacher $/month over student $/month plus maintenance,
  thresholds in `dagnam.audit.thresholds`) and returns a `Verdict` whose
  status is `candidate`, `marginal`, `not_worth_it`, or the reason a workload
  is not priced (`not_audited` for free text, `too_few_samples`,
  `unknown_cost`). `PriceTable` loads a
  versioned vendor price table bundled under `dagnam/audit/prices/` (the
  newest is the default; a model without a row prices to `None`, never a
  guess) plus the platform's estimated serving rates. `build_scan_report` /
  `write_scan_report` produce `scan-report.json`.

- **`dagnam.audit` candidate datasets.** `derive_rows` turns a workload's
  records into training rows in the recipe's row format, `redact_rows`
  applies every PII class the shared `dagnam_contracts.hygiene` contract
  detects (`PII_POLICY`, counted in `RedactStats`) before a row touches
  disk, `dedup_rows` drops exact duplicates, and `time_split` makes the
  time-ordered train / `eval_holdout` split (`HOLDOUT_SHARE` = 20%), snapped
  so no session straddles the boundary. `build_dataset` / `write_workload`
  write the workspace (`WorkloadDataset`, `DeriveStats`, `DedupStats`).

- **`dagnam.audit` orchestration and scoring.** `run_audit` runs the
  resumable frontier over workloads x `CANDIDATES`: the `AuditState` is
  saved after every step, so an interrupt or crash resumes from the last
  completed step, and a `KeyboardInterrupt` is never caught. `replay_holdout`
  replays the holdout through a served candidate over the OpenAI-compatible
  route with a deployment-scoped key that `SecretStore` keeps in the OS
  keyring (the `audit` extra), else a `0600` file under the audit dir;
  `score_labels` / `score_json` compute client-side agreement with a 95%
  Wilson interval, and `frontier` picks the cheapest candidate whose
  agreement *lower bound* clears the floor (`Winner`, `CandidateResult`).

### Changed

- The `dagnam-contracts` floor rises from `0.1.3` to `0.2.0`, which ships the
  `hygiene` PII contract the audit redaction step is built on.

### Fixed

- **`dagnam audit delete` now deletes the training runs it created**, before the
  datasets they name. It previously stopped at `HTTP 409: Dataset is referenced
  by a training run and cannot be deleted`, because a run's specification pins
  the dataset version it trained on and the platform refuses to destroy the data
  under it; deleting the job takes that specification with it. A platform
  refusal is no longer an abort either: the id is recorded in `deleted.json` as
  `{"status": "blocked", "reason": ...}` (e.g. a job that is not yet terminal,
  which the platform will not delete), every other id is still deleted and
  recorded, and the local workload rows and deployment keys are kept so a later
  `audit delete` can finish the job.

- **`dagnam audit run`'s holdout replay waits out a transient gateway refusal.**
  Four workers replaying a holdout outrun the inference route's per-minute rate
  limit, and every `HTTP 429` counted as a failed call -- a 396-row holdout came
  back `unreliable: 276 of 396 replay calls failed` with no agreement measured.
  A `502`, `503` or `504` counted the same way, and one of them (the gateway
  forwarding to a scale-to-zero replica that hiccupped) was enough to score 395
  of 396 rows and fail an identity check on the row count alone. Every status in
  `TRANSIENT_STATUSES` -- `429`, `502`, `503`, `504` -- is now waited out and the
  same request retried, sharing one budget of `RATE_LIMIT_RETRIES` attempts; only
  an exhausted budget is an error, and every other status (a `500` included)
  still fails its call at once. The wait is the seconds the refusal's
  `Retry-After` names, and when it names none -- the gateway currently sends no
  such header on a 429 -- a doubling backoff (1, 2, 4, 8, 16, 32, 64 s, each
  capped at `RATE_LIMIT_SLEEP_MAX_SECONDS`), so a call outlasts a full per-minute
  window instead of spending every attempt inside the one window it already
  exhausted. The reported latency is the successful attempt's alone, so the waits
  never show up as the endpoint's p50/p95.
- **`dagnam audit cancel` no longer aborts on a deployment the platform refuses
  to pause.** A revision that never activated leaves its deployment in
  `not_provisioned`, and the pause came back `HTTP 409: Invalid status
  transition from not_provisioned to paused`, which stopped the cancel before it
  reached the remaining candidates. The refusal is now recorded as
  `{"action": "pause_refused", "reason": ...}`, every other job is still
  cancelled, and the audit is still halted. The `audit run` deploy wait tolerates
  the same refusal when it pauses a deployment that timed out.
- **`dagnam audit status` no longer crashes on a deployment that is already
  gone.** A state file that still names a deployment the platform has since
  deleted made the command exit with `Error: Deployment '...' not found`; the
  row now simply reports no `requests_7d` and everything else is unchanged.

## [0.11.0] - 2026-09-06

### Added

- **`dagnam.audit` trace readers.** `read_traces(path, source=...)` streams a
  Langfuse observation export, a LangSmith run export (JSONL or Parquet), an
  OpenAI batch/stored-completion JSONL, or any JSONL/CSV file described by a
  `column_map`, into the canonical `TraceRecord`. Rows stream through polars in
  batches; malformed rows are counted in `ReadStats` and skipped, and a share
  above 5% ends the read with `MalformedExportError` naming the first three
  offending rows. A new `audit` extra (`pip install dagnam[audit]`) declares
  the keychain dependency the audit CLI will use.

- **Attach mode for platform-run training scripts.** When `DAGNAM_JOB_ID` is
  set, `dagnam.training.init` attaches the metrics uploader to that existing
  job instead of registering a new local run: the resolved API key (a
  short-lived run token) is used directly as the upload credential, no project
  id is required, and events are tagged `source.kind = "local_attach"`, the kind the platform
  already accepts for an attached run, rather than `local_stream`.

- **`dagnam.deployments.create_revision`.** Rolls a deployment to a new model
  version via `POST /deployments/{id}/revisions`, sending the required
  `Idempotency-Key` (minted when omitted). Returns the revision as created;
  activation is asynchronous, so poll `revisions()` for `is_active`. Available
  on the sync and async clients as `create_deployment_revision`.

- **`dagnam.datasets.create_explicit_splits` / `scan_pii`.** Enqueue an
  explicit-membership split (`POST /datasets/{id}/versions/{vid}/splits/explicit`)
  or a PII scan (`POST .../pii-scan`; an empty policy reports only) and get
  back a `LongRunningOperation` over the dataset task, like `upload_from_url`.
  Available on the sync and async clients under the same names.

- **`dagnam.deployments.create_from_training_job`.** `create()` with the
  serverless chat-model defaults (`vllm` / `text` / `modal-serverless`)
  filled in, so serving a finished job takes a name, a project and the job id.

- **`dagnam.foundation.wait_run` and `dagnam.RunFailedError`.** Polls a
  fine-tuning run until it settles; returns the completed run, raises
  `RunFailedError` (with the server's `error_message` as `reason`) on `failed`,
  `cancelled` or `timeout`, and `TimeoutError` past `timeout`. `sleep`/`now`
  are injectable.

- **Foundation fine-tuning SDK: `dagnam.foundation`.** `list_bases` pages the
  curated base models the platform will fine-tune, `list_recipes` returns the
  shipped training recipes, `submit` starts a run against a dataset version,
  and `get_run` reads the run's frozen specification plus its job's live
  status. Each recipe carries its hyperparameter **JSON Schema** and is
  returned exactly as served -- the SDK never re-declares a hyperparameter or
  filters an unrecognised key, so a form rendered from the schema and the
  validation the API performs read the same declaration and cannot drift.
  Submitting sends an `Idempotency-Key`, so a retry replays rather than
  starting a second paid run. Available on the sync and async clients as
  `list_foundation_catalog`, `list_training_recipes`, `create_foundation_run`
  and `get_foundation_run`. New `FoundationRunNotFoundError` joins the
  top-level `dagnam` and `dagnam.exceptions` exports.

- **Foundation evaluation runs.** `dagnam.foundation.evaluate` starts an
  evaluation run, `get_evaluation` reads one, and `list_evaluations` lists a
  model version's runs. New `EvaluationRunNotFoundError` joins the top-level `dagnam` and
  `dagnam.exceptions` exports; as with every other lookup, a run that belongs
  to someone else answers the same 404.

- **Two-factor authentication.** `dagnam.account.two_factor_enabled` /
  `enable_two_factor` / `verify_two_factor` / `disable_two_factor`, on the
  sync and async clients too, and a `dagnam account 2fa` command group.
  Enrollment is two steps: `enable` returns the secret and backup codes but
  activates nothing until `verify` confirms a code. The password is read via
  `getpass`, never from argv; `disable` asks for a typed confirmation.

### Fixed

- A dataset whose metadata declares format `json` but whose data file is
  `.jsonl` now loads as line-delimited JSON through every converter
  (`to_polars`, the tabular sample iterator, and the framework loaders).
  Previously the file was not found at all, or parsed as a JSON document.

### Changed

- Wheels are built with `hatchling>=1.27,<1.32` so their core metadata stays
  at version 2.4; hatchling 1.32 defaults to 2.5, which the publish action's
  twine still rejects.

## [0.10.0] - 2026-08-14

### Added

- **Model registry SDK: `dagnam.models`.** `push` creates a model entry and
  draft version, uploads every artifact file (its registry `artifact_type`
  inferred from the filename), and finalizes the version in one call.
  `resolve`, `get_lineage`, and `get_task_contract` fetch version metadata,
  lineage, and task contracts. `download` caches artifacts locally under
  `~/.dagnam/models/`, named from the artifact's real `filename` (or
  `logical_key`) when the server provides one, with the same checksum
  verification, atomic promotion, and LRU eviction discipline as
  `download_checkpoint` — eviction only runs once a `max_model_cache_size`
  (or shared `max_cache_size`) budget is configured, and never evicts the
  entry a call just wrote. New `ModelError` / `ModelNotFoundError`
  exceptions join the top-level `dagnam` and `dagnam.exceptions` exports.
  A `dagnam models` CLI group (`push`, `get`, `list`, `download`, `lineage`,
  `task-contract`) is also available.

- **Deployment revision history: `dagnam.deployments.revisions`.** Lists a
  deployment's revisions newest first — revision number, model version,
  serving engine, capacity mode, status, failure reason, creation time, and
  which one is currently serving. Paged via `page` / `limit` (the server caps
  a page at 200). Available on the sync and async clients as
  `get_deployment_revisions`, and as a `dagnam deployments revisions <id>`
  CLI subcommand.

- **`dagnam.models.push_run_artifacts`.** The client a platform-run training
  job pushes its weights with: it resolves the job from `DAGNAM_JOB_ID` and
  the credential from the run token already in the worker's environment, and
  lets the server decide where the artifacts land, so a run token never
  chooses a model name or slug. Completion is driven by the server's
  per-artifact `committed` flag, so an artifact the server has already
  verified is never re-uploaded. Raises `ModelError` (a sibling of
  `APIError`, not a subclass) for a rejected upload.

### Removed

- **Breaking:** the deprecated `dagnam._contracts` shim, as 0.9.0 announced.
  It re-exported `dagnam_contracts` unchanged and had no consumers in the
  package; import `dagnam_contracts` directly.

## [0.9.0] - 2026-08-10

### Removed

- **The 1.18 MB validation corpus no longer ships in the wheel.** Every
  `pip install dagnam` carried `dagnam/_contracts/validation-corpus.json`, which
  has no runtime consumer — it is test data for the validators. It moves to
  `tests/_contracts/`, where a fixture belongs. The built wheel drops from
  390,680 to 311,723 bytes compressed, and `dagnam/_contracts/` is now the
  deprecation shim alone.
  `dagnam._contracts` still re-exports every public name from
  `dagnam_contracts` unchanged, so no import breaks in this release. **That
  shim is removed in 0.10.0** — import `dagnam_contracts` directly.

### Added

- **Video / 3-D data path.** New `dagnam.data.loaders.video` provides a
  `resize_frames` loader, driven by the dataset's declared metadata rather than
  by its name, so a video dataset can feed a `Conv3d`-style model. `"video"`
  joins the existing `apply_transform` dispatch and `_KIND_TO_MODALITY` tables,
  and the PyTorch-native loader applies its channels-first transpose to 5-D
  batches exactly as it already did for 2-D images (TensorFlow and Flax already
  matched, being channels-last). Clips are a plain numeric `.npz` read by the
  existing `ArrayDecoder` with `allow_pickle=False` — no new decoder and no
  relaxation of the deserialization defense.
- **Async dataset methods gain `version=` parity with the sync client.**
  `AsyncDagnamClient.get_dataset_meta`, `get_system_dataset_meta`, and
  `download_dataset` previously had no `version` parameter at all, so pinning
  a dataset version was only possible through the sync client. `version` is
  now accepted on all three (keyword-only on `download_dataset`, matching the
  sync signature) and forwarded as a `?version=` query parameter.

### Changed

- **Generated training scripts now pull the exact rows a platform split
  contains.** `/meta` serves each split's `member_row_indices`, and the
  converters resolve `split="train"` against that membership instead of
  re-deriving it from a client-side seeded shuffle. A percentage was never a
  consumable split: the eval holdout a contamination guard had cleared was not
  the one a run evaluated against. Datasets with no server-declared splits are
  unaffected and keep the deterministic ratio partition. `train`/`val`/`test`
  alias onto a platform `eval_holdout` split; a split that resolves to nothing
  now raises rather than quietly ratio-slicing rows the declared splits
  already own. Folder-backed image/audio loaders keep the ratio partition by
  design — a server row index does not name a file in a directory walk.
- **`dagnam <typo>` suggests the intended command again.** argparse renders
  its "invalid choice" message with or without quotes depending on the Python
  version, and the parser only scraped the quoted form — so on an unquoted
  build every unknown-command error silently lost its "Did you mean ...?"
  line. Candidates now come from the subparser's real `choices`, with a
  message fallback that accepts either rendering.

- **`version=` now resolves against real dataset versions, not only the legacy
  metadata map.** The backend previously resolved `?version=` solely against a
  string-keyed map in the dataset's metadata, so a numeric key like `"1"` only
  worked if someone had explicitly registered it there. With server-side
  `DatasetVersion` records, `"1"` (and any `version_number`, and a version's
  UUID) now resolves to that version's stored file — and every pre-existing
  dataset was backfilled as version 1. A call that passed `version="1"` and
  previously 404'd will now succeed and return the backfilled v1 content;
  legacy metadata-map keys keep resolving exactly as before, and are still
  tried when a numeric/UUID lookup finds nothing.

### Security

- **`keras>=3.15.0` security floor added to the `tensorflow` and `all` extras.**
  keras arrives only as TensorFlow's own dependency, and TensorFlow asks for
  nothing newer than `keras>=3.12.0`, so the resolved 3.14.1 carried six open
  advisories: arbitrary code execution via `Lambda` deserialization
  (GHSA-5gwj-m78q-7pq3) and `TorchModuleWrapper.from_config`
  (GHSA-v2w2-w228-c444), local-file disclosure via HDF5 ExternalLinks
  (GHSA-m8wh-29wm-52mv) and virtual datasets (GHSA-26c4-7vv6-867j), and path
  traversal via `DiskIOStore` layer names (GHSA-gh82-f9x8-5frx) and tar symlink
  entries (GHSA-58hv-7753-xmfq). 3.15.0 is the first release clearing all six.
  The SDK itself only calls `tf.keras.utils.image_dataset_from_directory`, so
  none of the affected paths run inside `dagnam` — but every one of them is
  reached by a user who loads a `.keras`/`.h5` artifact with the keras this
  extra installs, so the floor is declared in the published metadata rather
  than pinned only in this repo's lockfile.

## [0.8.0] - 2026-07-28

### Security

- **Breaking: `deployments.rollback()` no longer accepts a filesystem path.**
  `dagnam.deployments.rollback(deployment_id, checkpoint_path)` is now
  `dagnam.deployments.rollback(deployment_id, checkpoint_id)`, and the CLI's
  `dagnam deployments rollback --checkpoint-path` is now `--checkpoint-id`.
  The previous `checkpoint_path` parameter let a caller point a deployment at
  an arbitrary server-side filesystem path with no ownership check — a path
  traversal vector, and (via the resulting error message) a way to probe
  whether an arbitrary path existed on the server. The checkpoint is now
  resolved and re-authorized server-side by id through the same
  checkpoint -> training job -> project -> owner chain used elsewhere, and a
  checkpoint you don't own returns a uniform 404. Callers must update
  `rollback(...)` calls and the `dagnam deployments rollback` CLI invocation
  to pass a checkpoint id instead of a path.

- **Typed exceptions for server rejections that previously arrived
  undifferentiated.** `EmailNotVerifiedError`, `AccountSuspendedError`,
  `AccountLockedError`, `PayloadTooLargeError` and `InvalidURLError` are
  exported at package top level and raised from the corresponding backend
  markers. Each subclasses `APIError`, so existing `except APIError:` handlers
  keep working unchanged — this widens what callers *can* catch without
  narrowing what they already do. Every response mapper was updated, not only
  the one that surfaced the problem: a partial fix would have left the same
  rejection raising different types depending on which call path reached it.
  A caller can now, for example, distinguish "verify your email" from "your
  account is locked" and prompt correctly instead of showing a generic failure.

### Fixed

- Dataset upload pointed at endpoints that no longer exist; it now targets the
  live routes.
- **The test suite no longer segfaults when PyTorch and TensorFlow share a
  process.** `import tensorflow` followed by `import torchvision` crashes the
  interpreter (SIGSEGV); the reverse order is fine, and importing plain `torch`
  first is not enough. The suite exercises all three framework loaders in one
  process, so whichever imported first decided whether the run survived. The
  root `conftest.py` now pins the order. This affects consumers too: an
  application that mixes both frameworks should import the PyTorch side first.

### Changed

- Security floors raised on dependencies with known advisories: `pillow>=12.3.0`
  (13 advisories), `torch>=2.13.0` (GHSA-rrmf-rvhw-rf47), and a resolution
  constraint of `setuptools>=83.0.0` (PYSEC-2026-3447, reached only transitively
  via chex/tensorboard/tensorflow/torch).

### Compatibility

- Python `>=3.12`.
- Requires a Dagnam backend that resolves deployment rollbacks by checkpoint id
  and emits the `email_not_verified` / `account_suspended` / `account_locked` /
  `blocked_ip` / `invalid_url` markers this release maps to typed exceptions.
  Against an older backend the new exception classes simply never surface, but
  `deployments.rollback()` will fail — the id it now sends is not a path.


## [0.7.0] - 2026-07-13

### Added

- **Automatic retries for transient failures.** Retry-safe API requests
  (idempotent methods, plus any request carrying an idempotency key) now retry
  on connection errors and `429`/`5xx` responses using equal-jitter exponential
  backoff, bounded by a per-client **retry budget** (token bucket) so a flapping
  backend cannot trigger an unbounded retry storm. A server `Retry-After` header
  is honored but capped, so a hostile value cannot wedge the client in a long
  sleep.
- **Idempotency keys** for retry-safe writes: a retriable `POST` mints one
  `uuid4` `Idempotency-Key` and reuses it across retries, so a retried create is
  not applied twice server-side.
- **Cross-process cache locking.** Dataset/checkpoint cache writes and LRU
  eviction are serialized with a file lock (new `filelock>=3.13` base
  dependency), so multiple processes sharing a cache root no longer race or
  corrupt entries.
- Async parity: `dagnam.aio.AsyncDagnamClient` now applies the same
  retry/backoff and transport-error handling as the sync client, fully
  non-blocking on the event loop.
- `dagnam.ResponseError` — a public `APIError` subclass for malformed,
  undecodable, or wrong-shape server response bodies.
- Library logging contract: a package-level `NullHandler`, namespaced child
  loggers (`dagnam.http`/`dagnam.cache`/`dagnam.lro`/`dagnam.sse`) with a
  credential-redacting filter, and `dagnam.enable_debug_logging()` convenience.

### Changed

- `response_json_value`/`response_json_object`/`response_json_array` and
  `BaseDagnamClient._expect_object` now raise `ResponseError` (an `APIError`
  subclass) instead of a raw `TypeError`/`ValueError`/`json.JSONDecodeError`
  when a server response body is malformed, undecodable, or the wrong JSON
  shape. This affects every sync client method that decodes a response body
  (datasets, training, account, hub, projects, deployments, codegen). Code
  that narrowly caught `except TypeError`/`except ValueError` around these
  calls should catch `except dagnam.ResponseError` (or the broader
  `dagnam.APIError`/`dagnam.DagnamError`) instead. Client mixins that
  optionally fall back to a plain-text response body preserve that behavior.

### Security

- **Path traversal / arbitrary-file-write (critical).** A server-supplied
  dataset filename (dataset metadata / `Content-Disposition`) is now reduced to
  a safe bare basename before it is joined under the download directory.
  Previously a malicious or compromised server could return an absolute path or
  a `..` sequence and make a dataset download write outside the cache — a
  remote-code-execution vector (for example, overwriting a shell rc file).
- **Presigned-URL credential redaction.** The log/error redaction filter now
  scrubs presigned-URL credential parameters (`X-Amz-Signature`,
  `X-Amz-Credential`, `Signature`, `sig`, `AWSAccessKeyId`, …) in addition to
  `token`/`api_key`/`signature`, and transport-error text is scrubbed before it
  reaches a log record or exception — so a presigned dataset/checkpoint URL can
  no longer leak its signature into logs.
- **Cache trust boundary.** The SDK warns once when the cache root is
  group/world-writable (a same-size cache-poisoning risk on a shared host) and
  documents `verify=True` to force a full checksum re-verification on load for
  shared caches.
- The `LongRunningOperation` poll interval is floored at 0.1 s, so a hostile
  server flooding `429`/`503` cannot drive a `sleep(0)` busy-loop.

## [0.6.0] - 2026-07-02

### Added

- Full public **exception hierarchy** re-exported from the top-level package and
  a new `dagnam.exceptions` module, so callers can write `except dagnam.APIError`
  (or `except dagnam.DagnamError`) without reaching into the private
  `dagnam._core.exceptions` path.
- `numpy` and `Pillow` are now declared **base dependencies** (they are imported
  eagerly by the dataset layer / image loaders), so a plain `pip install dagnam`
  can load datasets instead of failing with a bare `ModuleNotFoundError`.
- `torchvision` is now part of the `pytorch` and `all` extras — the PyTorch
  image-folder loaders require it, and the README already promised it.
- Cross-platform **Agent Skill** for AI coding agents (Claude Code and Codex),
  shipped as package data and activated with `dagnam agent install`
  (auto-detect & prompt; `--claude` / `--codex` / `--all` / `--yes` / `--symlink`
  for explicit and CI installs; `dagnam agent uninstall` to remove). It installs a
  single write-once skill (`SKILL.md` + on-demand domain `reference/*.md` + helper
  `scripts/`) into each harness's auto-discovered skills directory, plus per-harness
  adapters: a Claude plugin with the `dagnam-runner` subagent and a `PreToolUse`
  guard hook, and a Codex `openai.yaml` + idempotent guard-hook merge into
  `~/.codex/hooks.json`. Skill/adapter versions are stamped to the installed SDK.
- **Dry-run / preview-by-default guardrail** for costly, irreversible, or public
  actions (training-job and deployment creation, deletes, hub publish): a behavioral
  gate in the skill plus a fail-open cross-platform `PreToolUse` deny hook
  (`python -m dagnam._agent.guardhook`).
- Claude plugin marketplace listing (`.claude-plugin/marketplace.json`) as an
  alternate `/plugin install` door.
- `dagnam codegen download` now accepts a **directory** as the destination and
  auto-extracts the generated files into it (the SDK `download(dest=<dir>)` returns
  the directory `Path`); previously passing a directory raised a `PermissionError`.
- Uniform CLI output convention: `codegen validate`/`preview`/`generate` and
  `projects architecture` now accept `--json` (JSON to stdout) and `--output PATH`
  (JSON to file), with human-readable summaries by default — matching the other
  command groups.

### Changed

- **BREAKING (SDK/SSE):** `open_training_stream`, `open_deployment_stream`, and
  async `stream_training_events` now mint short-lived, resource-scoped
  stream-access tokens per (re)connect and send streams `?token=...`; the SDK no
  longer sends long-lived `?api_key=` on SSE URLs. Direct consumers of the SSE
  endpoints must switch to `POST .../stream-access-token` before opening a
  stream, and the backend + SDK must deploy together.
- **BREAKING (CLI):** `dagnam codegen download --output PATH` is renamed to
  `--dest PATH` (no alias). `--output` now uniformly means "write the JSON result
  to a file"; `--dest` is the downloaded-artifact destination (a file path streams
  the ZIP; a directory auto-extracts). Migrate `codegen download ... --output X`
  to `... --dest X`.

- Internal: the quality gates are now clean at zero — `ruff` reports no errors
  and `pyright` (strict, minus the unavoidable untyped-ML-library completeness
  diagnostics) reports no errors. Remaining suppressions are centralized and
  documented in `pyproject.toml`. Async checkpoint/codegen downloads now write
  to disk via `asyncio.to_thread` so they no longer block the event loop, and
  `dagnam.data.loaders` submodules are imported lazily via `__getattr__`
  (PEP 562). No public API or runtime behavior changed.
- Widened `requires-python` to `>=3.12` (no upper cap) and declared Python 3.13
  support; corrected dependency floors so extras are actually installable
  (`tensorflow>=2.16`, the first Python-3.12-capable TF; `torch>=2.4` for the
  `pytorch` extra).
- Label encoding is unified across `to_arrays()` and every framework loader via
  one canonical string-identity mapping, so an integer label column no longer
  crashes `to_pytorch_loader()` / `to_tensorflow_dataset()` / `to_flax_dataset()`
  with a bare `KeyError`.

### Fixed

- The PyTorch native-numpy validation split no longer leaks the **entire** train
  set into "val" when the validation ratio rounds down to zero.
- Cache size/info scans skip a file removed mid-scan (concurrent eviction)
  instead of aborting with `FileNotFoundError`.
- A corrupt (non-integer) `max_cache_size` config value no longer crashes a
  freshly completed download; it falls back to the default eviction budget.
- `to_arrays()` builds an object array for ragged (variable-length) features
  instead of raising `ValueError` under numpy ≥ 1.24.

### Security

- System-dataset `.npz` decoding now uses `allow_pickle=False` and refuses
  pickled object arrays — closing an arbitrary-code-execution vector where a
  crafted dataset could run code inside `numpy.load`.
- Generated-code ZIP extraction is hardened against **zip-slip**: archive
  members with path-traversal, absolute, or symlink paths are rejected instead
  of being written outside the destination directory.
- Checkpoint downloads that arrive without a server checksum (e.g. an S3
  presigned redirect) are now flagged with a loud warning instead of being
  silently accepted unverified.

## [0.5.0] - 2026-06-03

### Added

- Release-process documentation for maintainers.
- CLI version and account inspection commands: `dagnam --version`, `-v`,
  `version`, `whoami`, `logout`, and read-only `config list` / `config get`.
- Expanded CLI help text with command descriptions, examples, and argument
  descriptions.
- Training-job lifecycle SDK + CLI: `dagnam training create/list/get/cancel/
  delete/logs/metrics/metrics-summary`, including after-the-fact retrieval of
  historical logs and metrics.
- `dagnam usage` command and `dagnam.account.*` helpers for plan, entitlement,
  storage-quota, and per-API-key usage inspection.
- Width-aware table rendering, pagination footers, and consistent
  `--json` / `--output` flags across table-printing commands.
- `--no-progress` flag for `dagnam codegen download` (progress is also hidden
  automatically when stderr is not a TTY), matching `dagnam dataset download`.

### Changed

- Set the supported Python runtime to Python 3.12 so `dagnam[all]` includes
  every optional integration.
- CLI/SDK error messages now unwrap FastAPI `{"detail": ...}` bodies (including
  422 validation arrays) into concise human-readable text instead of raw JSON.
- **Breaking (SDK):** `dagnam.download_checkpoint(job_id)` with no
  `checkpoint_id` now selects the **latest** checkpoint by epoch/step (was
  best-then-latest). Pass `prefer_best=True` to restore the previous behavior.
  The `dagnam checkpoint download <job> best` CLI keyword maps to
  `prefer_best=True`.

### Changed

- Internal: the `load_dataset` implementation module moved from
  `dagnam._core.load` to `dagnam.data.load` so it sits in the `data` layer it
  composes (cache, dataset adapters, system loaders), enforcing the
  `cli -> resources -> data -> _core` import contract. The public entry point
  `dagnam.load_dataset` is unchanged; only the internal module path moved.

### Removed

- Internal compatibility-shim modules that only forwarded old import paths:
  the `dagnam.services` package, `dagnam._core._common` / `_resolver` / `_sse`,
  the `dagnam.data.loaders.*_loader` forwarders (`audio_loader`, `csv_loader`,
  `flax_loader`, `image_folder_loader`, `json_loader`, `system_loader`,
  `tf_loader`) and `data.loaders.media_utils`, and the redundant
  `dagnam/resources/datasets_upload.py` module. These were internal forwarders
  with no external consumers; the canonical modules (`dagnam.resources.*`,
  `dagnam._core.{client.common,resolver,sse}`, `dagnam.data.loaders.{audio,csv,
  flax,image_folder,json_array,system,tf,media}`) are unchanged, and the
  documented public API (`dagnam.*` exports, the `dagnam.resources.datasets_upload`
  alias) keeps resolving.

## [0.1.0] - 2026-05-10

First public PyPI release of the official Dagnam.AI Python SDK.

### Added

- Dataset loading with API-key authentication, metadata lookup, local caching,
  SHA-256 verification, LRU eviction, resumable downloads, presigned download
  URLs, and dataset version selection.
- `DagnamDataset` adapters for polars, PyTorch, TensorFlow, and Flax/JAX.
- Tabular CSV, TSV, JSON, and JSONL loaders with deterministic train/validation
  and test splits.
- Image-folder and audio-folder dataset loaders with archive extraction,
  deterministic fallback splits, and framework transform hooks.
- System dataset resolution by friendly name.
- Dataset upload helpers for local files and server-side URL ingestion.
- Inference helpers for single and batch prediction plus deployment health.
- Checkpoint download with checksum verification and a dedicated checkpoint
  cache.
- Synchronous SSE training stream iterator with reconnect support.
- Deployment, Model Hub, project, and code generation resource modules.
- `LongRunningOperation` for polling asynchronous platform operations.
- `dagnam.aio.AsyncDagnamClient` for low-level async API access.
- CLI commands for login, datasets, cache, inference, checkpoints, streams,
  deployments, hub, projects, and code generation.
- Typed package marker via `dagnam/py.typed`.

### Security

- `dagnam login` writes `~/.dagnam/config.json` with owner-only permissions on
  POSIX systems and creates `~/.dagnam` with owner-only directory permissions.
- Archive extraction rejects unsafe paths, symlinks, special files, and oversized
  archives before unpacking media datasets.
- Downloaded dataset and checkpoint filenames from `Content-Disposition` are
  sanitized before writing to disk.

### Changed

- Licensed the SDK under Apache License 2.0 with a root `NOTICE` file included
  in source and wheel distributions.

### Compatibility

- Python `>=3.12,<3.13`.
- Dagnam backend `>=0.5.0, <0.7.0`.
