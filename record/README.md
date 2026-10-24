# dagnam-record

A small recorder for the LLM calls a program makes. It writes each `openai` and `anthropic` call to
a private file on your machine so that you can later hand that file to a Dagnam.AI audit. It is a
separate package from the `dagnam` SDK, has no dependencies (standard library only), needs Python
3.9 or newer, and ships type information (`py.typed`).

**It records your prompts and the model's answers in clear text.** Read [What is stored](#what-is-stored)
before you switch it on in anything that handles personal or confidential data.

## Quick start

```bash
pip install dagnam-record
```

```python
import dagnam_record

dagnam_record.install()  # once, before the first openai / anthropic call; returns True if an SDK was patched
# ... run your code as usual: every call is recorded under ~/.dagnam/traces ...
```

Look at what was recorded (and what was not):

```bash
python -m dagnam_record status
```

Nothing is sent anywhere. To use the file, see [Using the file](#using-the-file).

## What is stored

Everything is written to `~/.dagnam/traces` unless you pass `dir=`:

- One JSON line per call, in a file per UTC day named `YYYY-MM-DD.jsonl`, plus `.status.json`
  (counters, the start of the window, the SDK versions seen).
- Each line holds the call's **raw request body, the raw response body and a normalized copy**
  (`norm`: model, messages as text, the answer, its tool calls, token counts) with the time, the
  latency and the HTTP status.
- **The text is stored as it was**: system prompts, user messages, tool results, answers, the
  arguments of tool calls. If a prompt contains an email address, a customer record or a secret,
  so does the file, unless you redact it first (see [Redaction](#redaction)).
- **Never stored:** request headers, API keys, organisation ids, anything passed as `extra_headers`,
  `extra_query`, `extra_body`, `timeout`, `api_key` or `auth_token`, and the client's own settings.
  The two credential fields an Anthropic request body can carry (`authorization_token` of an MCP
  server, `fallback_credit_token`) are replaced by `<redacted>`.
- Images, audio, documents and other media in a request or answer are replaced by
  `{"type": "media_omitted", "bytes": N}`: the file says that media was there and how big it was,
  not what it was. Your program still gets the real payload.
- The directory is created with mode `0700` and the files with `0600`, so other users of the machine
  cannot read them. That is the only protection: the files are not encrypted, and anything that
  copies your home directory (a backup, a synced folder) copies them too.

## Nothing leaves your machine

`dagnam-record` contains no network code: it never opens a connection, and a test fails the build if
it imports a networking module. It only writes the file. Uploading is a separate step that you start
yourself with the `dagnam` command line (below), and that step asks you to confirm.

## How long it records

Recording stops by itself **14 days after the first `install()` in a directory** (`window_days`,
1 to 365). The recorder then logs one INFO line, counts later calls as `dropped_expired` and your
program carries on. To record again, run `python -m dagnam_record purge` (which also forgets the
start of the window) or choose another `dir`.

Old files are removed **only when the recorder is set up**, that is when `install()` runs with new
or first-time options: daily files dated more than `window_days` days before today (UTC) are deleted
then, and never at midnight in a long-running process. **Upload within the window**: a program
restarted on day 15 deletes day 0's file before you have used it.

## Switching it off

Set `DAGNAM_RECORD` to `0`, `false`, `off`, `no`, `disable` or `disabled` (any case, spaces ignored)
and `install()` does nothing, validates nothing, creates nothing and returns `False`; `record_call()`
is silent too. Any other value, including unset, empty, `1` and `true`, leaves the recorder on. The
variable is read when `install()` runs, not on every call.

## `install()`

```python
dagnam_record.install(
    window_days=14,            # whole days, 1 to 365
    dir="~/.dagnam/traces",    # str or path; "~" is expanded; a relative path is fixed when install() runs
    sample=1.0,                # share of calls kept, above 0 and at most 1
    no_raw=False,              # True: write only the normalized copy, never the request or response
    redact=None,               # None, a function from text to text, or "auto"
)  # -> bool
```

- **Returns `True`** when at least one SDK (`openai`, `anthropic`) was patched, `False` when neither
  is installed or the kill switch is on. A missing SDK is not an error, and the recorder is set up
  either way, so `record_call()` works.
- **Raises `ValueError`** for an option out of range (nothing changes, no directory is made), and
  `ImportError` for `redact="auto"` without the extra. It is not meant to raise for anything else:
  an SDK that fails to import is skipped with one logged warning, and a directory that cannot be
  used is logged and counted.
- **Calling it again is safe**, from any thread. With the same options it only patches an SDK that
  has appeared since (so call it again after importing an SDK late); with different options the
  recorder is reconfigured and the old writer hands over what it still holds. A directory it cannot
  use leaves a running recorder as it was.
- `sample` keeps a fixed share of calls, chosen from a hash of each call's id.
- `dagnam_record.is_installed()` says whether a recorder is running. `dagnam_record.uninstall()`
  puts the SDKs' own methods back, writes what is queued and stops; it is safe to call twice.
- A relative `dir` is made absolute when `install()` runs, so a later `os.chdir` does not move it.

## Calls the SDKs do not make: `record_call`

For a framework that talks to a provider without these SDKs, record the bodies yourself after
`install()`:

```python
dagnam_record.record_call(
    provider="openai",          # "openai", "anthropic" or "other"
    endpoint="chat.completions",  # "chat.completions", "responses", "messages" or "custom"
    request={...},              # the JSON request body
    response={...},             # the JSON response body
    latency_ms=412.0,
    status=200,                 # use 599 for "no HTTP response"
)
```

It never raises: a bad call is counted instead. It goes through the same sampling, redaction, size
limit and writer as a patched call. Before `install()` it records nothing and logs one warning.
A `provider` of `"other"` is written without the normalized copy, because there is no shape to
normalize.

## Using the file

The `dagnam` command line reads it:

```bash
dagnam audit scan --source dagnam ~/.dagnam/traces/2026-10-24.jsonl
```

The `dagnam` source arrives with the first `dagnam` release after 0.22.1. If `dagnam audit scan
--help` does not list `dagnam` among the sources, upgrade `dagnam` first. The reader takes one file;
to scan several days, join them: `cat ~/.dagnam/traces/*.jsonl > traces.jsonl`. Only calls that
returned HTTP 200 are used. The scan itself makes no network call; `dagnam audit run` is the step
that uploads the rows it derived, after you confirm.

## `status` and `purge`

```bash
python -m dagnam_record [--dir DIR] status
python -m dagnam_record [--dir DIR] purge [--dry-run]
```

`--dir` may come before or after the command and defaults to `~/.dagnam/traces`.

- `status` prints the directory, the number of files and lines, the first and last time, when the
  window started and whether it has ended, and the six counters below. It never prints the content
  of a call.
- `purge` deletes the recorder's own files in the directory (the dated `.jsonl` files and
  `.status.json`) and nothing else. With `--dry-run` it only lists them. It refuses a directory that
  holds no recorder file, and a symlinked directory.
- **Exit codes:** `0` done; `1` nothing to show, or a purge that was refused or could not delete a
  file (the reason is on stderr); `2` a usage error.
- Purging while a program is recording is safe: the writer notices the deleted file and starts a
  new one with its next line. Stop the program first if you want the directory to stay empty.
- Counts are for what the recorder did, not for what is on disk: `written` still counts lines that
  you purged later. If several processes write to one directory they overwrite each other's
  `.status.json` (the data lines are never mixed up), so `status` shows whatever was saved last.

## Which SDK versions

| SDK | Range the recorder is built and tested for |
|---|---|
| `openai` | `>=1.66,<3` |
| `anthropic` | `>=0.40,<1` |

The tests run on Python 3.9, 3.11, 3.12 and 3.13, with the lowest (`openai` 1.66.0, `anthropic`
0.40.0) and the highest releases inside those ranges. Outside a range, the SDK is still recorded, but
**only as the normalized copy** (no raw request or response), with one logged warning, and the version
is saved in `.status.json`. `record_call()` is not narrowed by this, because you supply the bodies.

## What is recorded and what is not

Recorded, for the sync and the async clients (also `AzureOpenAI`, `AnthropicBedrock` and
`AnthropicVertex`, which use the same classes; Bedrock streaming was not exercised against a real
AWS stream):

- OpenAI: `chat.completions.create` and `responses.create`, with or without `stream=True`, and the
  `parse()` and `stream()` helpers where the SDK has them (`beta.chat.completions` in older versions).
- Anthropic: `messages.create` with or without `stream=True`, the `messages.stream()` helper, `parse()`,
  the `beta` twins of these, and the requests a tool runner makes. `betas` is kept in the request.
- A **failed call** is recorded too (the HTTP status of the error, or `599` when no response came:
  a refused connection, a timeout), and the error reaches your program unchanged.

Streams are recorded **once, when the end of the stream was seen** (the final chunk with a finish
reason for Chat Completions, `response.completed`/`incomplete`/`failed` for Responses,
`message_stop` for Anthropic), from the pieces your program read. A stream abandoned or cut off
before that point is not written, because a partial answer is no training row: it is counted as
`dropped_partial`. An OpenAI Chat Completions stream you stop reading right after the finish
chunk is recorded without the usage chunk that follows, so its token counts are 0.

Not recorded:

- `with_raw_response` and `with_streaming_response` calls (the call works; it is counted as
  `dropped_error`). `count_tokens`, batches and other endpoints are not wrapped and not counted.
- Anything passed as `extra_body`, which some vendors use for real body parameters.
- Calls made by a framework that does not use these SDKs. Use `record_call`.

An `anthropic` before 0.40 or an `openai` before 1.66 is outside the range. On older `anthropic`
releases (0.40.0 is the one tested), a `messages.stream()` helper that you leave early is counted
(`dropped_partial`), or a finished one is written, when the stream is garbage collected or the
program exits, not at the moment you leave the `with` block.

## Limits and counters

A line may not exceed 1,000,000 bytes. Queued lines wait in memory: at most 1000, so a worst case is
on the order of a gigabyte for very large bodies. Tool-heavy conversations reach the line limit
sooner than you may expect, because the normalized copy repeats the earlier tool calls next to the
raw request; use `no_raw=True` or `sample` if that bites.

The six counters in `status`:

| counter | meaning |
|---|---|
| `written` | lines written |
| `dropped_full` | the in-memory queue was full (a very busy program or a stuck disk) |
| `dropped_large` | the line would be over 1,000,000 bytes |
| `dropped_error` | anything else that went wrong: a body that could not be converted, a `redact` function that raised, an unusable directory, a `with_raw_response` call |
| `dropped_expired` | calls after the 14-day window ended |
| `dropped_partial` | a call or stream the recorder saw only part of (abandoned, cut off, cancelled, interrupted) |

Status `599` in a line means the call got no HTTP response at all.

## It never breaks your program, within limits

The recorder runs inside your calls and is built to fail open: an error in it is counted and the
call goes on, with the same result or the same exception your program would have had. Writing is
done by one background thread through a bounded queue, so a slow or full disk never blocks a call.
It prints nothing in normal operation and changes no global state (logging, signals, hooks).

What it cannot promise:

- **Lines still queued are lost if the process ends without running Python's exit hooks**: `os._exit`,
  the default action of `SIGTERM` (when your program installs no handler), a kill, and the workers
  that `multiprocessing` starts (they end with `os._exit`). Normally the queue holds a few
  milliseconds of calls; call `dagnam_record.uninstall()` before such an exit to write them.
- A finished stream that you drop without closing is written at the next safe point: your next
  recorded call, the writer's idle check (up to 30 seconds), `uninstall()` or the end of the program.
- `redact` runs on the thread that makes the call (only the file write is asynchronous), so a slow
  function slows each call. Building the line, including the conversion of the request, also runs
  there, before the request is sent.
- `install()` itself does one-time file work (a directory, a thread) and can raise `ValueError`.
- A forked child gets a fresh queue and zero counts; two processes sharing one directory share its
  `.status.json`, last writer wins.

## Redaction

`redact` is called twice per call, with the request and then the response as JSON text, **before**
the line and its normalized copy are built, and must return JSON text of an object:

```python
def scrub(text: str) -> str:
    return text.replace("ACME-INTERNAL", "[internal]")

dagnam_record.install(redact=scrub)
```

If it raises, or returns something else, the call is **dropped and counted** (`dropped_error`), never
written unredacted. `redact="auto"` uses the PII redactor of `dagnam-contracts`:

```bash
pip install "dagnam-record[redact]"
```

It is pattern based: it replaces what it recognizes (email addresses, phone numbers, card and bank
numbers, national ids, IP addresses, dates of birth, credential-like values) and cannot recognize
names, addresses in free text or confidential business content. Look at a sample of your own file
before relying on it. `dagnam-contracts` needs Python 3.12 or newer, so `redact="auto"` is only
available there; a function of your own works on every supported Python.

## Licence

Apache-2.0, see `LICENSE` and `NOTICE`.
