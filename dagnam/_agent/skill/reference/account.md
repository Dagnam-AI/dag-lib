# Account reference

Identity, plan, usage, quota, and local config.

## CLI
- `dagnam login [--api-url URL] [--training-metrics-path PATH]` — interactive: authenticate and save credentials. (Tell the user to run this themselves; never fabricate a key.)
- `dagnam whoami` — resolved API URL, masked API key, and where the key came from.
- `dagnam version [--json]` — dagnam version, Python version, platform.
- `dagnam usage [--json] [--output]` — plan and real-time usage against limits.
- `dagnam logout` — remove the stored API key from config.
- `dagnam config list` — print all config values (api_key masked).
- `dagnam config get <key>` — print one config value.
- `dagnam config set <key> <value>` — set a supported config value (e.g. `training_metrics_path`).
- `dagnam config unset <key>` — unset a supported config value.

## SDK (`import dagnam`)
- `dagnam.account.entitlements() -> dict` — plan (`plan.display_name` / `plan.code`) and `limits` (a list of `{key, current, limit}`). Use this to surface cost/quota impact in the guardrail.
- `dagnam.account.storage_quota() -> dict` — storage usage vs. limit.
- `dagnam.account.api_key_usage(key_id) -> dict` — per-key usage.
- `dagnam.configure(api_key=..., api_url=...)`, `dagnam.get_api_key()`, `dagnam.get_api_url()`, `dagnam.get_config_value(key)`.

## Auth resolution order
explicit argument -> `dagnam.configure(...)` -> `DAGNAM_API_KEY` / `DAGNAM_API_URL` env ->
`~/.dagnam/config.json` -> default `https://api.dagnam.ai`.

## Security model

- **API keys are the programmatic credential** and are the only way the SDK/CLI
  authenticates. They are **unaffected by two-factor authentication (2FA)** — 2FA
  protects the interactive web login, not API-key requests. Manage keys from the
  web dashboard; a leaked key is revoked there.
- **Expensive actions require a verified email.** Creating a training job,
  uploading a dataset, and creating a deployment require the API key's owning
  account to have a **verified email address**. If it is not verified, these
  calls raise `dagnam.EmailNotVerifiedError` (an actionable message plus the
  verification link, surfaced only when the server returns an `https` URL);
  verify the email in the web app, then retry. Browsing,
  reading, and designing are unaffected.
- **Upload guards.** An upload larger than the server's per-request size cap
  raises `dagnam.PayloadTooLargeError` (a `QuotaExceededError` subclass).
- **A rejected source URL surfaces on one of two paths.** `upload_from_url`
  starts a server-side ingest job, so a bad URL can be caught either before or
  after the job is accepted:
  - **Synchronously**, if the server rejects the URL up front, the call raises
    `dagnam.InvalidURLError` (an `UploadError` subclass) and no job is created.
  - **Asynchronously**, if the URL is only found to be unusable once the ingest
    job runs (unreachable host, wrong content, a download that fails), the call
    still returns a `LongRunningOperation`; the rejection appears as the job's
    terminal failure state, so `op.wait(...).result()` raises
    `dagnam.LROFailedError` with the server's message as `.detail`.

  Handle both: supply a publicly reachable `https` URL to a real dataset file,
  and check the operation's result rather than assuming the return of
  `upload_from_url` means the dataset landed.
- **Account-status rejections.** If the API key's owning account is
  administratively suspended, calls raise `dagnam.AccountSuspendedError` — this
  is not self-clearing; contact support. A request from a blocked IP address
  raises the existing `dagnam.AuthError` (no dedicated exception type, since
  there is no different remediation an SDK caller can take for it). Both are
  `APIError` subclasses carrying `.status_code == 403`, as is
  `EmailNotVerifiedError`, so an existing `except dagnam.APIError` handler still
  catches them.
- **Lockout applies to interactive login only.** After repeated failed
  interactive-login attempts the account is temporarily locked out, and the
  login endpoint answers 423 — so `dagnam login` (and any direct login call)
  raises `dagnam.AccountLockedError` (an `APIError` subclass, `.status_code ==
  423`). It clears itself once the lockout window elapses; retry after waiting.
  API-key requests are not subject to the lockout and never raise it.
