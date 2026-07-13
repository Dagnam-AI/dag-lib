# Security Policy

## Supported Versions

Security fixes are provided for the current public release line.

| Version | Supported |
| --- | --- |
| `0.7.x` | yes |
| `< 0.7` | no (please upgrade) |

## Reporting a Vulnerability

Do not open a public GitHub issue for suspected vulnerabilities.

Email security reports to `security@dagnam.ai` with:

- A description of the issue and affected SDK version
- Reproduction steps or proof of concept, if available
- Impact assessment, including whether credentials, datasets, checkpoints, or
  generated code could be exposed or modified
- Any relevant logs or stack traces with secrets removed

We will acknowledge receipt as soon as practical, investigate privately, and
coordinate a fix and disclosure timeline based on severity.

## Secret Handling

Never include real API keys, presigned dataset URLs, private checkpoint URLs, or
customer data in bug reports, tests, screenshots, logs, or pull requests.

The SDK looks for credentials in explicit arguments, `DAGNAM_API_KEY`, and
`~/.dagnam/config.json`. Treat that config file as sensitive.
