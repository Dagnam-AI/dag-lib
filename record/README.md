# dagnam-record

A local recorder for the workload of a training run. It is a separate package from the
`dagnam` SDK so that it can sit next to any training code without pulling anything in.

- **No dependencies.** The standard library only; Python 3.9 and newer.
- **No network.** The package never opens a connection. It writes a file on your machine, and
  nothing leaves it until you upload that file yourself.
- **Typed.** It ships `py.typed`.

This is an early scaffold; the recorder itself arrives in later releases.

## Licence

Apache-2.0, see `LICENSE` and `NOTICE`.
