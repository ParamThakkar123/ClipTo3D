"""Entry point shim.

This used to be `print("Hello from cliptoworld!")` — the only thing in the repo
that looked like an entry point, and it did nothing (MPO-245). The real CLI is
`cli.py`, installed as the `clipto3d` command.
"""

from cli import main

if __name__ == "__main__":
    raise SystemExit(main())
