"""Runs one job's command and records its exit code, so status survives a server restart.

    python -m descriptron_mcp._jobwrap <exit_code_file> -- <command ...>
"""
import os
import signal
import subprocess
import sys


def main() -> int:
    exit_file, sep, *cmd = sys.argv[1:]
    if sep != "--" or not cmd:
        raise SystemExit(__doc__)
    child = subprocess.Popen(cmd)

    def forward(signum, _frame):
        child.send_signal(signum)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, forward)
    code = child.wait()
    tmp = exit_file + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(str(code))
    os.replace(tmp, exit_file)                  # atomic: status never reads half a number
    return code


if __name__ == "__main__":
    sys.exit(main())
