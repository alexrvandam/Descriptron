"""
descriptron_mcp.jobs — run a program now, or start it as a background job
=========================================================================

A tool call cannot block for the hours SAM2-PAL or a full pipeline takes, so a
long program is started as a job: detached from the server (it survives the
client closing), logging to a file, and writing its exit code when it ends.
Job state is plain files under the state directory, so a restarted server
still knows every job it started.

    DESCRIPTRON_MCP_STATE   state directory (default ~/.cache/descriptron-mcp)
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from .catalogue import Program, command_for, environment_for

# Popen objects of jobs started by THIS server process. They must be polled,
# otherwise a finished child stays a zombie and looks alive to os.kill(pid, 0).
_LIVE: dict[str, subprocess.Popen] = {}


def state_dir() -> Path:
    d = Path(os.environ.get("DESCRIPTRON_MCP_STATE", "~/.cache/descriptron-mcp")).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _tail(text: str, lines: int) -> str:
    parts = text.splitlines()
    return "\n".join(parts[-lines:]) if lines > 0 else ""


def output_dirs(args: list[str], workdir: Path) -> list[Path]:
    """Directories a program was told to write to: the value after any flag
    whose name contains 'out' (--output_dir, --output_base, --out, ...)."""
    dirs = []
    for i, a in enumerate(args[:-1]):
        if a.startswith("-") and "out" in a.lower():
            p = Path(args[i + 1]).expanduser()
            p = p if p.is_absolute() else workdir / p
            dirs.append(p if p.suffix == "" else p.parent)
    return dirs


def changed_files(dirs: list[Path], since: float, limit: int = 50, scan_cap: int = 20000) -> dict:
    seen, hits = 0, []
    for d in dirs:
        if not d.is_dir():
            continue
        for root, _, files in os.walk(d):
            for f in files:
                seen += 1
                if seen > scan_cap:
                    break
                p = Path(root) / f
                try:
                    if p.stat().st_mtime >= since - 1:
                        hits.append(str(p))
                except OSError:
                    pass
    hits.sort()
    return {"count": len(hits), "files": hits[:limit], "truncated": len(hits) > limit}


def run_now(program: Program, args: list[str], workdir: Path, timeout_s: int, tail_lines: int) -> dict:
    """Run to completion and return the tail of its output plus the files it wrote."""
    logs = state_dir() / "runs"
    logs.mkdir(exist_ok=True)
    log = logs / f"{datetime.now():%Y%m%d_%H%M%S}_{program.name}_{uuid.uuid4().hex[:6]}.log"
    cmd = command_for(program, args)
    start = time.time()
    try:
        proc = subprocess.run(cmd, cwd=workdir, env=environment_for(program), capture_output=True,
                              text=True, errors="replace", timeout=timeout_s)
        code, out, err, timed_out = proc.returncode, proc.stdout, proc.stderr, False
    except subprocess.TimeoutExpired as exc:
        code, timed_out = None, True
        out = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
    log.write_text(f"$ {' '.join(cmd)}\n# cwd {workdir}\n\n--- stdout ---\n{out}\n--- stderr ---\n{err}",
                   encoding="utf-8")
    result = {
        "program": program.name, "exit_code": code, "ok": code == 0,
        "seconds": round(time.time() - start, 1),
        "stdout_tail": _tail(out, tail_lines), "stderr_tail": _tail(err, tail_lines),
        "full_log": str(log),
        "files_written": changed_files(output_dirs(args, workdir) or [workdir], start),
    }
    if timed_out:
        result["note"] = (f"stopped after {timeout_s} s. Long programs should be run with start_job, "
                          "which runs in the background and can be polled.")
    return result


# --------------------------------------------------------------------- jobs --
def _job_dir(job_id: str) -> Path:
    if not job_id or any(c in job_id for c in "/\\.") :
        raise ValueError(f"not a job id: {job_id!r}")
    d = state_dir() / "jobs" / job_id
    if not d.is_dir():
        raise ValueError(f"no job {job_id!r}; list_jobs shows the known ones")
    return d


def start(program: Program, args: list[str], workdir: Path) -> dict:
    job_id = f"{datetime.now():%Y%m%d_%H%M%S}_{program.name[:40]}_{uuid.uuid4().hex[:6]}"
    d = state_dir() / "jobs" / job_id
    d.mkdir(parents=True)
    cmd = command_for(program, args)
    wrapper = [sys.executable, "-m", "descriptron_mcp._jobwrap", str(d / "exit_code"), "--", *cmd]
    kw: dict = {}
    if os.name == "posix":
        kw["start_new_session"] = True           # own process group: survives the client, cancellable as a group
    else:
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    with open(d / "log.txt", "wb") as log:
        proc = subprocess.Popen(wrapper, cwd=workdir, env=environment_for(program), stdout=log,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, **kw)
    _LIVE[job_id] = proc
    meta = {"job_id": job_id, "program": program.name, "package": program.package,
            "args": args, "command": cmd, "workdir": str(workdir), "pid": proc.pid,
            "started": time.time(), "started_iso": datetime.now().isoformat(timespec="seconds")}
    (d / "job.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return {"job_id": job_id, "pid": proc.pid, "log": str(d / "log.txt")}


def _alive(pid: int) -> bool:
    if os.name == "posix":
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout
    return str(pid) in out


def status(job_id: str, tail_lines: int = 20) -> dict:
    d = _job_dir(job_id)
    meta = json.loads((d / "job.json").read_text(encoding="utf-8"))
    proc = _LIVE.get(job_id)
    if proc is not None:
        proc.poll()                              # reap, so a finished job is not a zombie
    code_file = d / "exit_code"
    if code_file.exists():
        raw = code_file.read_text().strip()
        code = int(raw) if raw.lstrip("-").isdigit() else None
        state = "finished" if code == 0 else ("cancelled" if (d / "cancelled").exists() else "failed")
    elif proc is not None and proc.returncode is None:
        state, code = "running", None
    elif proc is None and _alive(meta["pid"]):
        state, code = "running", None
    else:
        state, code = ("cancelled" if (d / "cancelled").exists() else "lost"), None
    log_text = (d / "log.txt").read_text(encoding="utf-8", errors="replace")
    end = code_file.stat().st_mtime if code_file.exists() else time.time()
    out = {"job_id": job_id, "program": meta["program"], "state": state, "exit_code": code,
           "elapsed_s": round(end - meta["started"], 1), "started": meta["started_iso"],
           "log": str(d / "log.txt"), "log_tail": _tail(log_text, tail_lines)}
    if state in ("finished", "failed"):
        out["files_written"] = changed_files(
            output_dirs(meta["args"], Path(meta["workdir"])) or [Path(meta["workdir"])], meta["started"])
    if state == "lost":
        out["note"] = "the process is gone but left no exit code (killed from outside, or the machine restarted)"
    return out


def log(job_id: str, tail_lines: int = 200) -> str:
    text = (_job_dir(job_id) / "log.txt").read_text(encoding="utf-8", errors="replace")
    return _tail(text, tail_lines)


def cancel(job_id: str) -> dict:
    d = _job_dir(job_id)
    meta = json.loads((d / "job.json").read_text(encoding="utf-8"))
    if (d / "exit_code").exists():
        return status(job_id, 5)
    (d / "cancelled").touch()
    pid = meta["pid"]
    try:
        if os.name == "posix":
            os.killpg(pid, signal.SIGTERM)       # the wrapper leads its own group
        else:
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    except ProcessLookupError:
        pass
    proc = _LIVE.get(job_id)
    if proc is not None:
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(pid, signal.SIGKILL)
    return status(job_id, 5)


def all_jobs(limit: int = 30) -> list[dict]:
    root = state_dir() / "jobs"
    if not root.exists():
        return []
    ids = sorted((p.name for p in root.iterdir() if (p / "job.json").exists()), reverse=True)[:limit]
    rows = []
    for j in ids:
        s = status(j, 0)
        rows.append({k: s[k] for k in ("job_id", "program", "state", "exit_code", "elapsed_s", "started")})
    return rows
