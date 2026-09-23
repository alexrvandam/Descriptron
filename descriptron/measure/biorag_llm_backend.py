#!/usr/bin/env python3
"""
biorag_llm_backend.py — pluggable LLM backends for BioRAG
=========================================================

Two interchangeable backends expose the SAME call used throughout BioRAG:

    client.messages.create(model=..., max_tokens=..., system=..., messages=[...])
    -> response.content[0].text, response.stop_reason, response.usage

  api          Anthropic Messages API (pay per token; needs ANTHROPIC_API_KEY)
  claude-code  the Claude Code CLI in headless mode (`claude -p`), which uses
               the user's Claude subscription instead of API credit. Images
               are passed as base64 content blocks through
               --input-format stream-json, the system prompt through
               --system-prompt-file, and all tools are disabled so the call
               behaves like a plain API request.

Every call is appended to a JSONL provenance log (model actually used,
backend, tokens, duration) when a log path is set.

Also provides load_prompt_library() for the universal system-prompt file.

Usage
-----
    from biorag_llm_backend import make_llm_client, load_prompt_library
    client = make_llm_client("claude-code")            # or "api"
    r = client.messages.create(model="claude-sonnet-4-6", max_tokens=4000,
                               system="You are ...",
                               messages=[{"role": "user", "content": "Hi"}])
    print(r.content[0].text)

Requirements for claude-code: Claude Code installed and logged in
(`claude` on PATH, or set CLAUDE_BIN / --claude-bin).
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# Environment variables set by a parent Claude Code session. They are removed
# for the child process so `claude -p` runs as an independent session.
_PARENT_SESSION_VARS = (
    "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_SESSION_ID",
    "CLAUDE_CODE_CHILD_SESSION", "CLAUDE_CODE_MESSAGING_SOCKET",
    "CLAUDE_CODE_MESSAGING_TOKEN", "CLAUDE_PID", "CLAUDE_CODE_SESSION_ATTENDED",
    "CLAUDE_CODE_EXECPATH",
)

_LOG_LOCK = threading.Lock()


@dataclass
class _Text:
    text: str
    type: str = "text"


@dataclass
class _Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class LLMResponse:
    content: List[_Text]
    stop_reason: str = "end_turn"
    usage: _Usage = field(default_factory=_Usage)
    model: str = ""
    backend: str = ""


def _log_call(log_path: Optional[str], record: Dict):
    if not log_path:
        return
    with _LOG_LOCK:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")


def cli_model_alias(model: str, override: Optional[str] = None) -> str:
    """Map an API model id to a Claude Code model argument."""
    if override:
        return override
    m = (model or "").lower()
    for fam in ("opus", "sonnet", "haiku", "fable"):
        if fam in m:
            return fam
    return model or "sonnet"


class _ClaudeCodeMessages:
    def __init__(self, parent: "ClaudeCodeClient"):
        self._p = parent

    def create(self, model: str = "", max_tokens: int = 0, system=None,
               messages: Optional[List[Dict]] = None, **kwargs) -> LLMResponse:
        p = self._p
        messages = messages or []
        system_text = system if isinstance(system, str) else \
            "\n".join(b.get("text", "") for b in (system or []) if isinstance(b, dict))
        # Flatten earlier turns into the final user turn (single-shot call).
        content_blocks: List[Dict] = []
        history = []
        for msg in messages[:-1]:
            txt = msg["content"] if isinstance(msg["content"], str) else \
                "\n".join(b.get("text", "") for b in msg["content"] if b.get("type") == "text")
            history.append(f"[{msg['role'].upper()}]\n{txt}")
        last = messages[-1]["content"] if messages else ""
        if history:
            content_blocks.append({"type": "text", "text": "Conversation so far:\n"
                                   + "\n\n".join(history) + "\n\n[USER]"})
        if isinstance(last, str):
            content_blocks.append({"type": "text", "text": last})
        else:
            for b in last:
                if b.get("type") in ("text", "image"):
                    content_blocks.append(b)
        user_line = json.dumps({"type": "user",
                                "message": {"role": "user", "content": content_blocks}})

        alias = cli_model_alias(model, p.model_override)
        last_err = ""
        for attempt in range(1, p.max_retries + 1):
            t0 = time.time()
            with tempfile.TemporaryDirectory(prefix="biorag_cc_") as td:
                sys_file = Path(td) / "system.txt"
                sys_file.write_text(system_text or "")
                cmd = [p.claude_bin, "-p",
                       "--input-format", "stream-json",
                       "--output-format", "stream-json", "--verbose",
                       "--system-prompt-file", str(sys_file),
                       "--model", alias,
                       "--tools", "",
                       "--no-session-persistence",
                       "--setting-sources", "",
                       "--strict-mcp-config",
                       "--disable-slash-commands"]
                env = {k: v for k, v in os.environ.items()
                       if k not in _PARENT_SESSION_VARS}
                try:
                    proc = subprocess.run(cmd, input=user_line + "\n", text=True,
                                          capture_output=True, timeout=p.timeout,
                                          cwd=td, env=env)
                except subprocess.TimeoutExpired:
                    last_err = f"timeout after {p.timeout}s"
                    time.sleep(p.retry_delay * attempt)
                    continue
            result = None
            for line in proc.stdout.splitlines():
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get("type") == "result":
                    result = ev
            dt = time.time() - t0
            if result is None or result.get("is_error"):
                last_err = (result or {}).get("result") or proc.stderr[-500:] or \
                    f"no result (exit {proc.returncode})"
                _log_call(p.log_path, {"backend": "claude-code", "ok": False,
                                       "attempt": attempt, "error": last_err,
                                       "model_arg": alias, "seconds": round(dt, 1)})
                # rate limit / overload -> longer back-off
                wait = p.retry_delay * attempt * (6 if re.search(
                    r'rate|limit|overload|529|429', str(last_err), re.I) else 1)
                time.sleep(wait)
                continue
            usage = result.get("usage", {}) or {}
            used_models = list((result.get("modelUsage") or {}).keys())
            text = result.get("result", "") or ""
            _log_call(p.log_path, {
                "backend": "claude-code", "ok": True, "attempt": attempt,
                "model_arg": alias, "models_used": used_models,
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "cost_usd_equivalent": result.get("total_cost_usd"),
                "seconds": round(dt, 1), "stop_reason": result.get("stop_reason"),
            })
            return LLMResponse(content=[_Text(text)],
                               stop_reason=result.get("stop_reason") or "end_turn",
                               usage=_Usage(usage.get("input_tokens", 0) or 0,
                                            usage.get("output_tokens", 0) or 0),
                               model=",".join(m for m in used_models if "haiku" not in m)
                               or alias,
                               backend="claude-code")
        raise RuntimeError(f"claude-code backend failed after {p.max_retries} attempts: {last_err}")


class ClaudeCodeClient:
    """Drop-in for anthropic.Anthropic() that routes calls through `claude -p`."""

    def __init__(self, claude_bin: Optional[str] = None, model_override: Optional[str] = None,
                 timeout: int = 900, max_retries: int = 3, retry_delay: float = 10.0,
                 log_path: Optional[str] = None):
        self.claude_bin = claude_bin or os.environ.get("CLAUDE_BIN") or shutil.which("claude")
        if not self.claude_bin:
            raise EnvironmentError(
                "Claude Code CLI not found. Install Claude Code and log in, "
                "or pass --claude-bin /path/to/claude.")
        self.model_override = model_override
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.log_path = log_path
        self.messages = _ClaudeCodeMessages(self)


class _LoggedAPIMessages:
    def __init__(self, inner, log_path):
        self._inner = inner
        self._log = log_path

    def create(self, **kwargs):
        t0 = time.time()
        r = self._inner.messages.create(**kwargs)
        u = getattr(r, "usage", None)
        _log_call(self._log, {"backend": "api", "ok": True, "model": kwargs.get("model"),
                              "input_tokens": getattr(u, "input_tokens", None),
                              "output_tokens": getattr(u, "output_tokens", None),
                              "seconds": round(time.time() - t0, 1),
                              "stop_reason": getattr(r, "stop_reason", None)})
        return r


class LoggedAnthropicClient:
    def __init__(self, api_key: Optional[str] = None, log_path: Optional[str] = None,
                 base_url: Optional[str] = None):
        import anthropic
        kw = {}
        if api_key:
            kw["api_key"] = api_key
        if base_url:
            kw["base_url"] = base_url            # any Anthropic-compatible endpoint
        inner = anthropic.Anthropic(**kw)
        self.messages = _LoggedAPIMessages(inner, log_path)


def models_in_log(log_path) -> Dict[str, int]:
    """Which models ANSWERED the calls in a provenance log, and how many calls each took part in.

    A report must name the model that ran, not the one that was asked for. Through a
    subscription the two differ: `--model claude-sonnet-4-6` is passed to the command line as
    the alias `sonnet`, which resolves to whichever Sonnet is current, and every report in this
    project that printed the requested id therefore named a model that never saw the data.
    The command-line client also runs a small model for its own housekeeping; it is listed
    here because it is in the log, and `answering_model` leaves it out."""
    out: Dict[str, int] = {}
    try:
        with open(log_path) as f:
            for line in f:
                try:
                    j = json.loads(line)
                except ValueError:
                    continue
                if not j.get("ok"):
                    continue
                for m in (j.get("models_used") or ([j["model"]] if j.get("model") else [])):
                    out[m] = out.get(m, 0) + 1
    except OSError:
        pass
    return out


def answering_model(log_path) -> Optional[str]:
    """The model that wrote the replies: the most used one, ignoring the client's small helper."""
    used = models_in_log(log_path)
    main = {m: n for m, n in used.items() if "haiku" not in m.lower()} or used
    return max(main, key=main.get) if main else None


def client_from_args(args, log_path: Optional[str] = None):
    """Build the client from the standard flags, so every step accepts the same options:
    a Claude Code subscription (the default), an API key, another provider's endpoint via
    --api-base-url, or no model at all."""
    import os
    key_env = getattr(args, "api_key_env", None) or "ANTHROPIC_API_KEY"
    return make_llm_client(getattr(args, "llm_backend", "claude-code"),
                           api_key=os.environ.get(key_env),
                           claude_bin=getattr(args, "claude_bin", None),
                           cc_model=getattr(args, "cc_model", None),
                           base_url=(getattr(args, "api_base_url", None)
                                     or os.environ.get("ANTHROPIC_BASE_URL")),
                           log_path=log_path or getattr(args, "llm_log", None))


def make_llm_client(backend: str = "claude-code", api_key: Optional[str] = None,
                    claude_bin: Optional[str] = None, cc_model: Optional[str] = None,
                    log_path: Optional[str] = None, timeout: int = 900,
                    base_url: Optional[str] = None):
    backend = (backend or "api").lower()
    if backend in ("claude-code", "claude_code", "cc", "subscription"):
        return ClaudeCodeClient(claude_bin=claude_bin, model_override=cc_model,
                                timeout=timeout, log_path=log_path)
    if backend == "api":
        return LoggedAnthropicClient(api_key=api_key, log_path=log_path, base_url=base_url)
    raise ValueError(f"Unknown LLM backend: {backend} (use 'api' or 'claude-code')")


def add_backend_args(parser):
    """Standard CLI flags shared by all BioRAG v2 scripts."""
    parser.add_argument("--llm-backend", default="claude-code",
                        choices=["api", "claude-code", "none"],
                        help="'claude-code' (default) = a Claude Code subscription via `claude -p`, which "
                             "is what this workflow was built and run with; 'api' = an API key, for anyone "
                             "without a subscription or wanting another provider (see --api-base-url); "
                             "'none' = no model at all, deterministic output only")
    parser.add_argument("--api-base-url", default=None,
                        help="With --llm-backend api: the endpoint to call. Anything exposing the "
                             "Anthropic messages API works, so a different provider or a local model "
                             "behind a compatible gateway can be substituted without touching the code "
                             "(default: $ANTHROPIC_BASE_URL, else Anthropic's own)")
    parser.add_argument("--api-key-env", default="ANTHROPIC_API_KEY",
                        help="Name of the environment variable holding the API key, so a provider using "
                             "its own variable needs no edit")
    parser.add_argument("--claude-bin", default=None,
                        help="Path to the Claude Code CLI (default: `claude` on PATH or $CLAUDE_BIN)")
    parser.add_argument("--cc-model", default=None,
                        help="Model alias for the claude-code backend (sonnet|opus|haiku or a full id); "
                             "default derives from --model")
    parser.add_argument("--llm-log", default=None,
                        help="JSONL provenance log of every LLM call (default: <output>/llm_calls.jsonl)")
    parser.add_argument("--system-prompts", default=None,
                        help="Universal system-prompt file (default: biorag_prompts/biorag_system_prompts_v2.txt; v1 reproduces the first BioRAG v2 run)")
    parser.add_argument("--taxon-profile", default=None,
                        help="Taxon profile YAML with names, structure terms, ratio definitions, questions")
    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Prompt library
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PROMPTS_FILE = Path(__file__).parent / "biorag_prompts" / "biorag_system_prompts_v2.txt"


class PromptLibrary:
    def __init__(self, sections: Dict[str, str], path: str):
        self.sections = sections
        self.path = path

    def get(self, name: str, **fields) -> str:
        if name not in self.sections:
            raise KeyError(f"Prompt section [{name}] not found in {self.path}")
        text = self.sections[name]
        values = {"evidence_policy": self.sections.get("evidence_policy", "")}
        values.update(fields)

        class _Keep(dict):
            def __missing__(self, key):
                return "{" + key + "}"
        return text.format_map(_Keep(values)).strip()

    def raw(self, name: str) -> str:
        return self.sections[name]


def load_prompt_library(path: Optional[str] = None) -> PromptLibrary:
    p = Path(path) if path else DEFAULT_PROMPTS_FILE
    sections: Dict[str, str] = {}
    current = None
    buf: List[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        m = re.match(r'^###\s*\[([\w\-]+)\]\s*$', line)
        if m:
            if current:
                sections[current] = "\n".join(buf).strip("\n")
            current, buf = m.group(1), []
            continue
        if current is None:
            continue           # header comments
        buf.append(line)
    if current:
        sections[current] = "\n".join(buf).strip("\n")
    return PromptLibrary(sections, str(p))


def parse_json_response(text: str) -> Dict:
    """Parse a JSON object from an LLM reply (tolerates fences / preamble)."""
    t = (text or "").strip()
    t = re.sub(r"^```(?:json(?:-ld)?)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        start = t.find("{")
        if start >= 0:
            obj, _end = json.JSONDecoder().raw_decode(t[start:])
            return obj
        raise


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Smoke-test a BioRAG LLM backend")
    ap.add_argument("--llm-backend", default="claude-code")
    ap.add_argument("--model", default="claude-sonnet-4-6")
    args = ap.parse_args()
    lib = load_prompt_library()
    c = make_llm_client(args.llm_backend)
    r = c.messages.create(model=args.model, max_tokens=200,
                          system=lib.get("taxonomist_persona"),
                          messages=[{"role": "user", "content":
                                     'Reply with JSON {"ok": true, "sections": %d}'
                                     % len(lib.sections)}])
    print(r.model, r.content[0].text)
