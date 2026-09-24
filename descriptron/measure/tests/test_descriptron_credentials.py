"""Tests for descriptron_credentials (v2.0.4): lookup order, privacy of the file, and when it may ask."""
import os
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import descriptron_credentials as dc  # noqa: E402


@pytest.fixture
def env(tmp_path, monkeypatch):
    f = tmp_path / "cfg" / "credentials"
    monkeypatch.setenv("DESCRIPTRON_CREDENTIALS", str(f))
    for k in ("ANTHROPIC_API_KEY", "HF_TOKEN", "DESCRIPTRON_NO_KEY_PROMPT"):
        monkeypatch.delenv(k, raising=False)
    return f


def test_environment_wins_over_file(env, monkeypatch):
    dc.write_file({"HF_TOKEN": "from_file"})
    monkeypatch.setenv("HF_TOKEN", "from_env")
    assert dc.ensure_key("HF_TOKEN") == "from_env"
    assert dc.load_into_environment() == {}          # the file never overrides the environment


def test_file_is_used_and_private(env):
    dc.write_file({"ANTHROPIC_API_KEY": "sk-test"})
    assert stat.S_IMODE(os.stat(env).st_mode) == 0o600
    assert dc.ensure_key("ANTHROPIC_API_KEY") == "sk-test"
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-test"   # children inherit it


def test_no_prompt_without_terminal(env, monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    assert dc.ensure_key("HF_TOKEN") is None
    assert not env.exists()


def test_switch_off_blocks_even_the_gui_dialog(env, monkeypatch):
    monkeypatch.setenv("DESCRIPTRON_NO_KEY_PROMPT", "1")
    asked = []
    assert dc.ensure_key("HF_TOKEN", ask=lambda *a: asked.append(a) or "hf_x") is None
    assert asked == [] and not env.exists()


def test_dialog_answer_is_saved_once(env):
    calls = []
    v = dc.ensure_key("HF_TOKEN", ask=lambda n, p, w: calls.append(n) or " hf_abc ")
    assert v == "hf_abc" and dc.read_file()["HF_TOKEN"] == "hf_abc"
    os.environ.pop("HF_TOKEN")
    assert dc.ensure_key("HF_TOKEN", ask=lambda *a: calls.append("again") or "other") == "hf_abc"
    assert calls == ["HF_TOKEN"]                     # asked the first time only


def test_empty_answer_skips(env):
    assert dc.ensure_key("HF_TOKEN", ask=lambda *a: "") is None
    assert not env.exists()


def test_remove_and_list(env, capsys):
    dc.write_file({"HF_TOKEN": "hf_1", "ANTHROPIC_API_KEY": "sk_1"})
    assert dc.main(["--remove", "HF_TOKEN"]) == 0
    assert dc.read_file() == {"ANTHROPIC_API_KEY": "sk_1"}
    dc.main(["--list"])
    out = capsys.readouterr().out
    assert "sk_1" not in out and "ANTHROPIC_API_KEY" in out   # values are never printed


def test_file_format_tolerates_comments_and_quotes(env):
    env.parent.mkdir(parents=True)
    env.write_text('# comment\n\nHF_TOKEN="hf_q"\nBROKEN LINE\n')
    assert dc.read_file() == {"HF_TOKEN": "hf_q"}
