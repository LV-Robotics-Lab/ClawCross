"""Read and message remote Claude Code background-agent sessions over SSH.

Claude Code background agents cannot be resumed with ``claude --resume`` while
owned by the agents daemon. Session discovery and transcript reads use files on
the remote host; replies go through the daemon control socket's ``reply`` op.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import os
import shlex
import subprocess
from typing import Any, Iterable

from utils.runtime_paths import DATA_DIR


DEFAULT_REMOTE_HOST = ""
DEFAULT_REMOTE_USER = ""
CACHE_FILENAME = "remote_claude_sessions_cache.json"
TARGETS_FILENAME = "remote_claude_targets.json"
REMOTE_KEY_SEPARATOR = "::"
ACPX_SESSION_PREFIX = "acpx:"
DEFAULT_REMOTE_ACPX_TOOLS = "codex,claude"


@dataclass(frozen=True)
class RemoteClaudeConfig:
    host: str
    user: str
    ssh_binary: str = "ssh"
    timeout_sec: float = 8.0
    connect_timeout_sec: int = 4
    hostname: str = ""

    @property
    def enabled(self) -> bool:
        return bool(self.host and self.user)

    @property
    def destination(self) -> str:
        return f"{self.user}@{self.host}"


def load_remote_claude_config() -> RemoteClaudeConfig:
    """Load remote Claude SSH settings from the environment.

    Remote hosts come from CLAWCROSS_REMOTE_CLAUDE_TARGETS, the local target
    registry, or CLAWCROSS_REMOTE_CLAUDE_HOST / USER. There is no project-specific
    baked-in default host.
    """

    host = (
        os.environ.get("CLAWCROSS_REMOTE_CLAUDE_HOST")
        or os.environ.get("REMOTE_CLAUDE_HOST")
        or DEFAULT_REMOTE_HOST
    ).strip()
    user = (
        os.environ.get("CLAWCROSS_REMOTE_CLAUDE_USER")
        or os.environ.get("REMOTE_CLAUDE_USER")
        or DEFAULT_REMOTE_USER
    ).strip()
    ssh_binary = (
        os.environ.get("CLAWCROSS_REMOTE_CLAUDE_SSH_BINARY")
        or os.environ.get("REMOTE_CLAUDE_SSH_BINARY")
        or "ssh"
    ).strip()
    timeout_raw = (
        os.environ.get("CLAWCROSS_REMOTE_CLAUDE_TIMEOUT_SEC")
        or os.environ.get("REMOTE_CLAUDE_TIMEOUT_SEC")
        or "30"
    )
    connect_timeout_raw = (
        os.environ.get("CLAWCROSS_REMOTE_CLAUDE_CONNECT_TIMEOUT_SEC")
        or os.environ.get("REMOTE_CLAUDE_CONNECT_TIMEOUT_SEC")
        or "10"
    )
    try:
        timeout_sec = max(1.0, float(timeout_raw))
    except ValueError:
        timeout_sec = 30.0
    try:
        connect_timeout_sec = max(1, int(connect_timeout_raw))
    except ValueError:
        connect_timeout_sec = 10
    return RemoteClaudeConfig(
        host=host,
        user=user,
        ssh_binary=ssh_binary,
        timeout_sec=timeout_sec,
        connect_timeout_sec=connect_timeout_sec,
    )


def _truthy(value: str | None) -> bool:
    return bool(value and value.strip().lower() in {"1", "true", "yes", "on"})


def _remote_agent_transports() -> set[str]:
    raw = (
        os.environ.get("CLAWCROSS_REMOTE_AGENT_TRANSPORT")
        or os.environ.get("CLAWCROSS_REMOTE_CODING_AGENT_TRANSPORT")
        or "acpx,claude"
    )
    values = {part.strip().lower() for part in raw.replace(";", ",").split(",") if part.strip()}
    if "legacy" in values:
        values.add("claude")
    if "all" in values:
        values.update({"acpx", "claude"})
    return values or {"acpx", "claude"}


def _remote_acpx_tools() -> list[str]:
    raw = os.environ.get("CLAWCROSS_REMOTE_ACPX_TOOLS") or DEFAULT_REMOTE_ACPX_TOOLS
    tools = []
    aliases = {"claude-code": "claude", "gemini-cli": "gemini"}
    for part in raw.replace(";", ",").split(","):
        tool = aliases.get(part.strip().lower(), part.strip().lower())
        if tool and tool not in tools:
            tools.append(tool)
    return tools or ["codex", "claude"]


def _remote_acpx_cwd() -> str:
    return os.environ.get("CLAWCROSS_REMOTE_ACPX_CWD", "~/.clawcross/acpx")


def _remote_acpx_env_path() -> str:
    return os.environ.get("CLAWCROSS_REMOTE_ACPX_ENV", "~/.clawcross/remote_acpx.env")


def _split_targets(value: str) -> list[str]:
    cleaned = value.replace("\n", ",").replace(";", ",")
    return [part.strip() for part in cleaned.split(",") if part.strip()]


def _parse_user_map(value: str | None) -> dict[str, str]:
    raw = (value or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = None
    if isinstance(parsed, dict):
        return {str(k).strip().lower(): str(v).strip() for k, v in parsed.items() if str(k).strip() and str(v).strip()}

    users: dict[str, str] = {}
    for part in _split_targets(raw):
        if "=" in part:
            host, user = part.split("=", 1)
        elif ":" in part:
            host, user = part.split(":", 1)
        else:
            continue
        host = host.strip().lower()
        user = user.strip()
        if host and user:
            users[host] = user
    return users


def _target_registry_path() -> str:
    return str(DATA_DIR / TARGETS_FILENAME)


def _load_registered_targets() -> list[dict[str, Any]]:
    paths = [
        os.environ.get("CLAWCROSS_REMOTE_CLAUDE_TARGETS_FILE", ""),
        _target_registry_path(),
    ]
    for raw_path in paths:
        path = str(raw_path or "").strip()
        if not path:
            continue
        try:
            with open(os.path.expanduser(path), "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception:
            continue
        if isinstance(payload, list):
            return [dict(item) for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            targets = payload.get("targets")
            if isinstance(targets, list):
                return [dict(item) for item in targets if isinstance(item, dict)]
    return []


def _registered_user_map() -> dict[str, str]:
    users = _parse_user_map(
        os.environ.get("CLAWCROSS_REMOTE_CLAUDE_USER_BY_HOST")
        or os.environ.get("REMOTE_CLAUDE_USER_BY_HOST")
    )
    for item in _load_registered_targets():
        user = str(item.get("user") or "").strip()
        if not user:
            target = str(item.get("target") or item.get("remote") or "").strip()
            if "@" in target:
                user = target.split("@", 1)[0].strip()
        if not user:
            continue
        for key in ("host", "ip", "hostname", "name", "dns_name"):
            value = str(item.get(key) or "").strip().lower().rstrip(".")
            if value:
                users[value] = user
        target = str(item.get("target") or item.get("remote") or "").strip()
        if "@" in target:
            host = target.rsplit("@", 1)[-1].strip().lower().rstrip(".")
            if host:
                users[host] = user
    return users


def _parse_remote_target(raw: str, base: RemoteClaudeConfig) -> RemoteClaudeConfig | None:
    target = str(raw or "").strip()
    if not target:
        return None
    if "@" in target:
        user, host = target.rsplit("@", 1)
        user = user.strip() or base.user
        host = host.strip()
    else:
        user = base.user
        host = target
    if not host or not user:
        return None
    return RemoteClaudeConfig(
        host=host,
        user=user,
        ssh_binary=base.ssh_binary,
        timeout_sec=base.timeout_sec,
        connect_timeout_sec=base.connect_timeout_sec,
    )


def _first_ipv4(values: Iterable[Any]) -> str:
    for value in values or []:
        text = str(value or "").strip()
        if "." in text and ":" not in text:
            return text
    return ""


def _infer_user_from_hostname(hostname: str) -> str:
    name = str(hostname or "").strip().lower().split(".", 1)[0]
    if not name:
        return ""
    for suffix in ("-pc", "-b850m-c", "-desktop", "-workstation", "-laptop"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name.split("-", 1)[0]


def _run_tailscale_status() -> dict[str, Any]:
    binary = (
        os.environ.get("CLAWCROSS_TAILSCALE_BINARY")
        or os.environ.get("TAILSCALE_BINARY")
        or "tailscale"
    ).strip()
    timeout_raw = os.environ.get("CLAWCROSS_REMOTE_CLAUDE_DISCOVERY_TIMEOUT_SEC") or "4"
    try:
        timeout = max(1.0, float(timeout_raw))
    except ValueError:
        timeout = 4.0
    proc = subprocess.run(
        shlex.split(binary) + ["status", "--json"],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "").strip() or "tailscale status failed")
    return json.loads(proc.stdout or "{}")


def discover_tailscale_remote_configs(base: RemoteClaudeConfig | None = None) -> list[RemoteClaudeConfig]:
    """Discover online Tailscale Linux peers and map them to SSH users."""

    if _truthy(os.environ.get("CLAWCROSS_REMOTE_CLAUDE_DISABLE_TAILSCALE")):
        return []
    base = base or load_remote_claude_config()
    try:
        status = _run_tailscale_status()
    except Exception:
        return []

    users = _registered_user_map()
    include_offline = _truthy(os.environ.get("CLAWCROSS_REMOTE_CLAUDE_INCLUDE_OFFLINE"))
    os_filter_raw = (os.environ.get("CLAWCROSS_REMOTE_CLAUDE_OS_FILTER") or "linux").strip().lower()
    os_filter = {part.strip() for part in os_filter_raw.split(",") if part.strip()}
    self_ips = set(status.get("Self", {}).get("TailscaleIPs") or [])
    configs: list[RemoteClaudeConfig] = []
    for peer in (status.get("Peer") or {}).values():
        if not isinstance(peer, dict):
            continue
        if not include_offline and peer.get("Online") is False:
            continue
        peer_os = str(peer.get("OS") or "").strip().lower()
        if os_filter and peer_os not in os_filter:
            continue
        host = _first_ipv4(peer.get("TailscaleIPs") or peer.get("AllowedIPs") or [])
        if not host or host in self_ips:
            continue
        hostname = str(peer.get("HostName") or "").strip()
        dns_name = str(peer.get("DNSName") or "").strip().rstrip(".")
        lookup_keys = [
            host.lower(),
            hostname.lower(),
            dns_name.lower(),
            dns_name.split(".", 1)[0].lower() if dns_name else "",
        ]
        user = ""
        for key in lookup_keys:
            if key and users.get(key):
                user = users[key]
                break
        if not user:
            user = _infer_user_from_hostname(hostname or dns_name)
        if not user:
            continue
        configs.append(
            RemoteClaudeConfig(
                host=host,
                user=user,
                ssh_binary=base.ssh_binary,
                timeout_sec=base.timeout_sec,
                connect_timeout_sec=base.connect_timeout_sec,
                hostname=hostname or dns_name,
            )
        )
    configs.sort(key=lambda item: (item.hostname or item.host).lower())
    return configs


def load_remote_claude_configs() -> list[RemoteClaudeConfig]:
    """Load remote Claude SSH targets.

    Tailscale discovery is the default so newly joined remote computers appear
    automatically. Explicit target env vars and the local registry only provide
    SSH usernames and fixed extra targets.
    """

    base = load_remote_claude_config()
    configs: list[RemoteClaudeConfig] = []
    if not _truthy(os.environ.get("CLAWCROSS_REMOTE_CLAUDE_DISABLE_TAILSCALE")):
        configs.extend(discover_tailscale_remote_configs(base))

    explicit_targets = (
        os.environ.get("CLAWCROSS_REMOTE_CLAUDE_TARGETS")
        or os.environ.get("REMOTE_CLAUDE_TARGETS")
        or ""
    )
    for raw in _split_targets(explicit_targets):
        config = _parse_remote_target(raw, base)
        if config:
            configs.append(config)

    # Targets registered via configure_remote_claude_dashboard.py
    # (~/.clawcross/data/remote_claude_targets.json) — JSON is the persistent
    # source of truth; the env var above stays as an ad-hoc override.
    for item in _load_registered_targets():
        raw = str(item.get("target") or item.get("remote") or "").strip()
        if not raw and item.get("user") and item.get("host"):
            raw = f"{item['user']}@{item['host']}"
        if not raw:
            continue
        config = _parse_remote_target(raw, base)
        if config:
            configs.append(config)

    if not configs:
        configs.append(base)

    deduped: list[RemoteClaudeConfig] = []
    seen: set[tuple[str, str]] = set()
    for config in configs:
        key = (config.user, config.host)
        if key in seen or not config.enabled:
            continue
        seen.add(key)
        deduped.append(config)
    return deduped


def _remote_payload(config: RemoteClaudeConfig, **extra: Any) -> dict[str, Any]:
    payload = {
        "host": config.host,
        "user": config.user,
        "hostname": config.hostname,
        "target": config.destination,
    }
    payload.update({k: v for k, v in extra.items() if v not in (None, "")})
    return payload


def _remote_script_list_sessions(tail_lines: int) -> str:
    return f"""
import glob, json, os, time
from collections import deque

tail_lines = {int(tail_lines)}
base = os.path.expanduser("~/.claude/sessions")
items = []

def candidate_transcript(data):
    explicit = (
        data.get("transcript")
        or data.get("transcript_path")
        or data.get("transcriptPath")
        or data.get("transcript_file")
        or data.get("transcriptFile")
        or ""
    )
    if explicit:
        return explicit if os.path.isabs(explicit) else os.path.expanduser(explicit)
    session_id = data.get("sessionId") or data.get("session_id") or data.get("localSessionId") or ""
    job_id = data.get("jobId") or data.get("job_id") or ""
    cwd = data.get("cwd") or data.get("workingDirectory") or ""
    candidates = []
    if cwd:
        slug = cwd.rstrip("/").replace("/", "-").replace(".", "-")
        if session_id:
            candidates.append(os.path.expanduser(f"~/.claude/projects/{{slug}}/{{session_id}}.jsonl"))
        if job_id:
            candidates.extend(glob.glob(os.path.expanduser(f"~/.claude/projects/{{slug}}/{{job_id}}-*.jsonl")))
    if job_id:
        candidates.append(os.path.expanduser(f"~/.claude/jobs/{{job_id}}/timeline.jsonl"))
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return ""

for path in sorted(glob.glob(os.path.join(base, "*.json"))):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        continue
    transcript = candidate_transcript(data)
    last_lines = []
    if transcript and os.path.exists(transcript):
        try:
            with open(transcript, "r", encoding="utf-8", errors="replace") as tfh:
                last_lines = list(deque(tfh, maxlen=tail_lines))
        except Exception:
            last_lines = []
    stat = None
    try:
        stat = os.stat(path)
    except Exception:
        pass
    item = {{
        "id": os.path.splitext(os.path.basename(path))[0],
        "title": data.get("title") or data.get("name") or data.get("task") or "",
        "status": data.get("status") or data.get("state") or "",
        "session_id": data.get("sessionId") or data.get("session_id") or data.get("localSessionId") or "",
        "bridge_session_id": data.get("bridgeSessionId") or data.get("bridge_session_id") or data.get("remoteSessionId") or "",
        "job_id": data.get("jobId") or data.get("job_id") or "",
        "pid": data.get("pid") or "",
        "kind": data.get("kind") or "",
        "cwd": data.get("cwd") or data.get("workingDirectory") or "",
        "transcript_path": transcript,
        "updated_at": data.get("updatedAt") or data.get("updated_at") or (stat.st_mtime if stat else None),
        "session_file": path,
        "tail_lines": last_lines,
    }}
    items.append(item)
print(json.dumps({{"sessions": items}}, ensure_ascii=False))
"""


def _remote_acpx_script_prelude() -> str:
    script = r"""
import json, os, shutil, subprocess, tempfile

TOOLS = __TOOLS__
ACPX_CWD = os.path.expanduser(__ACPX_CWD__)
ENV_PATH = os.path.expanduser(__ENV_PATH__)
TTL_SEC = __TTL_SEC__

def _load_env_file(path):
    env = os.environ.copy()
    home = os.path.expanduser("~")
    env["PATH"] = os.path.join(home, ".local", "bin") + os.pathsep + env.get("PATH", "")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("'").strip('"')
                if key:
                    env[key] = value
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return env

ENV = _load_env_file(ENV_PATH)
ACPX = shutil.which("acpx", path=ENV.get("PATH"))

def _json_from_stdout(text):
    body = (text or "").strip()
    if not body:
        return None
    try:
        return json.loads(body)
    except Exception:
        pass
    for line in body.splitlines():
        line = line.strip()
        if not line or line[0] not in "[{":
            continue
        try:
            return json.loads(line)
        except Exception:
            continue
    return None

def _base_cmd():
    return [ACPX, "--cwd", ACPX_CWD, "--ttl", str(TTL_SEC), "--format", "json", "--json-strict"]

def _response_base(**extra):
    payload = {
        "transport": "acpx",
        "available": bool(ACPX),
        "cwd": ACPX_CWD,
        "env_path": ENV_PATH,
        "tools": TOOLS,
    }
    payload.update(extra)
    return payload

def _print(payload):
    print(json.dumps(payload, ensure_ascii=False))
"""
    return (
        script.replace("__TOOLS__", json.dumps(_remote_acpx_tools()))
        .replace("__ACPX_CWD__", json.dumps(_remote_acpx_cwd()))
        .replace("__ENV_PATH__", json.dumps(_remote_acpx_env_path()))
        .replace("__TTL_SEC__", str(int(os.environ.get("CLAWCROSS_REMOTE_ACPX_TTL_SEC") or "300")))
    )


def _remote_script_list_acpx_sessions() -> str:
    return _remote_acpx_script_prelude() + r"""
sessions = []
errors = []
if not ACPX:
    _print(_response_base(ok=False, sessions=sessions, errors=["acpx missing on PATH"]))
    raise SystemExit(0)

os.makedirs(ACPX_CWD, exist_ok=True)
for tool in TOOLS:
    cmd = _base_cmd() + [tool, "sessions", "list"]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=25, env=ENV)
    except Exception as exc:
        errors.append(f"{tool}: {exc}")
        continue
    if proc.returncode != 0:
        errors.append(f"{tool}: {(proc.stderr or proc.stdout or '').strip() or 'sessions list failed'}")
        continue
    data = _json_from_stdout(proc.stdout)
    if isinstance(data, dict) and isinstance(data.get("sessions"), list):
        items = data["sessions"]
    elif isinstance(data, list):
        items = data
    else:
        items = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("sessionId") or item.get("session_id") or item.get("id") or "").strip()
        if not name:
            continue
        sessions.append({
            "tool": tool,
            "name": name,
            "sessionId": item.get("sessionId") or item.get("session_id") or item.get("id"),
            "acpxRecordId": item.get("acpxRecordId"),
            "closed": bool(item.get("closed")),
            "lastUsedAt": item.get("lastUsedAt"),
            "updatedAt": item.get("updatedAt") or item.get("updated_at"),
            "cwd": item.get("cwd"),
            "title": item.get("title"),
            "message_count": len(item.get("messages") or []) if isinstance(item.get("messages"), list) else 0,
        })
_print(_response_base(ok=True, sessions=sessions, errors=errors))
"""


def _remote_script_read_acpx_session(tool: str, name: str, line_limit: int) -> str:
    tool_json = json.dumps(str(tool))
    name_json = json.dumps(str(name))
    return _remote_acpx_script_prelude() + f"""
tool = {tool_json}
name = {name_json}
line_limit = {int(line_limit)}
if not ACPX:
    _print(_response_base(ok=False, found=False, error="acpx missing on PATH"))
    raise SystemExit(0)

cmd = _base_cmd() + [tool, "sessions", "read", name]
if line_limit > 0:
    cmd.extend(["--tail", str(line_limit)])
try:
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=30, env=ENV)
except Exception as exc:
    _print(_response_base(ok=False, found=False, error=str(exc)))
    raise SystemExit(0)

data = _json_from_stdout(proc.stdout)
ok = proc.returncode == 0
_print(_response_base(
    ok=ok,
    found=ok,
    tool=tool,
    name=name,
    stdout=proc.stdout,
    stderr=proc.stderr,
    data=data,
    error="" if ok else ((proc.stderr or proc.stdout or "").strip() or "sessions read failed"),
))
"""


def _remote_script_send_acpx_message(tool: str, name: str, text: str) -> str:
    tool_json = json.dumps(str(tool))
    name_json = json.dumps(str(name))
    text_json = json.dumps(str(text))
    return _remote_acpx_script_prelude() + f"""
tool = {tool_json}
name = {name_json}
text = {text_json}
if not ACPX:
    _print(_response_base(ok=False, found=False, response={{"ok": False, "error": "acpx missing on PATH", "code": "ENOACPX"}}))
    raise SystemExit(0)

os.makedirs(ACPX_CWD, exist_ok=True)
ensure_cmd = _base_cmd() + [tool, "sessions", "ensure", "--name", name]
subprocess.run(ensure_cmd, check=False, capture_output=True, text=True, timeout=20, env=ENV)

payload = [{{"type": "text", "text": text.strip() or "(empty prompt)"}}]
tmp_path = ""
try:
    fd, tmp_path = tempfile.mkstemp(prefix="clawcross_acpx_prompt_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    cmd = [ACPX, "--cwd", ACPX_CWD, "--ttl", str(TTL_SEC), "--approve-all", "--format", "json", "--json-strict", tool, "prompt", "-s", name, "--file", tmp_path]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=600, env=ENV)
finally:
    if tmp_path:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

ok = proc.returncode == 0
_print(_response_base(
    ok=ok,
    found=True,
    tool=tool,
    name=name,
    stdout=proc.stdout,
    stderr=proc.stderr,
    response={{"ok": ok, "via": "acpx", "returncode": proc.returncode, "error": "" if ok else ((proc.stderr or proc.stdout or "").strip() or "prompt failed")}},
))
"""


def _remote_script_close_acpx_session(tool: str, name: str) -> str:
    tool_json = json.dumps(str(tool))
    name_json = json.dumps(str(name))
    return _remote_acpx_script_prelude() + f"""
tool = {tool_json}
name = {name_json}
if not ACPX:
    _print(_response_base(ok=False, found=False, error="acpx missing on PATH"))
    raise SystemExit(0)

cmd = _base_cmd() + [tool, "sessions", "close", name]
try:
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=30, env=ENV)
except Exception as exc:
    _print(_response_base(ok=False, found=False, error=str(exc)))
    raise SystemExit(0)
ok = proc.returncode == 0
_print(_response_base(
    ok=ok,
    found=True,
    tool=tool,
    name=name,
    stdout=proc.stdout,
    stderr=proc.stderr,
    error="" if ok else ((proc.stderr or proc.stdout or "").strip() or "sessions close failed"),
))
"""


def _remote_script_read_messages(target: str, line_limit: int) -> str:
    target_json = json.dumps(str(target))
    return f"""
import glob, json, os
from collections import deque

target = {target_json}
line_limit = {int(line_limit)}
base = os.path.expanduser("~/.claude/sessions")
matched = None

def candidate_transcript(data):
    explicit = (
        data.get("transcript")
        or data.get("transcript_path")
        or data.get("transcriptPath")
        or data.get("transcript_file")
        or data.get("transcriptFile")
        or ""
    )
    if explicit:
        return explicit if os.path.isabs(explicit) else os.path.expanduser(explicit)
    session_id = data.get("sessionId") or data.get("session_id") or data.get("localSessionId") or ""
    job_id = data.get("jobId") or data.get("job_id") or ""
    cwd = data.get("cwd") or data.get("workingDirectory") or ""
    candidates = []
    if cwd:
        slug = cwd.rstrip("/").replace("/", "-").replace(".", "-")
        if session_id:
            candidates.append(os.path.expanduser(f"~/.claude/projects/{{slug}}/{{session_id}}.jsonl"))
        if job_id:
            candidates.extend(glob.glob(os.path.expanduser(f"~/.claude/projects/{{slug}}/{{job_id}}-*.jsonl")))
    if job_id:
        candidates.append(os.path.expanduser(f"~/.claude/jobs/{{job_id}}/timeline.jsonl"))
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return ""

for path in sorted(glob.glob(os.path.join(base, "*.json"))):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        continue
    keys = [
        os.path.splitext(os.path.basename(path))[0],
        data.get("sessionId") or data.get("session_id") or data.get("localSessionId") or "",
        data.get("bridgeSessionId") or data.get("bridge_session_id") or data.get("remoteSessionId") or "",
        data.get("jobId") or data.get("job_id") or "",
    ]
    if target not in [str(k) for k in keys if k]:
        continue
    transcript = candidate_transcript(data)
    matched = {{
        "id": os.path.splitext(os.path.basename(path))[0],
        "title": data.get("title") or data.get("name") or data.get("task") or "",
        "status": data.get("status") or data.get("state") or "",
        "session_id": data.get("sessionId") or data.get("session_id") or data.get("localSessionId") or "",
        "bridge_session_id": data.get("bridgeSessionId") or data.get("bridge_session_id") or data.get("remoteSessionId") or "",
        "job_id": data.get("jobId") or data.get("job_id") or "",
        "cwd": data.get("cwd") or data.get("workingDirectory") or "",
        "transcript_path": transcript,
        "session_file": path,
    }}
    break
if not matched:
    print(json.dumps({{"found": False, "error": "session not found"}}, ensure_ascii=False))
    raise SystemExit(0)
lines = []
path = matched.get("transcript_path") or ""
if path and os.path.exists(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = list(deque(fh, maxlen=line_limit))
print(json.dumps({{"found": True, "session": matched, "lines": lines}}, ensure_ascii=False))
"""


def _remote_script_send_message(target: str, text: str) -> str:
    target_json = json.dumps(str(target))
    text_json = json.dumps(str(text))
    return f"""
import glob, json, os, socket, subprocess

target = {target_json}
text = {text_json}
base = os.path.expanduser("~/.claude/sessions")
matched = None

for path in sorted(glob.glob(os.path.join(base, "*.json"))):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        continue
    keys = [
        os.path.splitext(os.path.basename(path))[0],
        data.get("sessionId") or data.get("session_id") or data.get("localSessionId") or "",
        data.get("bridgeSessionId") or data.get("bridge_session_id") or data.get("remoteSessionId") or "",
        data.get("jobId") or data.get("job_id") or "",
    ]
    if target not in [str(k) for k in keys if k]:
        continue
    matched = {{
        "id": os.path.splitext(os.path.basename(path))[0],
        "title": data.get("title") or data.get("name") or data.get("task") or "",
        "status": data.get("status") or data.get("state") or "",
        "session_id": data.get("sessionId") or data.get("session_id") or data.get("localSessionId") or "",
        "bridge_session_id": data.get("bridgeSessionId") or data.get("bridge_session_id") or data.get("remoteSessionId") or "",
        "job_id": data.get("jobId") or data.get("job_id") or "",
        "pid": data.get("pid") or "",
        "kind": data.get("kind") or "",
        "cwd": data.get("cwd") or data.get("workingDirectory") or "",
        "session_file": path,
    }}
    break

if not matched:
    print(json.dumps({{"found": False, "error": "session not found"}}, ensure_ascii=False))
    raise SystemExit(0)

short = str(matched.get("job_id") or "").strip()
if not short:
    pane_target = ""
    pane_error = ""
    pid = str(matched.get("pid") or matched.get("id") or "").strip()
    if pid:
        try:
            proc = subprocess.run(
                ["tmux", "list-panes", "-a", "-F", "#{{session_name}}\\t#{{pane_id}}\\t#{{pane_pid}}"],
                check=False,
                capture_output=True,
                text=True,
                timeout=4,
            )
            if proc.returncode == 0:
                for raw in (proc.stdout or "").splitlines():
                    parts = raw.split("\\t")
                    if len(parts) >= 3 and parts[2].strip() == pid:
                        pane_target = parts[1].strip()
                        break
            else:
                pane_error = (proc.stderr or proc.stdout or "").strip()
        except Exception as exc:
            pane_error = str(exc)
    if pane_target:
        try:
            load = subprocess.run(["tmux", "load-buffer", "-"], input=text, text=True, capture_output=True, timeout=4)
            if load.returncode != 0:
                raise RuntimeError((load.stderr or load.stdout or "tmux load-buffer failed").strip())
            paste = subprocess.run(["tmux", "paste-buffer", "-t", pane_target], check=False, capture_output=True, text=True, timeout=4)
            if paste.returncode != 0:
                raise RuntimeError((paste.stderr or paste.stdout or "tmux paste-buffer failed").strip())
            enter = subprocess.run(["tmux", "send-keys", "-t", pane_target, "C-m"], check=False, capture_output=True, text=True, timeout=4)
            if enter.returncode != 0:
                raise RuntimeError((enter.stderr or enter.stdout or "tmux send-keys failed").strip())
            print(json.dumps({{"found": True, "session": matched, "response": {{"ok": True, "via": "tmux", "pane": pane_target}}}}, ensure_ascii=False))
            raise SystemExit(0)
        except Exception as exc:
            pane_error = str(exc)
    print(json.dumps({{"found": True, "session": matched, "response": {{"ok": False, "error": "session has no daemon job id" + (": " + pane_error if pane_error else ""), "code": "ENOJOB"}}}}, ensure_ascii=False))
    raise SystemExit(0)

try:
    with open(os.path.expanduser("~/.claude/daemon/roster.json"), "r", encoding="utf-8") as fh:
        roster = json.load(fh)
except Exception as exc:
    print(json.dumps({{"found": True, "session": matched, "response": {{"ok": False, "error": "daemon roster unavailable: " + str(exc), "code": "ENOCONN"}}}}, ensure_ascii=False))
    raise SystemExit(0)

worker = (roster.get("workers") or {{}}).get(short)
if not worker:
    print(json.dumps({{"found": True, "session": matched, "short": short, "response": {{"ok": False, "error": "job not found - it may have already exited", "code": "ENOJOB"}}}}, ensure_ascii=False))
    raise SystemExit(0)

sock_ref = worker.get("rendezvousSock") or worker.get("ptySock") or ""
if not sock_ref:
    print(json.dumps({{"found": True, "session": matched, "short": short, "response": {{"ok": False, "error": "worker socket unavailable", "code": "ENOCONN"}}}}, ensure_ascii=False))
    raise SystemExit(0)
control_sock = os.path.join(os.path.dirname(os.path.dirname(sock_ref)), "control.sock")

payload = {{"proto": int(roster.get("proto") or 1), "op": "reply", "short": short, "text": text}}
try:
    client = socket.socket(socket.AF_UNIX)
    client.settimeout(4.0)
    client.connect(control_sock)
    client.sendall((json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\\n").encode("utf-8"))
    buf = b""
    while b"\\n" not in buf and len(buf) < 1024 * 1024:
        chunk = client.recv(4096)
        if not chunk:
            break
        buf += chunk
    client.close()
    line = buf.split(b"\\n", 1)[0].decode("utf-8", errors="replace").strip()
    response = json.loads(line) if line else {{"ok": False, "error": "empty daemon response", "code": "ENOCONN"}}
except Exception as exc:
    response = {{"ok": False, "error": str(exc), "code": "ENOCONN"}}

print(json.dumps({{"found": True, "session": matched, "short": short, "response": response}}, ensure_ascii=False))
"""


def _remote_script_rename_session(target: str, name: str) -> str:
    target_json = json.dumps(str(target))
    name_json = json.dumps(str(name))
    return f"""
import glob, json, os, time

target = {target_json}
name = {name_json}
base = os.path.expanduser("~/.claude/sessions")
matched = None
session_path = ""

for path in sorted(glob.glob(os.path.join(base, "*.json"))):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        continue
    keys = [
        os.path.splitext(os.path.basename(path))[0],
        data.get("sessionId") or data.get("session_id") or data.get("localSessionId") or "",
        data.get("bridgeSessionId") or data.get("bridge_session_id") or data.get("remoteSessionId") or "",
        data.get("jobId") or data.get("job_id") or "",
    ]
    if target not in [str(k) for k in keys if k]:
        continue
    matched = dict(data)
    matched["id"] = os.path.splitext(os.path.basename(path))[0]
    session_path = path
    break

if not matched:
    print(json.dumps({{"found": False, "ok": False, "error": "session not found"}}, ensure_ascii=False))
    raise SystemExit(0)

old_name = matched.get("name") or matched.get("title") or ""
now_ms = int(time.time() * 1000)
matched["name"] = name
matched["title"] = name
matched["updatedAt"] = max(int(matched.get("updatedAt") or 0), now_ms)
tmp_path = session_path + ".tmp"
with open(tmp_path, "w", encoding="utf-8") as fh:
    json.dump(matched, fh, ensure_ascii=False, indent=2)
    fh.write("\\n")
os.replace(tmp_path, session_path)

short = str(matched.get("jobId") or matched.get("job_id") or "").strip()
roster_updated = False
roster_error = ""
if short:
    roster_path = os.path.expanduser("~/.claude/daemon/roster.json")
    try:
        with open(roster_path, "r", encoding="utf-8") as fh:
            roster = json.load(fh)
        worker = (roster.get("workers") or {{}}).get(short)
        if isinstance(worker, dict):
            dispatch = worker.setdefault("dispatch", {{}})
            seed = dispatch.setdefault("seed", {{}})
            if isinstance(seed, dict):
                seed["name"] = name
            worker["name"] = name
            roster_tmp = roster_path + ".tmp"
            with open(roster_tmp, "w", encoding="utf-8") as fh:
                json.dump(roster, fh, ensure_ascii=False, indent=2)
                fh.write("\\n")
            os.replace(roster_tmp, roster_path)
            roster_updated = True
    except Exception as exc:
        roster_error = str(exc)

session = {{
    "id": matched.get("id") or os.path.splitext(os.path.basename(session_path))[0],
    "title": matched.get("title") or matched.get("name") or "",
    "status": matched.get("status") or matched.get("state") or "",
    "session_id": matched.get("sessionId") or matched.get("session_id") or matched.get("localSessionId") or "",
    "bridge_session_id": matched.get("bridgeSessionId") or matched.get("bridge_session_id") or matched.get("remoteSessionId") or "",
    "job_id": matched.get("jobId") or matched.get("job_id") or "",
    "cwd": matched.get("cwd") or matched.get("workingDirectory") or "",
    "session_file": session_path,
}}
print(json.dumps({{
    "found": True,
    "ok": True,
    "old_name": old_name,
    "name": name,
    "session": session,
    "short": short,
    "roster_updated": roster_updated,
    "roster_error": roster_error,
}}, ensure_ascii=False))
"""


def _remote_script_close_session(target: str, *, force: bool) -> str:
    target_json = json.dumps(str(target))
    force_json = "True" if force else "False"
    return f"""
import glob, json, os, signal, shutil, socket, time

target = {target_json}
force = {force_json}
base = os.path.expanduser("~/.claude/sessions")
matched = None
session_path = ""

for path in sorted(glob.glob(os.path.join(base, "*.json"))):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        continue
    keys = [
        os.path.splitext(os.path.basename(path))[0],
        data.get("sessionId") or data.get("session_id") or data.get("localSessionId") or "",
        data.get("bridgeSessionId") or data.get("bridge_session_id") or data.get("remoteSessionId") or "",
        data.get("jobId") or data.get("job_id") or "",
    ]
    if target not in [str(k) for k in keys if k]:
        continue
    matched = {{
        "id": os.path.splitext(os.path.basename(path))[0],
        "title": data.get("title") or data.get("name") or data.get("task") or "",
        "status": data.get("status") or data.get("state") or "",
        "session_id": data.get("sessionId") or data.get("session_id") or data.get("localSessionId") or "",
        "bridge_session_id": data.get("bridgeSessionId") or data.get("bridge_session_id") or data.get("remoteSessionId") or "",
        "job_id": data.get("jobId") or data.get("job_id") or "",
        "cwd": data.get("cwd") or data.get("workingDirectory") or "",
        "session_file": path,
    }}
    session_path = path
    break

if not matched:
    print(json.dumps({{"found": False, "ok": False, "error": "session not found"}}, ensure_ascii=False))
    raise SystemExit(0)

short = str(matched.get("job_id") or "").strip()
pid = None
kill_result = {{"attempted": False, "terminated": False, "error": ""}}
daemon_kill = {{"attempted": False, "ok": False, "error": "", "response": {{}}}}
control_sock = ""
if short:
    try:
        with open(os.path.expanduser("~/.claude/daemon/roster.json"), "r", encoding="utf-8") as fh:
            roster = json.load(fh)
        worker = (roster.get("workers") or {{}}).get(short) or {{}}
        raw_pid = worker.get("pid")
        pid = int(raw_pid) if raw_pid else None
        sock_ref = worker.get("rendezvousSock") or worker.get("ptySock") or ""
        if sock_ref:
            control_sock = os.path.join(os.path.dirname(os.path.dirname(sock_ref)), "control.sock")
    except Exception as exc:
        kill_result["error"] = "roster unavailable: " + str(exc)

if short and control_sock:
    daemon_kill["attempted"] = True
    try:
        payload = {{"proto": int(roster.get("proto") or 1), "op": "kill", "short": short}}
        client = socket.socket(socket.AF_UNIX)
        client.settimeout(4.0)
        client.connect(control_sock)
        client.sendall((json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\\n").encode("utf-8"))
        buf = b""
        while b"\\n" not in buf and len(buf) < 1024 * 1024:
            chunk = client.recv(4096)
            if not chunk:
                break
            buf += chunk
        client.close()
        line = buf.split(b"\\n", 1)[0].decode("utf-8", errors="replace").strip()
        response = json.loads(line) if line else {{"ok": False, "error": "empty daemon response"}}
        daemon_kill["response"] = response
        daemon_kill["ok"] = bool(response.get("ok"))
    except Exception as exc:
        daemon_kill["error"] = str(exc)
    time.sleep(0.5)

if pid:
    kill_result["attempted"] = True
    try:
        try:
            os.kill(pid, 0)
            alive = True
        except ProcessLookupError:
            alive = False
        if alive and not daemon_kill.get("ok"):
            os.kill(pid, signal.SIGTERM)
            time.sleep(1.0)
            try:
                os.kill(pid, 0)
                alive = True
            except ProcessLookupError:
                alive = False
        if alive and force:
            os.kill(pid, signal.SIGKILL)
            time.sleep(0.2)
            try:
                os.kill(pid, 0)
                alive = True
            except ProcessLookupError:
                alive = False
        kill_result["terminated"] = not alive
    except ProcessLookupError:
        kill_result["terminated"] = True
    except Exception as exc:
        kill_result["error"] = str(exc)

archive_path = ""
archive_result = {{"attempted": False, "archived": False, "error": ""}}
if session_path and os.path.exists(session_path):
    archive_result["attempted"] = True
    archive_dir = os.path.join(base, ".clawcross-archive")
    try:
        os.makedirs(archive_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        archive_path = os.path.join(archive_dir, os.path.basename(session_path) + "." + stamp + ".json")
        counter = 1
        while os.path.exists(archive_path):
            archive_path = os.path.join(archive_dir, os.path.basename(session_path) + "." + stamp + f".{{counter}}.json")
            counter += 1
        shutil.move(session_path, archive_path)
        archive_result["archived"] = True
    except Exception as exc:
        archive_result["error"] = str(exc)

session_file_gone = bool(session_path) and not os.path.exists(session_path)
ok = (bool(archive_result.get("archived")) or session_file_gone) and (not pid or kill_result.get("terminated") or not force)
print(json.dumps({{
    "found": True,
    "ok": ok,
    "session": matched,
    "short": short,
    "pid": pid,
    "archive_path": archive_path,
    "session_file_gone": session_file_gone,
    "daemon_kill": daemon_kill,
    "kill": kill_result,
    "archive": archive_result,
}}, ensure_ascii=False))
"""


def _ssh_prefix(config: RemoteClaudeConfig) -> list[str]:
    prefix = shlex.split(config.ssh_binary) if config.ssh_binary else ["ssh"]
    if os.path.basename(prefix[0]) == "ssh":
        prefix.extend(
            [
                "-o",
                "BatchMode=yes",
                "-o",
                f"ConnectTimeout={config.connect_timeout_sec}",
                "-o",
                "ServerAliveInterval=3",
                "-o",
                "ServerAliveCountMax=1",
                "-o",
                "StrictHostKeyChecking=accept-new",
            ]
        )
    return prefix


def _run_remote_python(script: str, *, config: RemoteClaudeConfig) -> dict[str, Any]:
    if not config.enabled:
        raise RuntimeError("remote Claude host/user is not configured")
    cmd = _ssh_prefix(config) + [config.destination, "python3 -c " + shlex.quote(script)]
    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=config.timeout_sec,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(err or f"remote ssh command failed with code {proc.returncode}")
    out = (proc.stdout or "").strip()
    if not out:
        return {}
    return json.loads(out)


def _safe_json_loads(line: str) -> dict[str, Any] | None:
    try:
        value = json.loads(line)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _cache_path() -> str:
    return str(DATA_DIR / CACHE_FILENAME)


def _load_cached_sessions() -> list[dict[str, Any]]:
    path = _cache_path()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception:
        return []
    sessions = payload.get("sessions") if isinstance(payload, dict) else None
    if not isinstance(sessions, list):
        return []
    return [dict(s) for s in sessions if isinstance(s, dict)]


def _write_cached_sessions(sessions: list[dict[str, Any]]) -> None:
    if not sessions:
        return
    path = _cache_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        safe_sessions = []
        for session in sessions:
            safe = {k: v for k, v in session.items() if k not in {"session_file", "transcript_path"}}
            safe_sessions.append(safe)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"sessions": safe_sessions}, fh, ensure_ascii=False, indent=2)
    except Exception:
        return


def _content_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or item.get("input") or ""
                if isinstance(text, list):
                    text = _content_to_text(text)
                if text:
                    parts.append(str(text))
                elif item.get("type"):
                    name = item.get("name") or item.get("tool_name") or item.get("id") or ""
                    label = f"[{item.get('type')}{':' + str(name) if name else ''}]"
                    parts.append(label)
        return "\n".join(p for p in parts if p)
    if isinstance(value, dict):
        for key in ("text", "content", "result", "summary"):
            if key in value:
                text = _content_to_text(value.get(key))
                if text:
                    return text
        if value.get("type"):
            return f"[{value.get('type')}]"
    return str(value)


def parse_claude_transcript_lines(lines: Iterable[str], *, limit: int = 80) -> list[dict[str, Any]]:
    """Parse Claude Code JSONL transcript lines into compact UI messages."""

    messages: list[dict[str, Any]] = []
    for idx, line in enumerate(lines):
        obj = _safe_json_loads(line)
        if not obj:
            continue
        raw_type = str(obj.get("type") or "").strip()
        msg = obj.get("message") if isinstance(obj.get("message"), dict) else {}
        role = str(msg.get("role") or obj.get("role") or raw_type or "event").strip()
        content = ""
        if msg:
            content = _content_to_text(msg.get("content"))
        if not content:
            content = _content_to_text(obj.get("content"))
        if not content and raw_type == "result":
            content = _content_to_text(obj.get("result") or obj.get("error") or obj.get("subtype"))
        if not content:
            continue
        messages.append(
            {
                "id": obj.get("uuid") or obj.get("id") or idx,
                "role": role or "event",
                "type": raw_type or role,
                "content": content,
                "timestamp": obj.get("timestamp") or obj.get("created_at") or obj.get("time") or "",
            }
        )
    return messages[-limit:]


def _summarize_last_message(lines: Iterable[str]) -> dict[str, Any] | None:
    parsed = parse_claude_transcript_lines(lines, limit=1)
    return parsed[-1] if parsed else None


def _normalize_session(item: dict[str, Any], *, config: RemoteClaudeConfig | None = None) -> dict[str, Any]:
    tail_lines = item.pop("tail_lines", []) or []
    last = _summarize_last_message(tail_lines)
    item["display_id"] = (
        item.get("bridge_session_id")
        or item.get("session_id")
        or item.get("id")
        or item.get("job_id")
        or ""
    )
    item["last_message"] = last
    item["message_count_hint"] = len(tail_lines)
    if config:
        item["remote_host"] = config.host
        item["remote_user"] = config.user
        item["remote_hostname"] = config.hostname
        item["remote"] = _remote_payload(config)
        if item.get("display_id"):
            item["remote_key"] = f"{config.destination}{REMOTE_KEY_SEPARATOR}{item['display_id']}"
    return item


def _normalize_acpx_session(item: dict[str, Any], *, config: RemoteClaudeConfig) -> dict[str, Any]:
    tool = str(item.get("tool") or "").strip().lower() or "codex"
    name = str(item.get("name") or item.get("sessionId") or item.get("session_id") or item.get("id") or "").strip()
    status = "closed" if item.get("closed") else "idle"
    session = {
        "id": name,
        "title": str(item.get("title") or name),
        "status": status,
        "session_id": name,
        "bridge_session_id": "",
        "job_id": "",
        "kind": "acpx",
        "cwd": item.get("cwd") or "",
        "updated_at": item.get("lastUsedAt") or item.get("updatedAt") or "",
        "transport": "acpx",
        "agent_tool": tool,
        "display_id": name,
        "last_message": None,
        "message_count_hint": int(item.get("message_count") or 0),
        "remote_host": config.host,
        "remote_user": config.user,
        "remote_hostname": config.hostname,
        "remote": _remote_payload(config),
        "remote_key": f"{config.destination}{REMOTE_KEY_SEPARATOR}{ACPX_SESSION_PREFIX}{tool}:{name}",
    }
    return session


def _parse_acpx_session_ref(session_key: str) -> tuple[str, str] | None:
    raw = str(session_key or "").strip()
    if not raw.startswith(ACPX_SESSION_PREFIX):
        return None
    rest = raw[len(ACPX_SESSION_PREFIX):]
    if ":" not in rest:
        return None
    tool, name = rest.split(":", 1)
    tool = tool.strip().lower()
    name = name.strip()
    if not tool or not name:
        return None
    return tool, name


def _is_acpx_smoke_session(item: dict[str, Any]) -> bool:
    title = str(item.get("title") or "").strip()
    if title == "Reply exactly: clawcross-acpx-ok":
        return True
    name = str(item.get("name") or item.get("sessionId") or item.get("session_id") or item.get("id") or "").strip()
    return name == "clawcross-smoke"


def _pick_acpx_text(obj: Any) -> str | None:
    if isinstance(obj, dict):
        if obj.get("jsonrpc") == "2.0":
            method = str(obj.get("method") or "")
            params = obj.get("params") if isinstance(obj.get("params"), dict) else {}
            if method.endswith("agent_message_chunk"):
                chunk = params.get("chunk") or params.get("text")
                if isinstance(chunk, str) and chunk.strip():
                    return chunk
        for key in ("text", "content", "message", "summary", "reply"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value
        result = obj.get("result")
        nested = _pick_acpx_text(result)
        if nested:
            return nested
    if isinstance(obj, list):
        parts = [_pick_acpx_text(item) for item in obj]
        text = "\n".join(part for part in parts if part)
        return text or None
    return None


def _extract_acpx_text(output: str) -> str:
    chunks: list[str] = []
    fallback = ""
    for raw in str(output or "").splitlines():
        line = raw.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        text = _pick_acpx_text(obj)
        if not text:
            continue
        if isinstance(obj, dict) and str(obj.get("method") or "").endswith("agent_message_chunk"):
            chunks.append(text)
        else:
            fallback = text
    return "".join(chunks).strip() or fallback.strip()


def _parse_acpx_read_messages(payload: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    data = payload.get("data")
    if not isinstance(data, dict):
        text = _extract_acpx_text(str(payload.get("stdout") or ""))
        return [{"id": "acpx-read", "role": "assistant", "type": "acpx", "content": text, "timestamp": ""}] if text else []

    messages_raw = data.get("messages")
    if not isinstance(messages_raw, list):
        text = _pick_acpx_text(data) or _extract_acpx_text(str(payload.get("stdout") or ""))
        return [{"id": "acpx-read", "role": "assistant", "type": "acpx", "content": text, "timestamp": ""}] if text else []

    messages: list[dict[str, Any]] = []
    for idx, item in enumerate(messages_raw):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or item.get("type") or "event").strip()
        content = _content_to_text(item.get("content"))
        if not content:
            content = _pick_acpx_text(item) or ""
        if not content:
            continue
        messages.append(
            {
                "id": item.get("id") or item.get("uuid") or idx,
                "role": role,
                "type": str(item.get("type") or role),
                "content": content,
                "timestamp": item.get("timestamp") or item.get("created_at") or item.get("time") or "",
            }
        )
    return messages[-limit:]


def _is_daemon_spare_session(item: dict[str, Any]) -> bool:
    """Hide Claude agents daemon spare workers that are not real conversations."""

    status = str(item.get("status") or "").strip().lower()
    if status != "idle":
        return False
    has_bridge = bool(str(item.get("bridge_session_id") or item.get("bridgeSessionId") or "").strip())
    has_transcript = bool(str(item.get("transcript_path") or item.get("transcript") or item.get("transcriptPath") or "").strip())
    has_tail = bool(item.get("tail_lines"))
    return not has_bridge and not has_transcript and not has_tail


def list_remote_claude_sessions(*, limit: int = 3, tail_lines: int = 30) -> dict[str, Any]:
    configs = load_remote_claude_configs()
    transports = _remote_agent_transports()
    sessions: list[dict[str, Any]] = []
    remotes: list[dict[str, Any]] = []
    errors: list[str] = []

    for config in configs:
        if "acpx" in transports:
            remote_record = _remote_payload(config, ok=False, session_count=0, transport="acpx")
            try:
                payload = _run_remote_python(_remote_script_list_acpx_sessions(), config=config)
                if payload.get("transport") != "acpx":
                    raise RuntimeError("remote ACPX probe did not return an ACPX payload")
                if not payload.get("available"):
                    raise RuntimeError("; ".join(payload.get("errors") or []) or "acpx missing on remote PATH")
                remote_sessions = [
                    _normalize_acpx_session(dict(item), config=config)
                    for item in payload.get("sessions", [])
                    if isinstance(item, dict)
                    and str(item.get("name") or item.get("sessionId") or item.get("session_id") or item.get("id") or "").strip()
                    and not _is_acpx_smoke_session(item)
                ]
                remote_record["ok"] = True
                remote_record["session_count"] = len(remote_sessions)
                remote_record["tools"] = payload.get("tools") or _remote_acpx_tools()
                remote_record["cwd"] = payload.get("cwd") or ""
                if payload.get("errors"):
                    remote_record["warning"] = "; ".join(str(e) for e in payload.get("errors") or [] if e)
                sessions.extend(remote_sessions)
            except Exception as exc:
                error = str(exc)
                remote_record["error"] = error
                errors.append(f"{config.destination} acpx: {error}")
            remotes.append(remote_record)

        if "claude" in transports:
            remote_record = _remote_payload(config, ok=False, session_count=0, transport="claude")
            try:
                payload = _run_remote_python(_remote_script_list_sessions(tail_lines), config=config)
            except Exception as exc:
                error = str(exc)
                remote_record["error"] = error
                errors.append(f"{config.destination} claude: {error}")
                remotes.append(remote_record)
                continue
            remote_sessions = [
                _normalize_session(dict(item), config=config)
                for item in payload.get("sessions", [])
                if isinstance(item, dict) and not _is_daemon_spare_session(item)
            ]
            for item in remote_sessions:
                item.setdefault("transport", "claude")
                item.setdefault("agent_tool", "claude")
            remote_record["ok"] = True
            remote_record["session_count"] = len(remote_sessions)
            remotes.append(remote_record)
            sessions.extend(remote_sessions)

    # Preserve older API expectations that the legacy Claude harness record is
    # the first remote entry while still attaching ACPX records for the same host.
    remotes.sort(key=lambda item: 0 if item.get("transport") == "claude" else 1)
    sessions.sort(key=lambda s: str(s.get("updated_at") or ""), reverse=True)
    if limit > 0:
        sessions = sessions[:limit]
    if sessions:
        _write_cached_sessions(sessions)

    if not any(remote.get("ok") for remote in remotes):
        cached = _load_cached_sessions()
        if limit > 0:
            cached = cached[:limit]
        if cached:
            return {
                "ok": False,
                "stale": True,
                "error": "; ".join(errors) or "remote Claude unavailable",
                "remote": _remote_payload(configs[0]) if configs else {},
                "remotes": remotes,
                "sessions": cached,
            }
        if errors:
            raise RuntimeError("; ".join(errors))

    return {
        "ok": any(remote.get("ok") for remote in remotes),
        "stale": False,
        "error": "; ".join(errors),
        "remote": _remote_payload(configs[0]) if configs else {},
        "remotes": remotes,
        "sessions": sessions,
    }


def _split_remote_target(target: str) -> tuple[str, str]:
    raw = str(target or "").strip()
    if REMOTE_KEY_SEPARATOR not in raw:
        return "", raw
    remote_ref, session_key = raw.split(REMOTE_KEY_SEPARATOR, 1)
    return remote_ref.strip(), session_key.strip()


def _config_matches_ref(config: RemoteClaudeConfig, remote_ref: str) -> bool:
    ref = str(remote_ref or "").strip().lower()
    if not ref:
        return True
    candidates = {
        config.destination.lower(),
        config.host.lower(),
        (config.hostname or "").lower(),
        f"{config.user}@{config.hostname}".lower() if config.hostname else "",
    }
    return ref in {item for item in candidates if item}


def _configs_for_target(target: str) -> tuple[list[RemoteClaudeConfig], str]:
    remote_ref, session_key = _split_remote_target(target)
    configs = [config for config in load_remote_claude_configs() if _config_matches_ref(config, remote_ref)]
    return configs or load_remote_claude_configs(), session_key


def _read_remote_acpx_messages(configs: list[RemoteClaudeConfig], tool: str, name: str, *, limit: int) -> dict[str, Any]:
    errors: list[str] = []
    for config in configs:
        try:
            payload = _run_remote_python(_remote_script_read_acpx_session(tool, name, max(limit * 3, 80)), config=config)
        except Exception as exc:
            errors.append(f"{config.destination}: {exc}")
            continue
        if not payload.get("found"):
            errors.append(f"{config.destination}: {payload.get('error') or 'session not found'}")
            continue
        ok = bool(payload.get("ok"))
        session = _normalize_acpx_session({"tool": tool, "name": name, "cwd": payload.get("cwd") or ""}, config=config)
        return {
            "ok": ok,
            "remote": _remote_payload(config),
            "session": session,
            "messages": _parse_acpx_read_messages(payload, limit=limit) if ok else [],
            "error": "" if ok else (payload.get("error") or "read failed"),
        }
    return {
        "ok": False,
        "remote": _remote_payload(configs[0]) if configs else {},
        "session": {},
        "messages": [],
        "error": "; ".join(errors) or "session not found",
    }


def read_remote_claude_messages(target: str, *, limit: int = 120) -> dict[str, Any]:
    configs, session_key = _configs_for_target(target)
    acpx_ref = _parse_acpx_session_ref(session_key)
    if acpx_ref:
        tool, name = acpx_ref
        return _read_remote_acpx_messages(configs, tool, name, limit=limit)

    line_limit = max(limit * 4, 80)
    errors: list[str] = []
    for config in configs:
        try:
            payload = _run_remote_python(_remote_script_read_messages(session_key, line_limit), config=config)
        except Exception as exc:
            errors.append(f"{config.destination}: {exc}")
            continue
        if not payload.get("found"):
            errors.append(f"{config.destination}: {payload.get('error') or 'session not found'}")
            continue
        session = dict(payload.get("session") or {})
        session = _normalize_session(session, config=config)
        messages = parse_claude_transcript_lines(payload.get("lines", []), limit=limit)
        return {
            "ok": True,
            "remote": _remote_payload(config),
            "session": session,
            "messages": messages,
        }
    return {
        "ok": False,
        "remote": _remote_payload(configs[0]) if configs else {},
        "error": "; ".join(errors) or "session not found",
        "messages": [],
    }


def _send_remote_acpx_message(configs: list[RemoteClaudeConfig], tool: str, name: str, message: str) -> dict[str, Any]:
    errors: list[str] = []
    for config in configs:
        try:
            payload = _run_remote_python(_remote_script_send_acpx_message(tool, name, message), config=config)
        except Exception as exc:
            errors.append(f"{config.destination}: {exc}")
            continue
        response = payload.get("response") if isinstance(payload.get("response"), dict) else {}
        if not payload.get("found"):
            errors.append(f"{config.destination}: {payload.get('error') or response.get('error') or 'session not found'}")
            continue
        ok = bool(response.get("ok"))
        stdout_text = _extract_acpx_text(str(payload.get("stdout") or ""))
        if stdout_text:
            response["text"] = stdout_text
        return {
            "ok": ok,
            "remote": _remote_payload(config),
            "session": _normalize_acpx_session({"tool": tool, "name": name, "cwd": payload.get("cwd") or ""}, config=config),
            "short": "",
            "response": response,
            "error": "" if ok else (response.get("error") or payload.get("error") or "send failed"),
        }
    return {
        "ok": False,
        "remote": _remote_payload(configs[0]) if configs else {},
        "session": {},
        "short": "",
        "response": {},
        "error": "; ".join(errors) or "session not found",
    }


def send_remote_claude_message(target: str, text: str) -> dict[str, Any]:
    message = str(text or "").strip()
    if not message:
        raise ValueError("message is empty")
    if len(message) > 50000:
        raise ValueError("message is too long")

    configs, session_key = _configs_for_target(target)
    acpx_ref = _parse_acpx_session_ref(session_key)
    if acpx_ref:
        tool, name = acpx_ref
        return _send_remote_acpx_message(configs, tool, name, message)

    errors: list[str] = []
    for config in configs:
        try:
            payload = _run_remote_python(_remote_script_send_message(session_key, message), config=config)
        except Exception as exc:
            errors.append(f"{config.destination}: {exc}")
            continue
        response = payload.get("response") if isinstance(payload.get("response"), dict) else {}
        if not payload.get("found"):
            errors.append(f"{config.destination}: {payload.get('error') or 'session not found'}")
            continue
        ok = bool(response.get("ok"))
        return {
            "ok": ok,
            "remote": _remote_payload(config),
            "session": _normalize_session(dict(payload.get("session") or {}), config=config),
            "short": payload.get("short") or "",
            "response": response,
            "error": "" if ok else (response.get("error") or payload.get("error") or "send failed"),
        }
    return {
        "ok": False,
        "remote": _remote_payload(configs[0]) if configs else {},
        "session": {},
        "short": "",
        "response": {},
        "error": "; ".join(errors) or "session not found",
    }


def rename_remote_claude_session(target: str, name: str) -> dict[str, Any]:
    session_key = str(target or "").strip()
    clean_name = " ".join(str(name or "").split())
    if not session_key:
        raise ValueError("session target is empty")
    if not clean_name:
        raise ValueError("session name is empty")
    if len(clean_name) > 120:
        clean_name = clean_name[:117].rstrip() + "..."

    configs, session_key = _configs_for_target(session_key)
    acpx_ref = _parse_acpx_session_ref(session_key)
    if acpx_ref:
        tool, session_name = acpx_ref
        return {
            "ok": False,
            "remote": _remote_payload(configs[0]) if configs else {},
            "session": _normalize_acpx_session({"tool": tool, "name": session_name}, config=configs[0]) if configs else {},
            "short": "",
            "old_name": session_name,
            "name": clean_name,
            "roster_updated": False,
            "roster_error": "",
            "error": "remote ACPX sessions do not support rename yet",
        }

    errors: list[str] = []
    for config in configs:
        try:
            payload = _run_remote_python(_remote_script_rename_session(session_key, clean_name), config=config)
        except Exception as exc:
            errors.append(f"{config.destination}: {exc}")
            continue
        if not payload.get("found"):
            errors.append(f"{config.destination}: {payload.get('error') or 'session not found'}")
            continue
        ok = bool(payload.get("ok"))
        return {
            "ok": ok,
            "remote": _remote_payload(config),
            "session": _normalize_session(dict(payload.get("session") or {}), config=config),
            "short": payload.get("short") or "",
            "old_name": payload.get("old_name") or "",
            "name": payload.get("name") or clean_name,
            "roster_updated": bool(payload.get("roster_updated")),
            "roster_error": payload.get("roster_error") or "",
            "error": "" if ok else (payload.get("error") or payload.get("roster_error") or "rename failed"),
        }
    return {
        "ok": False,
        "remote": _remote_payload(configs[0]) if configs else {},
        "session": {},
        "short": "",
        "old_name": "",
        "name": clean_name,
        "roster_updated": False,
        "roster_error": "",
        "error": "; ".join(errors) or "session not found",
    }


def _close_remote_acpx_session(configs: list[RemoteClaudeConfig], tool: str, name: str) -> dict[str, Any]:
    errors: list[str] = []
    for config in configs:
        try:
            payload = _run_remote_python(_remote_script_close_acpx_session(tool, name), config=config)
        except Exception as exc:
            errors.append(f"{config.destination}: {exc}")
            continue
        if not payload.get("found"):
            errors.append(f"{config.destination}: {payload.get('error') or 'session not found'}")
            continue
        ok = bool(payload.get("ok"))
        return {
            "ok": ok,
            "remote": _remote_payload(config),
            "session": _normalize_acpx_session({"tool": tool, "name": name, "cwd": payload.get("cwd") or ""}, config=config),
            "short": "",
            "pid": None,
            "archive_path": "",
            "session_file_gone": ok,
            "daemon_kill": {},
            "kill": {},
            "archive": {},
            "error": "" if ok else (payload.get("error") or "close failed"),
        }
    return {
        "ok": False,
        "remote": _remote_payload(configs[0]) if configs else {},
        "session": {},
        "short": "",
        "pid": None,
        "archive_path": "",
        "session_file_gone": False,
        "daemon_kill": {},
        "kill": {},
        "archive": {},
        "error": "; ".join(errors) or "session not found",
    }


def close_remote_claude_session(target: str, *, force: bool = True) -> dict[str, Any]:
    session_key = str(target or "").strip()
    if not session_key:
        raise ValueError("session target is empty")

    configs, session_key = _configs_for_target(session_key)
    acpx_ref = _parse_acpx_session_ref(session_key)
    if acpx_ref:
        tool, name = acpx_ref
        return _close_remote_acpx_session(configs, tool, name)

    errors: list[str] = []
    for config in configs:
        try:
            payload = _run_remote_python(_remote_script_close_session(session_key, force=force), config=config)
        except Exception as exc:
            errors.append(f"{config.destination}: {exc}")
            continue
        if not payload.get("found"):
            errors.append(f"{config.destination}: {payload.get('error') or 'session not found'}")
            continue
        ok = bool(payload.get("ok"))
        return {
            "ok": ok,
            "remote": _remote_payload(config),
            "session": _normalize_session(dict(payload.get("session") or {}), config=config),
            "short": payload.get("short") or "",
            "pid": payload.get("pid"),
            "archive_path": payload.get("archive_path") or "",
            "session_file_gone": bool(payload.get("session_file_gone")),
            "daemon_kill": payload.get("daemon_kill") or {},
            "kill": payload.get("kill") or {},
            "archive": payload.get("archive") or {},
            "error": "" if ok else (payload.get("error") or payload.get("archive", {}).get("error") or payload.get("kill", {}).get("error") or "close failed"),
        }
    return {
        "ok": False,
        "remote": _remote_payload(configs[0]) if configs else {},
        "session": {},
        "short": "",
        "pid": None,
        "archive_path": "",
        "session_file_gone": False,
        "daemon_kill": {},
        "kill": {},
        "archive": {},
        "error": "; ".join(errors) or "session not found",
    }
