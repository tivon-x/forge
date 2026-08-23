"""Project-resource trust preflight and the user trust store.

Trust is deliberately resolved from filesystem metadata only.  Project files are
not opened until the caller has a :class:`TrustResult` and has built filtered
resource paths from it.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from forge_coding.resources.base import ForgeResourcePaths

PROJECT_MARKERS = (".git", "pyproject.toml", "uv.lock", "setup.py", "package.json")
TRUST_FILE_VERSION = 1
TRUST_LOCK_TIMEOUT_SECONDS = 10.0
TRUST_STALE_LOCK_SECONDS = 600.0


class TrustError(ValueError):
    """Raised when a trust decision cannot be persisted or interpreted."""


@dataclass(frozen=True, slots=True)
class TrustRecord:
    """One persisted trust decision."""

    decision: str
    scope: str = "folder"


@dataclass(frozen=True, slots=True)
class TrustResult:
    """The immutable result of a project trust preflight."""

    cwd: Path
    project_root: Path
    project_resources_present: bool
    project_resources_allowed: bool
    decision: str
    source: str
    store_path: Path

    @property
    def allowed(self) -> bool:
        """Return whether project resources may be loaded."""
        return self.project_resources_allowed

    def describe(self) -> str:
        """Return a bounded user-facing status line."""
        if not self.project_resources_present:
            return "No project resources detected; project trust is not required."
        status = "trusted" if self.project_resources_allowed else "not trusted"
        return f"Project resources are {status} ({self.source}); run /reload after changes."


def canonical_path(path: Path | str) -> Path:
    """Return a stable resolved path for trust keys.

    ``strict=False`` is intentional: a target cwd can be created by the session
    manager after preflight, while all existing components are still resolved.
    ``normcase`` gives Windows drive/UNC keys one representation.
    """

    candidate = Path(path).expanduser()
    try:
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        resolved = candidate.absolute()
    normalized = os.path.normcase(os.path.normpath(str(resolved)))
    return Path(normalized)


class TrustStore:
    """Small JSON trust store with locked, atomic updates.

    The store is user-owned state.  A malformed file is read as empty and marked
    corrupt; writes refuse to replace it, so a bad file cannot silently become an
    allow-all or empty policy.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = (path or (Path.home() / ".forge" / "trust.json")).expanduser()
        self.last_error: str | None = None

    def lookup(self, cwd: Path) -> tuple[TrustRecord, Path] | None:
        """Return the nearest persisted decision for ``cwd``."""
        records = self._read_records()
        key = canonical_path(cwd)
        for candidate in (key, *key.parents):
            record = records.get(str(candidate))
            if record is not None:
                return record, candidate
        return None

    def set(
        self,
        cwd: Path,
        decision: str,
        *,
        scope: str = "folder",
        lock_timeout_seconds: float = TRUST_LOCK_TIMEOUT_SECONDS,
    ) -> None:
        """Persist an allow/deny decision for a folder or its parent."""
        normalized_decision = _normalize_store_decision(decision)
        if scope not in {"folder", "parent"}:
            raise TrustError("Trust scope must be folder or parent")
        current_key = canonical_path(cwd)
        key = current_key.parent if scope == "parent" else current_key
        try:
            with self._lock(timeout_seconds=lock_timeout_seconds):
                records = self._read_records()
                if self.last_error is not None:
                    raise TrustError("Trust store is corrupt; refusing to overwrite it")
                if scope == "parent":
                    records.pop(str(current_key), None)
                records[str(key)] = TrustRecord(normalized_decision, "folder")
                self._write_records(records)
        except TrustError:
            raise
        except OSError:
            raise TrustError("Could not update trust store") from None

    def remove(self, cwd: Path) -> bool:
        """Remove an exact persisted decision, returning whether one existed."""
        key = str(canonical_path(cwd))
        try:
            with self._lock(timeout_seconds=TRUST_LOCK_TIMEOUT_SECONDS):
                records = self._read_records()
                if self.last_error is not None:
                    raise TrustError("Trust store is corrupt; refusing to overwrite it")
                if key not in records:
                    return False
                del records[key]
                self._write_records(records)
                return True
        except TrustError:
            raise
        except OSError:
            raise TrustError("Could not update trust store") from None

    def _read_records(self) -> dict[str, TrustRecord]:
        self.last_error = None
        try:
            raw = self.path.read_text(encoding="utf-8")
            payload = json.loads(raw)
            return _parse_records(payload)
        except FileNotFoundError:
            return {}
        except (OSError, UnicodeDecodeError, TypeError, ValueError, RuntimeError) as exc:
            self.last_error = type(exc).__name__
            return {}

    def _write_records(self, records: Mapping[str, TrustRecord]) -> None:
        payload = {
            "version": TRUST_FILE_VERSION,
            "decisions": {
                key: {"decision": value.decision, "scope": value.scope}
                for key, value in sorted(records.items())
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temp_name = _temporary_path(self.path)
        temp_path = Path(temp_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.path)
        finally:
            with suppress(FileNotFoundError):
                temp_path.unlink()

    @contextmanager
    def _lock(self, *, timeout_seconds: float) -> Iterator[None]:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            raise TrustError("Could not prepare trust store") from None
        lock_path = self.path.with_name(f".{self.path.name}.lock")
        started = time.monotonic()
        descriptor: int | None = None
        while descriptor is None:
            try:
                descriptor = os.open(
                    lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
            except FileExistsError:
                if _stale_lock(lock_path):
                    with suppress(OSError):
                        lock_path.unlink()
                    continue
                if time.monotonic() - started >= timeout_seconds:
                    raise TrustError("Timed out waiting for trust store lock") from None
                time.sleep(0.02)
            except OSError:
                raise TrustError("Could not acquire trust store lock") from None
            else:
                try:
                    payload = f"{os.getpid()}\n".encode("ascii")
                    if os.write(descriptor, payload) != len(payload):
                        raise TrustError("Could not initialize trust store lock")
                except BaseException:
                    with suppress(OSError):
                        os.close(descriptor)
                    with suppress(OSError):
                        lock_path.unlink()
                    descriptor = None
                    raise
        try:
            yield
        finally:
            with suppress(OSError):
                os.close(descriptor)
            with suppress(FileNotFoundError):
                lock_path.unlink()


def resolve_project_trust(
    cwd: Path,
    *,
    paths: ForgeResourcePaths | None = None,
    store: TrustStore | None = None,
    cli_override: str | None = None,
    env: Mapping[str, str] | None = None,
    session_decision: str | None = None,
    interactive: bool = False,
    stderr: TextIO | None = None,
) -> TrustResult:
    """Resolve trust without reading project resource contents."""
    resolved_cwd = canonical_path(cwd)
    project_root = find_project_root(resolved_cwd)
    resource_paths = paths or ForgeResourcePaths(cwd=resolved_cwd)
    if resource_paths.cwd is None or canonical_path(resource_paths.cwd) != resolved_cwd:
        resource_paths = ForgeResourcePaths(
            root=resource_paths.root,
            cwd=resolved_cwd,
            agents_root=resource_paths.agents_root,
            paths=resource_paths.paths,
            project_resources_allowed=resource_paths.project_resources_allowed,
        )
    present = project_resources_present(resource_paths, project_root=project_root)
    trust_store = store or TrustStore()
    store_path = trust_store.path

    override = _normalize_override(cli_override)
    source = "cli"
    if override is None:
        environment = env if env is not None else os.environ
        override = _normalize_override(environment.get("FORGE_TRUST"))
        source = "environment" if override is not None else source
    if override is None:
        override = _normalize_override(session_decision)
        source = "session" if override is not None else source

    if not present:
        return _result(resolved_cwd, project_root, present, False, "none", "none", store_path)

    if override is None:
        stored = trust_store.lookup(resolved_cwd)
        if stored is not None:
            override = stored[0].decision
            source = "trust store"

    if override is None or override == "ask":
        if interactive and _can_prompt():
            choice = _prompt_for_trust(resolved_cwd, stderr=stderr)
            if choice in {"once", "always", "parent"}:
                if choice != "once":
                    try:
                        trust_store.set(
                            resolved_cwd,
                            "allow",
                            scope="parent" if choice == "parent" else "folder",
                        )
                        source = "trust store"
                    except TrustError as exc:
                        _warn(stderr, f"Could not save trust decision: {exc}")
                        source = "session"
                return _result(
                    resolved_cwd,
                    project_root,
                    present,
                    True,
                    "allow",
                    "session" if choice == "once" else source,
                    store_path,
                )
            override = "deny"
            source = "prompt"
        else:
            override = "deny"
            source = "default"
            _warn(stderr, "Project resources found; trust not granted (default deny).")

    allowed = override == "allow"
    return _result(
        resolved_cwd,
        project_root,
        present,
        allowed,
        override,
        source,
        store_path,
    )


def project_resources_present(paths: ForgeResourcePaths, *, project_root: Path) -> bool:
    """Return whether any loadable project resource exists using metadata only."""
    if paths.cwd is None:
        return False
    cwd = canonical_path(paths.cwd)
    if not _within(cwd, project_root):
        return False
    candidates = [
        *ancestor_agents_files(project_root, cwd),
        paths._paths().project_forge_dir(cwd) / "AGENTS.md",
        paths._paths().project_agents_dir(cwd) / "AGENTS.md",
    ]
    if any(
        _safe_regular_file(path, project_root) or _has_symlink_component(path, project_root)
        for path in candidates
    ):
        return True
    forge_root = paths._paths().project_forge_dir(cwd)
    agents_root = paths._paths().project_agents_dir(cwd)
    return (
        _directory_has_skill(forge_root / "skills", project_root)
        or _directory_has_markdown(forge_root / "prompts", project_root)
        or _directory_has_agent(forge_root / "agents", project_root)
        or _directory_has_skill(agents_root / "skills", project_root)
        or _directory_has_markdown(agents_root / "prompts", project_root)
    )


def find_project_root(cwd: Path) -> Path:
    """Find the nearest project marker without opening project files."""
    try:
        resolved = Path(cwd).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        resolved = Path(cwd).expanduser().absolute()
    for candidate in (resolved, *resolved.parents):
        for marker in PROJECT_MARKERS:
            with suppress(OSError):
                if (candidate / marker).exists():
                    return candidate
    return resolved


def ancestor_agents_files(project_root: Path, cwd: Path) -> tuple[Path, ...]:
    """Return the ancestor AGENTS.md chain from project root to cwd."""
    try:
        relative = cwd.relative_to(project_root)
    except ValueError:
        return (cwd / "AGENTS.md",)
    paths = [project_root / "AGENTS.md"]
    current = project_root
    for part in relative.parts:
        current = current / part
        paths.append(current / "AGENTS.md")
    return tuple(paths)


def project_path_is_safe(path: Path, *, project_root: Path) -> bool:
    """Reject symlink components and resolved paths outside the project root."""
    candidate = Path(path).expanduser()
    try:
        relative = candidate.absolute().relative_to(project_root.absolute())
    except ValueError:
        return False
    current = project_root
    for part in relative.parts:
        current = current / part
        try:
            if stat.S_ISLNK(current.lstat().st_mode):
                return False
        except FileNotFoundError:
            break
        except OSError:
            return False
    try:
        return _within(canonical_path(candidate), canonical_path(project_root))
    except (OSError, RuntimeError, ValueError):
        return False


def _directory_has_skill(path: Path, project_root: Path) -> bool:
    if not _safe_directory(path, project_root):
        return _has_symlink_component(path, project_root)
    try:
        for entry in path.iterdir():
            if not project_path_is_safe(entry, project_root=project_root):
                if _has_symlink_component(entry, project_root):
                    return True
                continue
            if not entry.is_dir():
                continue
            marker = entry / "SKILL.md"
            if _safe_regular_file(marker, project_root):
                return True
    except OSError:
        return False
    return False


def _directory_has_agent(path: Path, project_root: Path) -> bool:
    if not _safe_directory(path, project_root):
        return _has_symlink_component(path, project_root)
    try:
        for entry in path.iterdir():
            if not project_path_is_safe(entry, project_root=project_root):
                if _has_symlink_component(entry, project_root):
                    return True
                continue
            if not entry.is_dir():
                continue
            if _safe_regular_file(entry / "AGENT.md", project_root):
                return True
    except OSError:
        return False
    return False


def _directory_has_markdown(path: Path, project_root: Path) -> bool:
    if not _safe_directory(path, project_root):
        return _has_symlink_component(path, project_root)
    try:
        for entry in path.iterdir():
            if _has_symlink_component(entry, project_root):
                return True
            if entry.suffix.lower() == ".md" and _safe_regular_file(entry, project_root):
                return True
        return False
    except OSError:
        return False


def _safe_directory(path: Path, project_root: Path) -> bool:
    return project_path_is_safe(path, project_root=project_root) and path.is_dir()


def _safe_regular_file(path: Path, project_root: Path) -> bool:
    if not project_path_is_safe(path, project_root=project_root):
        return False
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except OSError:
        return False


def _has_symlink_component(path: Path, project_root: Path) -> bool:
    """Return whether a project resource path contains any symlink component."""
    try:
        relative = Path(path).absolute().relative_to(Path(project_root).absolute())
    except (OSError, ValueError, RuntimeError):
        return False
    current = Path(project_root)
    for part in relative.parts:
        current /= part
        try:
            if stat.S_ISLNK(current.lstat().st_mode):
                return True
        except FileNotFoundError:
            break
        except OSError:
            return True
    return False


def _parse_records(payload: object) -> dict[str, TrustRecord]:
    if not isinstance(payload, dict):
        raise ValueError("trust store must be an object")
    if type(payload.get("version")) is not int or payload["version"] != TRUST_FILE_VERSION:
        raise ValueError("unsupported trust store version")
    if set(payload) != {"version", "decisions"}:
        raise ValueError("unknown trust store fields")
    decisions = payload.get("decisions")
    if not isinstance(decisions, dict):
        raise ValueError("trust decisions must be an object")
    parsed: dict[str, TrustRecord] = {}
    for raw_key, raw_value in decisions.items():
        if not isinstance(raw_key, str):
            raise ValueError("trust key must be text")
        key_path = Path(raw_key).expanduser()
        if not key_path.is_absolute() or str(canonical_path(key_path)) != os.path.normcase(
            os.path.normpath(str(key_path))
        ):
            raise ValueError("trust key must be an absolute canonical path")
        if isinstance(raw_value, str):
            raise ValueError("trust decision must be an object")
        elif isinstance(raw_value, dict):
            if set(raw_value) != {"decision", "scope"}:
                raise ValueError("unknown trust decision fields")
            decision = raw_value.get("decision")
            scope = raw_value.get("scope")
        else:
            raise ValueError("trust decision must be text or object")
        if not isinstance(decision, str) or decision not in {"allow", "deny"}:
            raise ValueError("trust decision must be allow or deny")
        if scope != "folder":
            raise ValueError("trust scope must be folder")
        parsed[str(canonical_path(raw_key))] = TrustRecord(decision, scope)
    return parsed


def _normalize_override(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().casefold()
    if normalized in {"yes", "always", "allow", "true", "1"}:
        return "allow"
    if normalized in {"no", "never", "deny", "false", "0"}:
        return "deny"
    if normalized == "ask":
        return "ask"
    raise TrustError("Trust must be yes, no, or ask")


def _normalize_store_decision(value: str) -> str:
    normalized = _normalize_override(value)
    if normalized not in {"allow", "deny"}:
        raise TrustError("Persisted trust must be allow or deny")
    return normalized


def _result(
    cwd: Path,
    project_root: Path,
    present: bool,
    allowed: bool,
    decision: str,
    source: str,
    store_path: Path,
) -> TrustResult:
    return TrustResult(
        cwd=cwd,
        project_root=project_root,
        project_resources_present=present,
        project_resources_allowed=allowed,
        decision=decision,
        source=source,
        store_path=store_path,
    )


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _temporary_path(path: Path) -> tuple[int, str]:
    import tempfile

    return tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)


def _stale_lock(path: Path) -> bool:
    try:
        age = time.time() - path.stat().st_mtime
        if age <= TRUST_STALE_LOCK_SECONDS:
            return False
        raw = path.read_text(encoding="ascii").strip()
        pid = int(raw)
    except (OSError, UnicodeDecodeError, ValueError, RuntimeError):
        return False
    if pid <= 0:
        return False
    return not _pid_exists(pid)


def _pid_exists(pid: int) -> bool:
    if os.name == "nt":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _can_prompt() -> bool:
    return bool(getattr(sys.stdin, "isatty", lambda: False)())


def _prompt_for_trust(cwd: Path, *, stderr: TextIO | None) -> str:
    stream: TextIO = stderr or sys.stderr
    print(
        f"Project resources detected under {cwd}. Trust once, always, parent, or deny? "
        "[o/a/p/n]",
        file=stream,
        flush=True,
    )
    try:
        answer = input().strip().casefold()
    except (EOFError, KeyboardInterrupt):
        return "deny"
    return {
        "o": "once",
        "once": "once",
        "a": "always",
        "always": "always",
        "p": "parent",
        "parent": "parent",
    }.get(answer, "deny")


def _warn(stderr: TextIO | None, message: str) -> None:
    print(f"forge: {message}", file=stderr or sys.stderr)
