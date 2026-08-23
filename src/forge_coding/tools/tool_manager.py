"""Managed ``rg``/``fd`` discovery and verified installation.

The manager intentionally has one concrete implementation.  It is a small
boundary around the two binaries used by the native search tools; no generic
package manager or download abstraction belongs here.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import re
import shutil
import signal as process_signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any

from forge_agent.tools import ToolCancellationToken

MAX_DOWNLOAD_BYTES = 256 * 1024 * 1024
API_TIMEOUT_SECONDS = 10.0
API_IO_TIMEOUT_SECONDS = 2.0
DOWNLOAD_TIMEOUT_SECONDS = 120.0
DOWNLOAD_IO_TIMEOUT_SECONDS = 5.0
VERSION_TIMEOUT_SECONDS = 10.0
MAX_VERSION_OUTPUT_BYTES = 4096
LOCK_TIMEOUT_SECONDS = 130.0
STALE_LOCK_SECONDS = 10 * 60
_VERSION_BODY = (
    r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?"
)
_VERSION_RE = re.compile(rf"^v?({_VERSION_BODY})$")
_VERSION_TOKEN_RE = re.compile(rf"(?<![0-9A-Za-z])v?({_VERSION_BODY})(?![0-9A-Za-z])")
_DIGEST_RE = re.compile(r"^sha256:([0-9a-fA-F]{64})$")
_API_HOSTS = frozenset({"api.github.com"})
_ASSET_HOSTS = frozenset(
    {
        "github.com",
        "objects.githubusercontent.com",
        "release-assets.githubusercontent.com",
        "github-releases.githubusercontent.com",
    }
)


class ToolManagerError(RuntimeError):
    """A bounded, user-actionable managed-tool error."""


@dataclass(frozen=True, slots=True)
class ManagedToolSpec:
    name: str
    repository: str
    executable_name: str


TOOL_SPECS: Mapping[str, ManagedToolSpec] = MappingProxyType(
    {
        "rg": ManagedToolSpec("rg", "BurntSushi/ripgrep", "rg"),
        "fd": ManagedToolSpec("fd", "sharkdp/fd", "fd"),
    }
)


class ToolManager:
    """Discover or install one of Forge's fixed search binaries.

    The cache is per process.  ``ensure_tool`` is synchronous because the
    rare network path is already isolated behind a single cross-process lock;
    native tools call it in a worker thread so their async stream remains
    cancellable.
    """

    def __init__(
        self,
        *,
        home: str | Path | None = None,
        environ: Mapping[str, str] | None = None,
        system: str | None = None,
        machine: str | None = None,
        urlopen: Callable[..., Any] | None = None,
    ) -> None:
        self.home = Path.home() if home is None else Path(home).expanduser()
        self.bin_dir = self.home / ".forge" / "bin"
        self.environ = os.environ if environ is None else environ
        self.system = (system or platform.system()).casefold()
        self.machine = (machine or platform.machine()).casefold()
        self._urlopen = urllib.request.urlopen if urlopen is None else urlopen
        self._resolved: dict[str, Path] = {}

    def ensure_tool(
        self,
        name: str,
        *,
        signal: ToolCancellationToken | None = None,
    ) -> Path:
        """Return a validated executable, installing it only when necessary."""

        spec = TOOL_SPECS.get(name)
        if spec is None:
            raise ToolManagerError(f"Unsupported managed tool: {name}")
        cached = self._resolved.get(name)
        if cached is not None:
            return cached

        private = self._private_path(spec)
        if not private.is_symlink() and _valid_executable(private, spec.name):
            self._resolved[name] = private
            return private
        if private.exists():
            self._quarantine(private)

        path_candidate = self._path_candidate(spec)
        if path_candidate is not None:
            self._resolved[name] = path_candidate
            return path_candidate

        if _offline(self.environ):
            raise ToolManagerError(
                f"{name} is unavailable while FORGE_OFFLINE is enabled; "
                f"install {name} manually and place it on PATH"
            )

        try:
            self.bin_dir.mkdir(parents=True, exist_ok=True)
            lock = self.bin_dir / f".{name}.lock"
            self._acquire_lock(lock, signal=signal)
        except ToolManagerError:
            raise
        except OSError as exc:
            raise ToolManagerError(
                f"Unable to prepare tool installation: {_phase_error(exc)}"
            ) from None
        try:
            # A second process may have installed the binary while we waited.
            if not private.is_symlink() and _valid_executable(private, spec.name):
                self._resolved[name] = private
                return private
            if private.exists():
                self._quarantine(private)
            installed = self._download_and_install(spec, private, signal=signal)
            self._resolved[name] = installed
            return installed
        except ToolManagerError:
            raise
        except Exception as exc:  # noqa: BLE001 - bounded external boundary
            raise ToolManagerError(f"Unable to install {name}: {_phase_error(exc)}") from None
        finally:
            _remove_lock(lock)

    def _private_path(self, spec: ManagedToolSpec) -> Path:
        suffix = ".exe" if self.system == "windows" else ""
        return self.bin_dir / f"{spec.executable_name}{suffix}"

    def _path_candidate(self, spec: ManagedToolSpec) -> Path | None:
        names = ("rg",) if spec.name == "rg" else ("fd", "fdfind")
        for candidate_name in names:
            candidate = shutil.which(candidate_name, path=self.environ.get("PATH", ""))
            if candidate is None:
                continue
            path = Path(candidate)
            if _valid_executable(path, spec.name):
                return path
        return None

    def _quarantine(self, path: Path) -> None:
        try:
            stamp = f".invalid-{os.getpid()}-{time.time_ns()}"
            path.replace(path.with_name(path.name + stamp))
        except OSError:
            # A concurrent installer may have removed it; it is safe to proceed.
            pass

    def _acquire_lock(
        self,
        path: Path,
        *,
        signal: ToolCancellationToken | None,
    ) -> None:
        started = time.monotonic()
        while True:
            if signal is not None and signal.is_cancelled():
                raise ToolManagerError("Tool installation cancelled while waiting for lock")
            try:
                descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                if _stale_lock(path):
                    _remove_lock(path)
                    continue
                if time.monotonic() - started >= LOCK_TIMEOUT_SECONDS:
                    raise ToolManagerError("Timed out waiting for tool installation lock") from None
                time.sleep(0.1)
                continue
            except OSError as exc:
                raise ToolManagerError(
                    f"Unable to create tool installation lock: {_phase_error(exc)}"
                ) from None
            try:
                try:
                    payload = f"{os.getpid()}\n".encode("ascii")
                    if os.write(descriptor, payload) != len(payload):
                        raise OSError("short lock write")
                finally:
                    os.close(descriptor)
            except OSError as exc:
                _remove_lock(path)
                raise ToolManagerError(
                    f"Unable to initialize tool installation lock: {_phase_error(exc)}"
                ) from None
            return

    def _download_and_install(
        self,
        spec: ManagedToolSpec,
        destination: Path,
        *,
        signal: ToolCancellationToken | None,
    ) -> Path:
        if signal is not None and signal.is_cancelled():
            raise ToolManagerError("Tool installation cancelled")
        system, arch, extension = self._asset_platform()
        release = self._latest_release(spec, signal=signal)
        tag = _release_version(release.get("tag_name"))
        asset_name = _asset_name(spec, tag, system, arch, extension)
        assets = release.get("assets")
        if not isinstance(assets, list):
            raise ToolManagerError("release metadata has no assets")
        matching = [
            item for item in assets if isinstance(item, dict) and item.get("name") == asset_name
        ]
        if len(matching) != 1:
            raise ToolManagerError("release has no unique platform asset")
        asset = matching[0]
        digest_value = asset.get("digest")
        digest_match = _DIGEST_RE.fullmatch(digest_value) if isinstance(digest_value, str) else None
        if digest_match is None:
            raise ToolManagerError("release asset has no valid sha256 digest")
        download_url = asset.get("browser_download_url")
        if (
            not isinstance(download_url, str)
            or not _trusted_https_url(download_url, _ASSET_HOSTS)
        ):
            raise ToolManagerError("release asset URL is not an official HTTPS host")

        archive = self._download(download_url, asset, signal=signal)
        try:
            digest = hashlib.sha256()
            with archive.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    _check_cancelled(signal)
                    digest.update(chunk)
            _check_cancelled(signal)
            actual_digest = digest.hexdigest()
            if actual_digest.casefold() != digest_match.group(1).casefold():
                raise ToolManagerError("release asset digest mismatch")
            binary = _extract_binary(
                archive,
                archive_name=asset_name,
                executable_name=spec.executable_name + (".exe" if system == "windows" else ""),
            )
            _check_cancelled(signal)
            try:
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=f".{destination.name}.{os.getpid()}.",
                    suffix=".tmp",
                    dir=destination.parent,
                )
            except OSError as exc:
                raise ToolManagerError(
                    f"Unable to prepare binary installation: {_phase_error(exc)}"
                ) from None
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    descriptor = -1
                    handle.write(binary)
                if self.system != "windows":
                    temporary.chmod(0o755)
                if not _valid_executable(temporary, spec.name, expected_version=tag):
                    raise ToolManagerError("downloaded binary failed version validation")
                _check_cancelled(signal)
                os.replace(temporary, destination)
                return destination
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
                _remove_path(temporary)
        finally:
            _remove_path(archive)

    def _latest_release(
        self,
        spec: ManagedToolSpec,
        *,
        signal: ToolCancellationToken | None = None,
    ) -> dict[str, Any]:
        url = f"https://api.github.com/repos/{spec.repository}/releases/latest"
        deadline = time.monotonic() + API_TIMEOUT_SECONDS
        try:
            with self._urlopen(
                urllib.request.Request(
                    url,
                    headers={"Accept": "application/vnd.github+json", "User-Agent": "forge"},
                ),
                timeout=API_IO_TIMEOUT_SECONDS,
            ) as response:
                _require_https_response(response, url, allowed_hosts=_API_HOSTS)
                payload = bytearray()
                while len(payload) <= 4 * 1024 * 1024:
                    _check_cancelled(signal)
                    if time.monotonic() >= deadline:
                        raise ToolManagerError("release metadata request timed out")
                    chunk = response.read(
                        min(64 * 1024, 4 * 1024 * 1024 + 1 - len(payload))
                    )
                    if time.monotonic() >= deadline:
                        raise ToolManagerError("release metadata request timed out")
                    _check_cancelled(signal)
                    if not chunk:
                        break
                    payload.extend(chunk)
        except ToolManagerError:
            raise
        except Exception as exc:  # noqa: BLE001 - do not expose response body
            raise ToolManagerError(
                f"release metadata request failed: {_phase_error(exc)}"
            ) from None
        if len(payload) > 4 * 1024 * 1024:
            raise ToolManagerError("release metadata is too large")
        try:
            decoded = json.loads(bytes(payload).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ToolManagerError("release metadata is invalid JSON") from None
        if not isinstance(decoded, dict):
            raise ToolManagerError("release metadata is not an object")
        return decoded

    def _download(
        self,
        url: str,
        asset: Mapping[str, Any],
        *,
        signal: ToolCancellationToken | None,
    ) -> Path:
        if signal is not None and signal.is_cancelled():
            raise ToolManagerError("Tool installation cancelled")
        try:
            descriptor, temporary_name = tempfile.mkstemp(prefix="forge-tool-", dir=self.bin_dir)
        except OSError as exc:
            raise ToolManagerError(
                f"Unable to prepare binary download: {_phase_error(exc)}"
            ) from None
        temporary = Path(temporary_name)
        expected_size = asset.get("size")
        deadline = time.monotonic() + DOWNLOAD_TIMEOUT_SECONDS
        try:
            with (
                self._urlopen(
                    urllib.request.Request(
                        url,
                        headers={"Accept": "application/octet-stream", "User-Agent": "forge"},
                    ),
                    timeout=DOWNLOAD_IO_TIMEOUT_SECONDS,
                ) as response,
                os.fdopen(descriptor, "wb") as handle,
            ):
                descriptor = -1
                _require_https_response(response, url, allowed_hosts=_ASSET_HOSTS)
                header_size = response.headers.get("Content-Length")
                declared_size = _positive_int(header_size)
                if header_size is not None and declared_size is None:
                    raise ToolManagerError("invalid Content-Length")
                if declared_size is not None and declared_size > MAX_DOWNLOAD_BYTES:
                    raise ToolManagerError("download exceeds 256 MiB limit")
                if expected_size is not None and (
                    not isinstance(expected_size, int)
                    or isinstance(expected_size, bool)
                    or expected_size < 0
                ):
                    raise ToolManagerError("release asset has invalid size")
                if isinstance(expected_size, int) and expected_size > MAX_DOWNLOAD_BYTES:
                    raise ToolManagerError("release asset exceeds 256 MiB limit")
                total = 0
                while True:
                    _check_download_active(signal, deadline)
                    chunk = response.read(64 * 1024)
                    _check_download_active(signal, deadline)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_DOWNLOAD_BYTES:
                        raise ToolManagerError("download exceeds 256 MiB limit")
                    handle.write(chunk)
                if declared_size is not None and declared_size != total:
                    raise ToolManagerError("download size does not match Content-Length")
                if isinstance(expected_size, int) and expected_size != total:
                    raise ToolManagerError("download size does not match release metadata")
            return temporary
        except ToolManagerError:
            if descriptor >= 0:
                os.close(descriptor)
            _remove_path(temporary)
            raise
        except Exception as exc:  # noqa: BLE001 - no body/path disclosure
            if descriptor >= 0:
                os.close(descriptor)
            _remove_path(temporary)
            raise ToolManagerError(f"binary download failed: {_phase_error(exc)}") from None

    def _asset_platform(self) -> tuple[str, str, str]:
        if self.system == "windows":
            platform_name = "windows"
            extension = "zip"
        elif self.system == "linux":
            platform_name = "linux"
            extension = "tar.gz"
        elif self.system == "darwin":
            platform_name = "darwin"
            extension = "tar.gz"
        else:
            raise ToolManagerError("unsupported operating system for managed tool")
        arch_map = {
            "amd64": "x86_64",
            "x86_64": "x86_64",
            "x64": "x86_64",
            "aarch64": "aarch64",
            "arm64": "aarch64",
        }
        arch = arch_map.get(self.machine)
        if arch is None:
            raise ToolManagerError("unsupported CPU architecture for managed tool")
        return platform_name, arch, extension


def _asset_name(spec: ManagedToolSpec, version: str, system: str, arch: str, extension: str) -> str:
    if spec.name == "rg":
        target = {
            ("windows", "x86_64"): "x86_64-pc-windows-msvc",
            ("windows", "aarch64"): "aarch64-pc-windows-msvc",
            ("linux", "x86_64"): "x86_64-unknown-linux-musl",
            ("linux", "aarch64"): "aarch64-unknown-linux-gnu",
            ("darwin", "x86_64"): "x86_64-apple-darwin",
            ("darwin", "aarch64"): "aarch64-apple-darwin",
        }.get((system, arch))
        prefix = "ripgrep-"
    else:
        target = {
            ("windows", "x86_64"): "x86_64-pc-windows-msvc",
            ("windows", "aarch64"): "aarch64-pc-windows-msvc",
            ("linux", "x86_64"): "x86_64-unknown-linux-gnu",
            ("linux", "aarch64"): "aarch64-unknown-linux-gnu",
            ("darwin", "x86_64"): "x86_64-apple-darwin",
            ("darwin", "aarch64"): "aarch64-apple-darwin",
        }.get((system, arch))
        prefix = "fd-v"
    if target is None:
        raise ToolManagerError("no release asset for this platform")
    return f"{prefix}{version}-{target}.{extension}"


def _release_version(value: object) -> str:
    if not isinstance(value, str):
        raise ToolManagerError("release metadata has no valid version")
    match = _VERSION_RE.fullmatch(value.strip())
    if match is None:
        raise ToolManagerError("release metadata has no valid version")
    return match.group(1)


def _trusted_https_url(url: str, allowed_hosts: frozenset[str]) -> bool:
    parsed = urllib.parse.urlparse(url)
    return (
        parsed.scheme.casefold() == "https"
        and (parsed.hostname or "").casefold() in allowed_hosts
    )


def _require_https_response(
    response: Any,
    requested_url: str,
    *,
    allowed_hosts: frozenset[str],
) -> None:
    final_url = requested_url
    getter = getattr(response, "geturl", None)
    if callable(getter):
        candidate = getter()
        if isinstance(candidate, str):
            final_url = candidate
    if not _trusted_https_url(final_url, allowed_hosts):
        raise ToolManagerError("official HTTPS response required")


def _check_cancelled(signal: ToolCancellationToken | None) -> None:
    if signal is not None and signal.is_cancelled():
        raise ToolManagerError("Tool installation cancelled")


def _check_download_active(
    signal: ToolCancellationToken | None,
    deadline: float,
) -> None:
    _check_cancelled(signal)
    if time.monotonic() >= deadline:
        raise ToolManagerError("binary download timed out")


def _valid_executable(path: Path, name: str, *, expected_version: str | None = None) -> bool:
    if not path.is_file():
        return False
    result = _version_output(path)
    if result is None:
        return False
    returncode, output = result
    if returncode != 0:
        return False
    text = output.decode("utf-8", errors="ignore").casefold()
    if name == "rg":
        identity = "ripgrep" in text or re.search(r"(?:^|\s)rg(?:\s|$)", text) is not None
    else:
        identity = re.search(r"(?:^|\s)fd(?:\s|$)", text) is not None
    if not identity:
        return False
    versions = tuple(_VERSION_TOKEN_RE.finditer(text))
    if not versions:
        return False
    if expected_version is None:
        return True
    return any(
        match.group(1).casefold() == expected_version.casefold()
        for match in versions
    )


def _version_output(path: Path) -> tuple[int, bytes] | None:
    retained = bytearray()
    try:
        process = subprocess.Popen(
            [str(path), "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except OSError:
        return None
    stdout = process.stdout
    assert stdout is not None

    def drain() -> None:
        while len(retained) < MAX_VERSION_OUTPUT_BYTES:
            chunk = stdout.read(MAX_VERSION_OUTPUT_BYTES - len(retained))
            if not chunk:
                break
            remaining = MAX_VERSION_OUTPUT_BYTES - len(retained)
            retained.extend(chunk[:remaining])
        stdout.close()

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    try:
        returncode = process.wait(timeout=VERSION_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        _kill_sync_process_tree(process)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            stdout.close()
            return None
        returncode = process.returncode
    reader.join(timeout=5)
    if reader.is_alive() or returncode is None:
        _kill_sync_process_tree(process)
        return None
    return returncode, bytes(retained)


def _kill_sync_process_tree(process: subprocess.Popen[bytes]) -> None:
    if sys.platform != "win32":
        try:
            os.killpg(process.pid, process_signal.SIGKILL)
        except ProcessLookupError:
            return
        return
    try:
        completed = subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        completed = None
    if completed is None or completed.returncode != 0:
        with contextlib.suppress(OSError):
            process.kill()


def _extract_binary(archive: Path, *, archive_name: str, executable_name: str) -> bytes:
    if archive_name.endswith(".zip"):
        return _extract_zip_binary(archive, executable_name)
    return _extract_tar_binary(archive, executable_name)


def _extract_zip_binary(archive: Path, executable_name: str) -> bytes:
    candidates: list[zipfile.ZipInfo] = []
    names: set[str] = set()
    try:
        with zipfile.ZipFile(archive) as handle:
            for info in handle.infolist():
                _validate_archive_name(info.filename)
                normalized_name = info.filename.replace("\\", "/")
                if normalized_name in names:
                    raise ToolManagerError("archive contains duplicate entries")
                names.add(normalized_name)
                mode = (info.external_attr >> 16) & 0o170000
                if mode and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                    raise ToolManagerError("archive contains an unsupported file type")
                if mode and stat.S_ISDIR(mode) and not info.is_dir():
                    raise ToolManagerError("archive contains an invalid directory entry")
                if info.is_dir():
                    continue
                if PurePosixPath(info.filename).name == executable_name:
                    candidates.append(info)
            if len(candidates) != 1:
                raise ToolManagerError("archive does not contain one expected binary")
            info = candidates[0]
            if info.file_size > MAX_DOWNLOAD_BYTES:
                raise ToolManagerError("archive binary exceeds size limit")
            return handle.read(info)
    except ToolManagerError:
        raise
    except (OSError, zipfile.BadZipFile) as exc:
        raise ToolManagerError(f"archive extraction failed: {_phase_error(exc)}") from None


def _extract_tar_binary(archive: Path, executable_name: str) -> bytes:
    candidates: list[tarfile.TarInfo] = []
    names: set[str] = set()
    try:
        with tarfile.open(archive, mode="r:*") as handle:
            for info in handle.getmembers():
                _validate_archive_name(info.name)
                normalized_name = info.name.replace("\\", "/")
                if normalized_name in names:
                    raise ToolManagerError("archive contains duplicate entries")
                names.add(normalized_name)
                if info.issym() or info.islnk() or info.isdev() or info.isfifo():
                    raise ToolManagerError("archive contains an unsupported link or device")
                if info.isfile() and PurePosixPath(info.name).name == executable_name:
                    candidates.append(info)
            if len(candidates) != 1:
                raise ToolManagerError("archive does not contain one expected binary")
            info = candidates[0]
            if info.size > MAX_DOWNLOAD_BYTES:
                raise ToolManagerError("archive binary exceeds size limit")
            extracted = handle.extractfile(info)
            if extracted is None:
                raise ToolManagerError("archive binary cannot be read")
            return extracted.read(MAX_DOWNLOAD_BYTES + 1)
    except ToolManagerError:
        raise
    except (OSError, tarfile.TarError) as exc:
        raise ToolManagerError(f"archive extraction failed: {_phase_error(exc)}") from None


def _validate_archive_name(value: str) -> None:
    windows_path = PureWindowsPath(value)
    if (
        not value
        or PurePosixPath(value).is_absolute()
        or windows_path.anchor
        or windows_path.is_absolute()
    ):
        raise ToolManagerError("archive contains an absolute path")
    parts = PurePosixPath(value).parts
    if any(part in {"", ".", ".."} for part in parts) or ".." in PureWindowsPath(value).parts:
        raise ToolManagerError("archive contains a path traversal")


def _stale_lock(path: Path) -> bool:
    try:
        if time.time() - path.stat().st_mtime <= STALE_LOCK_SECONDS:
            return False
        text = path.read_text(encoding="ascii").strip()
        pid = int(text)
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        return False
    except (OSError, ValueError):
        return False


def _remove_lock(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _remove_path(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _offline(environ: Mapping[str, str]) -> bool:
    return environ.get("FORGE_OFFLINE", "").strip().casefold() in {"1", "true", "yes"}


def _positive_int(value: object) -> int | None:
    try:
        parsed = int(value) if isinstance(value, str) else value
    except ValueError:
        return None
    return parsed if isinstance(parsed, int) and parsed >= 0 else None


def _phase_error(exc: BaseException) -> str:
    if isinstance(exc, ToolManagerError):
        return str(exc)[:160]
    return exc.__class__.__name__


DEFAULT_TOOL_MANAGER = ToolManager()


__all__ = [
    "API_TIMEOUT_SECONDS",
    "DOWNLOAD_TIMEOUT_SECONDS",
    "LOCK_TIMEOUT_SECONDS",
    "MAX_DOWNLOAD_BYTES",
    "ManagedToolSpec",
    "DEFAULT_TOOL_MANAGER",
    "TOOL_SPECS",
    "ToolManager",
    "ToolManagerError",
]
