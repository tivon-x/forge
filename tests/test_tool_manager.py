from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import tarfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

import forge_coding.tools.tool_manager as tool_manager_module
from forge_agent.tools import ToolCancellationToken
from forge_coding.tools.tool_manager import (
    TOOL_SPECS,
    ToolManager,
    ToolManagerError,
    _asset_name,
    _extract_binary,
    _release_version,
    _valid_executable,
    _version_output,
)


def _fake_executable(path: Path, output: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        path = path.with_name(path.name + ".cmd")
        path.write_text(f"@echo off\necho {output}\n", encoding="utf-8")
    else:
        path.write_text(f"#!/bin/sh\nprintf '%s\\n' '{output}'\n", encoding="utf-8")
        path.chmod(0o755)
    return path


def test_private_cache_precedes_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    private = home / ".forge" / "bin" / "rg"
    private.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        private = private.with_suffix(".exe")
        private.write_bytes(b"private")
    else:
        _fake_executable(private, "ripgrep 14.0.0")
    path_dir = tmp_path / "path"
    path_dir.mkdir()
    _fake_executable(path_dir / "rg", "ripgrep 13.0.0")

    if os.name == "nt":
        monkeypatch.setattr(
            tool_manager_module,
            "_valid_executable",
            lambda path, name, expected_version=None: path == private and path.exists(),
        )
    manager = ToolManager(
        home=home,
        system="windows" if os.name == "nt" else "linux",
        environ={"PATH": str(path_dir)},
    )
    assert manager.ensure_tool("rg") == private


def test_cached_resolution_skips_revalidation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private = tmp_path / "home" / ".forge" / "bin" / "rg"
    private.parent.mkdir(parents=True)
    private.write_bytes(b"binary")
    calls = 0

    def valid(path: Path, name: str, *, expected_version: str | None = None) -> bool:
        nonlocal calls
        del name, expected_version
        calls += 1
        return path.exists()

    monkeypatch.setattr(tool_manager_module, "_valid_executable", valid)
    manager = ToolManager(home=tmp_path / "home", system="linux", environ={"PATH": ""})
    assert manager.ensure_tool("rg") == private
    assert manager.ensure_tool("rg") == private
    assert calls == 1


def test_path_fallback_prefers_fd_over_fdfind(tmp_path: Path) -> None:
    path_dir = tmp_path / "path"
    path_dir.mkdir()
    _fake_executable(path_dir / "fd", "fd 10.0.0")
    _fake_executable(path_dir / "fdfind", "fd 10.0.0")
    manager = ToolManager(
        home=tmp_path / "home",
        system="windows" if os.name == "nt" else "linux",
        environ={"PATH": str(path_dir)},
    )
    expected = path_dir / "fd"
    if os.name == "nt":
        expected = expected.with_name("fd.CMD")
    assert manager.ensure_tool("fd") == expected


def test_offline_mode_does_not_open_network(tmp_path: Path) -> None:
    calls = 0

    def urlopen(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("network must not be used in offline mode")

    manager = ToolManager(
        home=tmp_path / "home",
        system="linux",
        environ={"FORGE_OFFLINE": "1", "PATH": ""},
        urlopen=urlopen,
    )
    with pytest.raises(ToolManagerError, match="FORGE_OFFLINE"):
        manager.ensure_tool("rg")
    assert calls == 0


def test_offline_failure_is_bounded_and_actionable(tmp_path: Path) -> None:
    manager = ToolManager(
        home=tmp_path / "home",
        system="linux",
        environ={"FORGE_OFFLINE": "true", "PATH": ""},
    )
    with pytest.raises(ToolManagerError, match="FORGE_OFFLINE") as error:
        manager.ensure_tool("rg")
    assert "http" not in str(error.value).casefold()


def test_safe_archive_rejects_traversal_and_duplicate_binary(tmp_path: Path) -> None:
    traversal = tmp_path / "bad.zip"
    with zipfile.ZipFile(traversal, "w") as archive:
        archive.writestr("../rg", b"bad")
    with pytest.raises(ToolManagerError, match="traversal"):
        _extract_binary(traversal, archive_name="x.zip", executable_name="rg")

    duplicate = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(duplicate, "w") as archive:
        archive.writestr("one/rg", b"one")
        archive.writestr("two/rg", b"two")
    with pytest.raises(ToolManagerError, match="one expected binary"):
        _extract_binary(duplicate, archive_name="x.zip", executable_name="rg")

    device = tmp_path / "device.zip"
    info = zipfile.ZipInfo("rg")
    info.external_attr = (stat.S_IFIFO | 0o644) << 16
    with zipfile.ZipFile(device, "w") as archive:
        archive.writestr(info, b"not a binary")
    with pytest.raises(ToolManagerError, match="unsupported file type"):
        _extract_binary(device, archive_name="x.zip", executable_name="rg")


@pytest.mark.parametrize(
    ("name", "member_type"),
    [
        ("/rg", tarfile.REGTYPE),
        ("../rg", tarfile.REGTYPE),
        ("rg", tarfile.SYMTYPE),
        ("rg", tarfile.LNKTYPE),
        ("rg", tarfile.FIFOTYPE),
        ("rg", tarfile.CHRTYPE),
        ("rg", tarfile.BLKTYPE),
    ],
)
def test_safe_tar_rejects_traversal_links_and_devices(
    tmp_path: Path, name: str, member_type: bytes
) -> None:
    archive_path = tmp_path / "bad.tar.gz"
    with tarfile.open(archive_path, mode="w:gz") as archive:
        info = tarfile.TarInfo(name)
        info.type = member_type
        info.mode = 0o755
        if member_type == tarfile.REGTYPE:
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))
        else:
            info.linkname = "target"
            archive.addfile(info)
    with pytest.raises(ToolManagerError):
        _extract_binary(archive_path, archive_name="x.tar.gz", executable_name="rg")


class _Response:
    def __init__(
        self,
        payload: bytes,
        *,
        content_length: int | None = None,
        final_url: str | None = None,
    ) -> None:
        self.payload = payload
        self.headers = {"Content-Length": str(content_length)} if content_length is not None else {}
        self.final_url = final_url

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def geturl(self) -> str | None:
        return self.final_url

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            return self.payload
        result, self.payload = self.payload[:size], self.payload[size:]
        return result


def test_download_requires_digest_and_installs_verified_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    system = "windows" if os.name == "nt" else "linux"
    binary_name = "rg.exe" if system == "windows" else "rg"
    binary = b"#!/bin/sh\nprintf 'ripgrep 1.2.3\\n'\n"
    archive_buffer = io.BytesIO()
    if system == "windows":
        with zipfile.ZipFile(archive_buffer, "w") as archive_zip:
            archive_zip.writestr(f"ripgrep-1.2.3/{binary_name}", binary)
    else:
        with tarfile.open(fileobj=archive_buffer, mode="w:gz") as archive_tar:
            info = tarfile.TarInfo(f"ripgrep-1.2.3/{binary_name}")
            info.size = len(binary)
            info.mode = 0o755
            archive_tar.addfile(info, io.BytesIO(binary))
    archive_payload = archive_buffer.getvalue()
    extension = "zip" if system == "windows" else "tar.gz"
    asset_name = _asset_name(TOOL_SPECS["rg"], "1.2.3", system, "x86_64", extension)
    digest = hashlib.sha256(archive_payload).hexdigest()
    release = {
        "tag_name": "v1.2.3",
        "assets": [
            {
                "name": asset_name,
                "digest": f"sha256:{digest}",
                "browser_download_url": "https://github.com/asset",
                "size": len(archive_payload),
            }
        ],
    }

    def urlopen(request: Any, *, timeout: float) -> _Response:
        del timeout
        if "api.github.com" in request.full_url:
            return _Response(json.dumps(release).encode())
        return _Response(archive_payload, content_length=len(archive_payload))

    manager = ToolManager(
        home=tmp_path / "home",
        system=system,
        machine="x86_64",
        environ={"PATH": ""},
        urlopen=urlopen,
    )
    if system == "windows":
        monkeypatch.setattr(
            tool_manager_module,
            "_valid_executable",
            lambda path, name, expected_version=None: path.exists(),
        )
    installed = manager.ensure_tool("rg")
    assert installed.exists()
    assert installed.read_text(encoding="utf-8") == binary.decode()


def test_download_rejects_missing_mismatched_digest_and_sizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"archive"
    system = "linux"
    asset_name = _asset_name(TOOL_SPECS["rg"], "1.2.3", system, "x86_64", "tar.gz")

    def run_case(
        release_asset: dict[str, object],
        *,
        content_length: int | None = len(payload),
    ) -> str:
        release = {"tag_name": "v1.2.3", "assets": [release_asset]}

        def urlopen(request: Any, *, timeout: float) -> _Response:
            del timeout
            if "api.github.com" in request.full_url:
                return _Response(json.dumps(release).encode())
            return _Response(payload, content_length=content_length)

        manager = ToolManager(
            home=tmp_path / f"home-{time.time_ns()}",
            system=system,
            machine="x86_64",
            environ={"PATH": ""},
            urlopen=urlopen,
        )
        with pytest.raises(ToolManagerError) as error:
            manager.ensure_tool("rg")
        bin_dir = manager.bin_dir
        assert not list(bin_dir.glob("forge-tool-*"))
        assert not list(bin_dir.glob(".*.tmp"))
        return str(error.value)

    base = {
        "name": asset_name,
        "browser_download_url": "https://github.com/asset",
        "size": len(payload),
    }
    missing = run_case(dict(base))
    assert "digest" in missing
    mismatch = run_case({**base, "digest": "sha256:" + "0" * 64})
    assert "digest mismatch" in mismatch
    content_length_error = run_case(
        {**base, "digest": "sha256:" + hashlib.sha256(payload).hexdigest()},
        content_length=len(payload) + 1,
    )
    assert "Content-Length" in content_length_error
    metadata_error = run_case(
        {
            **base,
            "digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
            "size": len(payload) + 1,
        }
    )
    assert "release metadata" in metadata_error
    monkeypatch.setattr(tool_manager_module, "MAX_DOWNLOAD_BYTES", 2)
    too_large = run_case(
        {
            **base,
            "digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
        }
    )
    assert "256 MiB" in too_large


def test_download_timeout_and_invalid_release_inputs_are_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def timeout_urlopen(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise TimeoutError("https://secret.invalid")

    timeout_manager = ToolManager(
        home=tmp_path / "timeout",
        system="linux",
        machine="x86_64",
        environ={"PATH": ""},
        urlopen=timeout_urlopen,
    )
    with pytest.raises(
        ToolManagerError, match="release metadata request failed: TimeoutError"
    ) as error:
        timeout_manager.ensure_tool("rg")
    assert "secret.invalid" not in str(error.value)
    assert calls == 1

    with pytest.raises(ToolManagerError, match="Unsupported managed tool"):
        ToolManager(home=tmp_path).ensure_tool("nope")

    def wrong_asset_urlopen(request: Any, *, timeout: float) -> _Response:
        del timeout
        release = {"tag_name": "v1.2.3", "assets": [{"name": "wrong.zip"}]}
        if "api.github.com" in request.full_url:
            return _Response(json.dumps(release).encode())
        raise AssertionError("asset must not be opened")

    wrong_asset = ToolManager(
        home=tmp_path / "wrong-asset",
        system="linux",
        machine="x86_64",
        environ={"PATH": ""},
        urlopen=wrong_asset_urlopen,
    )
    with pytest.raises(ToolManagerError, match="unique platform asset"):
        wrong_asset.ensure_tool("rg")

    unsupported = ToolManager(
        home=tmp_path / "unsupported",
        system="linux",
        machine="mips",
        environ={"PATH": ""},
        urlopen=wrong_asset_urlopen,
    )
    with pytest.raises(ToolManagerError, match="CPU architecture"):
        unsupported.ensure_tool("rg")

    archive_buffer = io.BytesIO()
    wrong_binary = b"not-ripgrep"
    with tarfile.open(fileobj=archive_buffer, mode="w:gz") as archive:
        info = tarfile.TarInfo("ripgrep-1.2.3/rg")
        info.size = len(wrong_binary)
        archive.addfile(info, io.BytesIO(wrong_binary))
    archive_payload = archive_buffer.getvalue()
    asset_name = _asset_name(TOOL_SPECS["rg"], "1.2.3", "linux", "x86_64", "tar.gz")
    release = {
        "tag_name": "v1.2.3",
        "assets": [
            {
                "name": asset_name,
                "digest": "sha256:" + hashlib.sha256(archive_payload).hexdigest(),
                "browser_download_url": "https://github.com/asset",
                "size": len(archive_payload),
            }
        ],
    }

    def wrong_version_urlopen(request: Any, *, timeout: float) -> _Response:
        del timeout
        if "api.github.com" in request.full_url:
            return _Response(json.dumps(release).encode())
        return _Response(archive_payload, content_length=len(archive_payload))

    monkeypatch.setattr(
        tool_manager_module,
        "_valid_executable",
        lambda path, name, expected_version=None: (
            path.exists()
            and (expected_version is None or expected_version.encode() in path.read_bytes())
        ),
    )
    wrong_version = ToolManager(
        home=tmp_path / "wrong-version",
        system="linux",
        machine="x86_64",
        environ={"PATH": ""},
        urlopen=wrong_version_urlopen,
    )
    with pytest.raises(ToolManagerError, match="version validation"):
        wrong_version.ensure_tool("rg")


def test_concurrent_managers_install_only_once_and_leave_no_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    count = 0
    count_lock = threading.Lock()

    class CountingManager(ToolManager):
        def _download_and_install(
            self,
            spec: tool_manager_module.ManagedToolSpec,
            destination: Path,
            *,
            signal: ToolCancellationToken | None,
        ) -> Path:
            del spec, signal
            nonlocal count
            with count_lock:
                count += 1
            time.sleep(0.1)
            destination.write_bytes(b"managed")
            return destination

    monkeypatch.setattr(
        tool_manager_module,
        "_valid_executable",
        lambda path, name, expected_version=None: path.is_file(),
    )
    home = tmp_path / "shared-home"
    managers = [
        CountingManager(home=home, system="linux", environ={"PATH": ""}),
        CountingManager(home=home, system="linux", environ={"PATH": ""}),
    ]
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda manager: manager.ensure_tool("rg"), managers))
    assert count == 1
    assert results[0] == results[1]
    assert results[0].read_bytes() == b"managed"
    assert not (managers[0].bin_dir / ".rg.lock").exists()
    assert not list(managers[0].bin_dir.glob("forge-tool-*"))
    assert not list(managers[0].bin_dir.glob(".*.tmp"))


class _MutableSignal:
    def __init__(self) -> None:
        self.cancelled = False

    def is_cancelled(self) -> bool:
        return self.cancelled


def test_lock_wait_honors_cancellation_and_stale_dead_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = ToolManager(home=tmp_path / "home", system="linux", environ={"PATH": ""})
    manager.bin_dir.mkdir(parents=True)
    lock = manager.bin_dir / ".rg.lock"
    lock.write_text(f"{os.getpid()}\n", encoding="ascii")
    signal = _MutableSignal()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(manager._acquire_lock, lock, signal=signal)
        time.sleep(0.15)
        signal.cancelled = True
        with pytest.raises(ToolManagerError, match="cancelled"):
            future.result(timeout=2)
    lock.unlink()

    lock.write_text("999999999\n", encoding="ascii")
    old = time.time() - tool_manager_module.STALE_LOCK_SECONDS - 1
    os.utime(lock, (old, old))

    def dead_process(_pid: int, _sig: int) -> None:
        raise ProcessLookupError

    monkeypatch.setattr(tool_manager_module.os, "kill", dead_process)  # type: ignore[attr-defined]
    manager._acquire_lock(lock, signal=None)
    assert lock.exists()
    tool_manager_module._remove_lock(lock)


def test_lock_error_does_not_expose_private_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = ToolManager(home=tmp_path / "home", system="linux", environ={"PATH": ""})
    lock = manager.bin_dir / ".rg.lock"

    def fail_open(*_args: object, **_kwargs: object) -> int:
        raise PermissionError(f"private path {lock}")

    monkeypatch.setattr(tool_manager_module.os, "open", fail_open)  # type: ignore[attr-defined]
    with pytest.raises(ToolManagerError) as error:
        manager._acquire_lock(lock, signal=None)
    assert "PermissionError" in str(error.value)
    assert str(lock) not in str(error.value)


def test_lock_write_failure_removes_owned_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = ToolManager(home=tmp_path / "home", system="linux", environ={"PATH": ""})
    manager.bin_dir.mkdir(parents=True)
    lock = manager.bin_dir / ".rg.lock"

    def fail_write(_descriptor: int, _data: bytes) -> int:
        raise OSError("write failed")

    monkeypatch.setattr(tool_manager_module.os, "write", fail_write)
    with pytest.raises(ToolManagerError, match="initialize"):
        manager._acquire_lock(lock, signal=None)
    assert not lock.exists()


def test_download_rejects_https_redirect_downgrade(tmp_path: Path) -> None:
    release = {
        "tag_name": "v1.2.3",
        "assets": [
            {
                "name": "ripgrep-1.2.3-x86_64-unknown-linux-musl.tar.gz",
                "digest": "sha256:" + "0" * 64,
                "browser_download_url": "https://github.com/asset",
                "size": 1,
            }
        ],
    }

    def urlopen(request: Any, *, timeout: float) -> _Response:
        del timeout
        if "api.github.com" in request.full_url:
            return _Response(
                json.dumps(release).encode(), final_url="https://api.github.com/release"
            )
        return _Response(b"x", content_length=1, final_url="http://redirect.invalid/asset")

    manager = ToolManager(
        home=tmp_path / "home",
        system="linux",
        machine="x86_64",
        environ={"PATH": ""},
        urlopen=urlopen,
    )
    with pytest.raises(ToolManagerError, match="HTTPS"):
        manager.ensure_tool("rg")
    assert not (tmp_path / "home" / ".forge" / "bin" / "rg").exists()


def test_download_rejects_https_redirect_to_untrusted_host(tmp_path: Path) -> None:
    manager = ToolManager(
        home=tmp_path / "home",
        system="linux",
        machine="x86_64",
        environ={"PATH": ""},
        urlopen=lambda *_args, **_kwargs: _Response(
            b"x", content_length=1, final_url="https://evil.invalid/asset"
        ),
    )
    manager.bin_dir.mkdir(parents=True)

    with pytest.raises(ToolManagerError, match="official HTTPS"):
        manager._download(
            "https://github.com/asset",
            {"size": 1},
            signal=None,
        )
    assert not list(manager.bin_dir.glob("forge-tool-*"))


def test_download_has_total_deadline_and_honors_mid_read_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [0.0]

    class SlowResponse(_Response):
        def read(self, size: int = -1) -> bytes:
            clock[0] += tool_manager_module.DOWNLOAD_TIMEOUT_SECONDS + 1
            return super().read(size)

    monkeypatch.setattr(tool_manager_module.time, "monotonic", lambda: clock[0])
    manager = ToolManager(
        home=tmp_path / "deadline",
        system="linux",
        machine="x86_64",
        environ={"PATH": ""},
        urlopen=lambda *_args, **_kwargs: SlowResponse(
            b"x", content_length=1, final_url="https://github.com/asset"
        ),
    )
    manager.bin_dir.mkdir(parents=True)
    with pytest.raises(ToolManagerError, match="timed out"):
        manager._download("https://github.com/asset", {"size": 1}, signal=None)
    assert not list(manager.bin_dir.glob("forge-tool-*"))

    signal = _MutableSignal()

    class CancellingResponse(_Response):
        def read(self, size: int = -1) -> bytes:
            signal.cancelled = True
            return super().read(size)

    clock[0] = 0.0
    cancelled = ToolManager(
        home=tmp_path / "cancelled",
        system="linux",
        machine="x86_64",
        environ={"PATH": ""},
        urlopen=lambda *_args, **_kwargs: CancellingResponse(
            b"x", content_length=1, final_url="https://github.com/asset"
        ),
    )
    cancelled.bin_dir.mkdir(parents=True)
    with pytest.raises(ToolManagerError, match="cancelled"):
        cancelled._download("https://github.com/asset", {"size": 1}, signal=signal)
    assert not list(cancelled.bin_dir.glob("forge-tool-*"))


def test_release_metadata_request_has_total_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [0.0]

    class SlowResponse(_Response):
        def read(self, size: int = -1) -> bytes:
            clock[0] += tool_manager_module.API_TIMEOUT_SECONDS + 1
            return super().read(size)

    monkeypatch.setattr(tool_manager_module.time, "monotonic", lambda: clock[0])
    manager = ToolManager(
        home=tmp_path / "home",
        system="linux",
        machine="x86_64",
        environ={"PATH": ""},
        urlopen=lambda *_args, **_kwargs: SlowResponse(
            b"{}", final_url="https://api.github.com/release"
        ),
    )

    with pytest.raises(ToolManagerError, match="metadata request timed out"):
        manager._latest_release(TOOL_SPECS["rg"])


def test_asset_names_are_fixed_by_platform() -> None:
    assert (
        _asset_name(TOOL_SPECS["rg"], "14.1.0", "windows", "x86_64", "zip")
        == "ripgrep-14.1.0-x86_64-pc-windows-msvc.zip"
    )
    assert (
        _asset_name(TOOL_SPECS["fd"], "10.2.0", "linux", "aarch64", "tar.gz")
        == "fd-v10.2.0-aarch64-unknown-linux-gnu.tar.gz"
    )
    assert _release_version("v10.2.0") == "10.2.0"
    with pytest.raises(ToolManagerError):
        _release_version("release-v10.2.0")


def test_version_validation_does_not_accept_overlapping_numeric_substrings(
    tmp_path: Path,
) -> None:
    executable = _fake_executable(tmp_path / "rg", "ripgrep 11.2.30")
    assert _valid_executable(executable, "rg", expected_version="1.2.3") is False
    nameless_version = _fake_executable(tmp_path / "fd", "fd")
    assert _valid_executable(nameless_version, "fd") is False


def test_version_probe_retains_only_bounded_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = io.BytesIO(b"ripgrep 1.2.3\n" + b"x" * 100_000)
            self.returncode = 0
            self.pid = 1

        def wait(self, *, timeout: float) -> int:
            del timeout
            return 0

    monkeypatch.setattr(
        tool_manager_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: FakeProcess(),
    )
    result = _version_output(tmp_path / "rg")
    assert result is not None
    returncode, output = result
    assert returncode == 0
    assert output.startswith(b"ripgrep 1.2.3")
    assert len(output) == tool_manager_module.MAX_VERSION_OUTPUT_BYTES
