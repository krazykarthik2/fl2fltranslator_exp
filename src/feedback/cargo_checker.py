"""Run `cargo check` on generated Rust code and return (success, output)."""
from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Optional, Tuple


_CARGO_TOML = """\
[package]
name = "ncl_check"
version = "0.1.0"
edition = "2021"

[lib]
name = "ncl_check"
path = "src/lib.rs"
"""


class CargoChecker:
    """Validate Rust code snippets via ``cargo check``."""

    def __init__(self, workspace_dir: Optional[str] = None):
        self._owned_tmpdir = None
        if workspace_dir is None:
            self._owned_tmpdir = tempfile.TemporaryDirectory(prefix="ncl_cargo_")
            self._workspace = self._owned_tmpdir.name
        else:
            self._workspace = workspace_dir
        self._setup_workspace()

    # ------------------------------------------------------------------

    def _setup_workspace(self) -> None:
        src_dir = os.path.join(self._workspace, "src")
        os.makedirs(src_dir, exist_ok=True)
        cargo_toml = os.path.join(self._workspace, "Cargo.toml")
        if not os.path.exists(cargo_toml):
            with open(cargo_toml, "w", encoding="utf-8") as f:
                f.write(_CARGO_TOML)
        lib_rs = os.path.join(src_dir, "lib.rs")
        if not os.path.exists(lib_rs):
            with open(lib_rs, "w", encoding="utf-8") as f:
                f.write("// placeholder\n")

    def check(self, rust_code: str) -> Tuple[bool, str]:
        """Write *rust_code* to lib.rs and run ``cargo check``.

        Returns
        -------
        (success, output_text)
        """
        lib_rs = os.path.join(self._workspace, "src", "lib.rs")
        with open(lib_rs, "w", encoding="utf-8") as f:
            f.write(rust_code)

        try:
            result = subprocess.run(
                ["cargo", "check", "--message-format=json"],
                cwd=self._workspace,
                capture_output=True,
                text=True,
                timeout=60,
            )
            output = result.stdout + result.stderr
            success = result.returncode == 0
            return success, output
        except FileNotFoundError:
            return False, "cargo not found in PATH"
        except subprocess.TimeoutExpired:
            return False, "cargo check timed out"

    def cleanup(self) -> None:
        """Remove the temporary workspace (if we created it)."""
        if self._owned_tmpdir is not None:
            self._owned_tmpdir.cleanup()
            self._owned_tmpdir = None

    def __enter__(self) -> "CargoChecker":
        return self

    def __exit__(self, *_) -> None:
        self.cleanup()

    def __repr__(self) -> str:
        return f"CargoChecker(workspace={self._workspace!r})"
