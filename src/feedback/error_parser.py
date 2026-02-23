"""Parse cargo check output into structured CompileError objects."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


ERROR_CATEGORIES: Dict[str, str] = {
    # type errors
    "E0308": "type_error", "E0309": "type_error", "E0310": "type_error",
    "E0312": "type_error", "E0606": "type_error", "E0607": "type_error",
    # borrow errors
    "E0502": "borrow_error", "E0503": "borrow_error", "E0504": "borrow_error",
    "E0505": "borrow_error", "E0506": "borrow_error", "E0507": "borrow_error",
    "E0508": "borrow_error", "E0510": "borrow_error",
    # lifetime errors
    "E0106": "lifetime_error", "E0107": "lifetime_error", "E0261": "lifetime_error",
    "E0262": "lifetime_error", "E0263": "lifetime_error", "E0597": "lifetime_error",
    # move errors
    "E0382": "move_error", "E0383": "move_error", "E0384": "move_error",
    # syntax / parse
    "E0001": "syntax_error", "E0004": "syntax_error",
    # undefined
    "E0425": "name_error", "E0412": "name_error", "E0422": "name_error",
}


@dataclass
class CompileError:
    error_code: str
    message: str
    span: Optional[str] = None
    help_text: Optional[str] = None
    category: str = "other"

    def __repr__(self) -> str:
        return f"CompileError(code={self.error_code!r}, category={self.category!r}, msg={self.message[:60]!r})"


class RustErrorParser:
    """Parse ``cargo check --message-format=json`` output."""

    def parse(self, cargo_output: str) -> List[CompileError]:
        """Parse *cargo_output* (JSON lines or plain text) → list of errors."""
        errors: List[CompileError] = []
        for line in cargo_output.splitlines():
            line = line.strip()
            if not line:
                continue
            # Try JSON line format first
            if line.startswith("{"):
                err = self._parse_json_line(line)
                if err:
                    errors.append(err)
            else:
                # Plain text fallback
                err = self._parse_text_line(line)
                if err:
                    errors.append(err)
        return errors

    def _parse_json_line(self, line: str) -> Optional[CompileError]:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return None

        if obj.get("reason") != "compiler-message":
            return None
        msg_obj = obj.get("message", {})
        level = msg_obj.get("level", "")
        if level not in ("error", "warning"):
            return None

        code_obj = msg_obj.get("code") or {}
        code = code_obj.get("code", "") if code_obj else ""
        message = msg_obj.get("message", "")
        category = ERROR_CATEGORIES.get(code, "other")

        # Span info
        spans = msg_obj.get("spans", [])
        span_str = None
        if spans:
            s = spans[0]
            span_str = f"{s.get('file_name', '?')}:{s.get('line_start', '?')}:{s.get('column_start', '?')}"

        # Help text from children
        help_text = None
        for child in msg_obj.get("children", []):
            if child.get("level") == "help":
                help_text = child.get("message", "")
                break

        return CompileError(
            error_code=code,
            message=message,
            span=span_str,
            help_text=help_text,
            category=category,
        )

    _TEXT_PATTERN = re.compile(r"error\[?(E\d+)?\]?:?\s*(.*)", re.IGNORECASE)
    _SPAN_PATTERN = re.compile(r"--> (.+:\d+:\d+)")

    def _parse_text_line(self, line: str) -> Optional[CompileError]:
        m = self._TEXT_PATTERN.match(line)
        if not m:
            return None
        code = m.group(1) or ""
        message = m.group(2).strip()
        category = ERROR_CATEGORIES.get(code, "other")
        return CompileError(error_code=code, message=message, category=category)

    def to_correction_prompt(self, errors: List[CompileError], rust_code: str) -> str:
        """Format errors and Rust code into a correction instruction."""
        if not errors:
            return ""
        error_lines = []
        for e in errors:
            loc = f" at {e.span}" if e.span else ""
            help_part = f" (hint: {e.help_text})" if e.help_text else ""
            error_lines.append(f"  [{e.error_code}] {e.category}{loc}: {e.message}{help_part}")
        errors_str = "\n".join(error_lines)
        return (
            f"The following Rust code has compilation errors.\n"
            f"Errors:\n{errors_str}\n\n"
            f"Code:\n```rust\n{rust_code}\n```\n\n"
            f"Fix the errors and output the corrected Rust code."
        )

    def __repr__(self) -> str:
        return "RustErrorParser()"
