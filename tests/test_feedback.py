"""Tests for the feedback loop: error parsing and cargo checker."""
import pytest
from src.feedback.error_parser import RustErrorParser, CompileError


class TestRustErrorParser:
    def setup_method(self):
        self.parser = RustErrorParser()

    def test_error_parser_empty(self):
        errors = self.parser.parse("")
        assert errors == []

    def test_error_parser_text_line(self):
        line = "error[E0308]: mismatched types"
        errors = self.parser.parse(line)
        assert len(errors) == 1
        assert errors[0].error_code == "E0308"
        assert "mismatched types" in errors[0].message
        assert errors[0].category == "type_error"

    def test_error_parser_borrow_error(self):
        line = "error[E0502]: cannot borrow `x` as mutable because it is also borrowed as immutable"
        errors = self.parser.parse(line)
        assert len(errors) == 1
        assert errors[0].category == "borrow_error"

    def test_error_parser_unknown_code(self):
        line = "error[E9999]: some unknown error"
        errors = self.parser.parse(line)
        assert len(errors) == 1
        assert errors[0].category == "other"

    def test_error_parser_json_line(self):
        import json
        msg = {
            "reason": "compiler-message",
            "message": {
                "level": "error",
                "code": {"code": "E0308"},
                "message": "mismatched types",
                "spans": [{"file_name": "src/lib.rs", "line_start": 3, "column_start": 5}],
                "children": [{"level": "help", "message": "consider using i32"}],
            }
        }
        errors = self.parser.parse(json.dumps(msg))
        assert len(errors) == 1
        assert errors[0].error_code == "E0308"
        assert errors[0].span == "src/lib.rs:3:5"
        assert errors[0].help_text == "consider using i32"

    def test_to_correction_prompt(self):
        errors = [
            CompileError(error_code="E0308", message="mismatched types",
                         span="src/lib.rs:3:5", category="type_error")
        ]
        rust_code = "fn foo() -> i32 { \"hello\" }"
        prompt = self.parser.to_correction_prompt(errors, rust_code)
        assert "E0308" in prompt
        assert "type_error" in prompt
        assert rust_code in prompt
        assert "Fix the errors" in prompt

    def test_to_correction_prompt_empty(self):
        prompt = self.parser.to_correction_prompt([], "fn foo() {}")
        assert prompt == ""

    def test_multiple_errors(self):
        lines = [
            "error[E0308]: mismatched types",
            "error[E0502]: borrow error",
            "warning: unused variable",
        ]
        errors = self.parser.parse("\n".join(lines))
        # Should find 2 errors (warning is not an error in our text parser)
        assert len(errors) >= 2


class TestCargoChecker:
    """Cargo checker tests — skipped if cargo is not installed."""

    @pytest.fixture(autouse=True)
    def skip_if_no_cargo(self):
        import shutil
        if shutil.which("cargo") is None:
            pytest.skip("cargo not found in PATH")

    def test_cargo_checker_valid(self):
        from src.feedback.cargo_checker import CargoChecker
        with CargoChecker() as checker:
            success, output = checker.check("pub fn add(a: i32, b: i32) -> i32 { a + b }")
            assert success, f"Expected success, got output:\n{output}"

    def test_cargo_checker_invalid(self):
        from src.feedback.cargo_checker import CargoChecker
        with CargoChecker() as checker:
            success, output = checker.check("fn broken() -> i32 { \"not an int\" }")
            assert not success

    def test_cargo_checker_context_manager(self):
        from src.feedback.cargo_checker import CargoChecker
        with CargoChecker() as checker:
            assert checker is not None
