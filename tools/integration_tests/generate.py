# Copyright © 2026 Apple Inc.

"""
Generate the Swift integration tests.

    # normal use (needs python mlx and a GPU)
    python3 tools/integration_tests/generate.py

    # inspect the case list without running mlx
    python3 tools/integration_tests/generate.py --list

    # exercise the generator itself with a numpy backend (values are NOT valid
    # mlx values -- output goes to a temp directory)
    python3 tools/integration_tests/generate.py --self-test
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import check_api  # noqa: E402
import core  # noqa: E402
from cases import CASES  # noqa: E402
from modules import MODULE_CASES, MODULE_IMPORTS  # noqa: E402
from optimizers import (  # noqa: E402
    OPTIMIZER_CASES,
    OPTIMIZER_FUNCTION_CASES,
    OPTIMIZER_IMPORTS,
    SCHEDULE_CASES,
)

try:
    from cases import EXTRA_IMPORTS  # noqa: E402
except ImportError:  # pragma: no cover
    EXTRA_IMPORTS = {}

EXTRA_IMPORTS.update(MODULE_IMPORTS)
EXTRA_IMPORTS.update(OPTIMIZER_IMPORTS)

DEFAULT_OUTPUT = core.ROOT / "Tests/MLXTests/Integration/Generated"


def grouped() -> dict[str, list[core.Case | core.ModuleCase]]:
    files: dict[str, list[core.Case | core.ModuleCase]] = collections.OrderedDict()
    for case in [
        *CASES,
        *OPTIMIZER_FUNCTION_CASES,
        *MODULE_CASES,
        *OPTIMIZER_CASES,
        *SCHEDULE_CASES,
    ]:
        files.setdefault(case.file, []).append(case)
    return files


RNG_FILE = "RandomInputs"


def check_rng_coverage(files: dict[str, list[core.Case]]) -> list[str]:
    """warn when an op case draws from a generator/dtype the RNG file misses"""
    def specs(case) -> list:
        if isinstance(case, core.OptimizerCase):
            return list(case.parameters.values())
        if isinstance(case, core.ScheduleCase):
            return []
        return list(case.inputs.values())

    covered = {
        core.coverage_key(spec) for case in files.get(RNG_FILE, []) for spec in specs(case)
    }
    warnings = []
    seen = set()
    for file, cases in files.items():
        if file == RNG_FILE:
            continue
        for case in cases:
            for spec in specs(case):
                key = core.coverage_key(spec)
                if key is None or key in covered or key in seen:
                    continue
                seen.add(key)
                warnings.append(
                    f"{key[0]}(dtype={key[1]}) is used by {file}/{case.name} but has no "
                    f"case in {RNG_FILE}"
                )
    return warnings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="output directory")
    parser.add_argument("--only", action="append", help="only generate this file key (repeatable)")
    parser.add_argument("--list", action="store_true", help="list cases and exit")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run with the numpy backend and write to a temp directory (development only)",
    )
    parser.add_argument("--stdout", action="store_true", help="write to stdout instead of files")
    parser.add_argument(
        "--syntax-only",
        action="store_true",
        help="emit placeholder values without running mlx: checks that every case's "
        "Swift expression parses (writes to a temp directory)",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="skip the python API check that runs before generating",
    )
    parser.add_argument(
        "--no-format",
        action="store_true",
        help="skip running swift-format on the output (CI lints with it)",
    )
    args = parser.parse_args()

    files = grouped()
    if args.only:
        files = {k: v for k, v in files.items() if k in args.only}
        if not files:
            print(f"no cases for {args.only}", file=sys.stderr)
            return 1

    if args.list:
        for file, cases in files.items():
            print(f"{file} ({len(cases)} cases)")
            for case in cases:
                if isinstance(case, core.ModuleCase):
                    expression = f"{case.py} -> {case.call_py}"
                elif isinstance(case, core.OptimizerCase):
                    expression = f"{case.py} x{case.steps}"
                else:
                    expression = case.py
                print(f"    {case.name:36} seed={case.seed:<6} {expression}")
        print(f"\ntotal: {sum(len(c) for c in files.values())} cases in {len(files)} files")
        return 0

    for warning in check_rng_coverage(grouped()):
        print(f"warning: {warning}", file=sys.stderr)

    if not args.no_check:
        # catch stale/typo'd python arguments before spending time evaluating
        problems = check_api.check(
            [*CASES, *OPTIMIZER_FUNCTION_CASES],
            MODULE_CASES,
            optimizer_cases=OPTIMIZER_CASES,
            schedule_cases=SCHEDULE_CASES,
        )
        for problem in problems:
            print(f"python API: {problem}", file=sys.stderr)
        if problems:
            print(
                f"\n{len(problems)} case(s) do not match the installed python mlx API; "
                "fix them or pass --no-check",
                file=sys.stderr,
            )
            return 1

    if args.syntax_only:
        mx = None
        output = pathlib.Path(tempfile.mkdtemp(prefix="mlx-integration-syntax-"))
        print("syntax-only: placeholder values, not valid mlx values", file=sys.stderr)
    elif args.self_test:
        import fake_mlx

        mx = fake_mlx.mx
        output = pathlib.Path(tempfile.mkdtemp(prefix="mlx-integration-selftest-"))
        print(f"self-test: numpy backend, values are not valid mlx values", file=sys.stderr)
    else:
        import mlx.core as mx  # type: ignore

        output = pathlib.Path(args.output)

    failures: list[str] = []
    skipped: list[str] = []
    written: list[pathlib.Path] = []

    for file, cases in files.items():
        try:
            text, file_skipped = core.emit_file(
                mx, file, cases, skip_errors=args.self_test,
                extra_imports=EXTRA_IMPORTS.get(file, ()),
                evaluate=not args.syntax_only,
            )
            skipped += file_skipped
        except Exception as e:  # noqa: BLE001 -- report and continue
            failures.append(f"{file}: {type(e).__name__}: {e}")
            continue
        if args.stdout:
            print(text)
        else:
            output.mkdir(parents=True, exist_ok=True)
            # `Generated` prefix matches the suite type name and keeps these
            # from colliding with hand written tests (FFTTests, LinalgTests,
            # QuantizationTests all exist in Tests/MLXTests)
            path = output / f"Generated{file}Tests.swift"
            path.write_text(text)
            written.append(path)

    if written and not args.no_format:
        swift_format = shutil.which("swift-format")
        if swift_format is None:
            print(
                "swift-format not found: output is not formatted (CI style check will fail)",
                file=sys.stderr,
            )
        else:
            result = subprocess.run(
                [
                    swift_format, "format", "--in-place",
                    "--configuration", str(core.ROOT / ".swift-format"),
                    *[str(p) for p in written],
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                failures.append(f"swift-format: {result.stderr.strip()}")

    for path in written:
        print(f"wrote {path} ({len(path.read_text().splitlines())} lines)")
    for skip in skipped:
        print(f"skipped {skip}", file=sys.stderr)
    for failure in failures:
        print(f"FAILED {failure}", file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
