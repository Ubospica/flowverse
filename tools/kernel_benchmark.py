"""Run a task's existing evaluator locally or through an existing KCoral service.

This is a standalone command, usable from any flow's task or evaluator receipt.
KCoral owns the remote protocol, uploads, GPU scheduling and downloads.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import NoReturn


class Terminated(Exception):
    """The local runner was asked to stop by its owning flow."""


def terminate(_signum: int, _frame: object) -> NoReturn:
    raise Terminated


def relative_path(value: str) -> str:
    """Accept an artifact inside the benchmark bundle."""
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise argparse.ArgumentTypeError("artifact paths must stay inside the bundle")
    return str(path)


def positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("timeout must be positive")
    return number


def copy_artifact(source: Path, destination: Path) -> None:
    """Copy regular files and directories, refusing links and special files."""
    if source.is_symlink():
        raise ValueError(f"artifact is a symbolic link: {source}")
    if source.is_dir():
        destination.mkdir(parents=True)
        for child in source.iterdir():
            copy_artifact(child, destination / child.name)
    elif source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    else:
        raise ValueError(f"artifact is missing or not a regular file: {source}")


def local(args: argparse.Namespace, command: list[str]) -> int:
    """Run in the source bundle, then keep requested artifacts even after failure."""
    with subprocess.Popen(
        command, cwd=args.bundle, stdin=subprocess.DEVNULL, start_new_session=True
    ) as process:
        previous = signal.signal(signal.SIGTERM, terminate)
        try:
            code = process.wait(timeout=args.timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            code = 124 if isinstance(error, subprocess.TimeoutExpired) else 130
            print("benchmark interrupted or timed out", file=sys.stderr)
        except Terminated:
            code = 143
        finally:
            # Also reap children an evaluator left behind on an ordinary exit.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            signal.signal(signal.SIGTERM, previous)
    if args.out:
        args.out.mkdir()
        for name in args.fetch:
            source = args.bundle / name
            # A link in any ancestor is a link too; do not copy outside the bundle.
            for parent in [source, *source.parents]:
                if parent == args.bundle:
                    break
                if parent.is_symlink():
                    raise ValueError(
                        f"artifact path contains a symbolic link: {parent}"
                    )
            copy_artifact(source, args.out / args.bundle.name / name)
    return code if code >= 0 else 128 - code


def remote(args: argparse.Namespace, command: list[str]) -> NoReturn:
    """Replace this process with the official client; never retry a benchmark here."""
    executable = shutil.which("kcoral")
    if executable is None:
        raise ValueError("kcoral is not installed; install the KCoral client on PATH")
    if not (args.url or os.environ.get("KCORAL_URL", "").strip()):
        raise ValueError("the kcoral backend requires --url or KCORAL_URL")
    argv = [
        executable,
        "run",
        "shell",
        "--timeout",
        str(args.timeout),
        "--send",
        str(args.bundle),
    ]
    if args.url:
        argv.extend(["--url", args.url])
    if args.out:
        argv.extend(["--out", str(args.out)])
    for name in args.fetch:
        argv.extend(["--fetch", f"{args.bundle.name}/{name}"])
    # The uploaded directory keeps its name. Positional arguments preserve spaces,
    # shell metacharacters, and evaluator arguments without evaluating them as code.
    argv.extend(
        [
            "--",
            "sh",
            "-c",
            'cd "./$1" && shift && exec "$@"',
            "kernel-benchmark",
            args.bundle.name,
            *command,
        ]
    )
    os.execv(executable, argv)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--backend",
        choices=("local", "kcoral"),
        default=os.environ.get("HMZ_BENCHMARK_BACKEND", "local"),
    )
    parser.add_argument(
        "--bundle",
        required=True,
        type=Path,
        help="directory containing the evaluator, kernel and inputs",
    )
    parser.add_argument(
        "--url", help="existing KCoral server or Router; else KCORAL_URL"
    )
    parser.add_argument("--timeout", type=positive, default=300)
    parser.add_argument(
        "--fetch",
        action="append",
        type=relative_path,
        default=[],
        help="artifact relative to the bundle; repeatable",
    )
    parser.add_argument("--out", type=Path, help="new local artifact directory")
    raw = list(sys.argv[1:] if argv is None else argv)
    if "--" not in raw:
        parser.error("separate the evaluator command with --")
    at = raw.index("--")
    args = parser.parse_args(raw[:at])
    command = raw[at + 1 :]
    if not command:
        parser.error("an evaluator command is required after --")
    if args.backend not in ("local", "kcoral"):
        parser.error("HMZ_BENCHMARK_BACKEND must be local or kcoral")
    if args.bundle.is_symlink():
        parser.error("the bundle must not be a symbolic link")
    args.bundle = args.bundle.resolve()
    if not args.bundle.is_dir() or not args.bundle.name:
        parser.error("the bundle must be a named directory")
    if bool(args.fetch) != bool(args.out):
        parser.error("--fetch and --out must be supplied together")
    # A requested directory and its child would try to save the same file twice.
    for index, name in enumerate(args.fetch):
        for other in args.fetch[:index]:
            left, right = PurePosixPath(name), PurePosixPath(other)
            if left.is_relative_to(right) or right.is_relative_to(left):
                parser.error("artifact paths must not overlap")
    if args.out:
        args.out = args.out.absolute()
        if args.out.exists() or args.out.is_symlink() or not args.out.parent.is_dir():
            parser.error("--out must be new and its parent must exist")
        if args.out.resolve().is_relative_to(args.bundle):
            parser.error("--out must be outside the bundle")
    if args.backend == "local" and args.url:
        parser.error("--url is only used by the kcoral backend")
    try:
        return (
            remote(args, command) if args.backend == "kcoral" else local(args, command)
        )
    except (OSError, ValueError) as error:
        print(f"kernel benchmark: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
