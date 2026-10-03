"""Exercise the evaluator boundary without needing a GPU or an installed client."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

RUNNER = Path(__file__).parents[1] / "tools" / "kernel_benchmark.py"


def run(bundle, *args, env=None):
    return subprocess.run(
        [sys.executable, str(RUNNER), "--bundle", str(bundle), *args],
        text=True,
        capture_output=True,
        env={**os.environ, "HMZ_BENCHMARK_BACKEND": "local", **(env or {})},
        timeout=15,
        check=False,
    )


@pytest.fixture
def bundle(tmp_path):
    path = tmp_path / "experiment with spaces"
    path.mkdir()
    (path / "helper.py").write_text("VALUE = 42\n")
    (path / "evaluate.py").write_text(
        "import json, pathlib, sys\nfrom helper import VALUE\n"
        "pathlib.Path('results').mkdir(exist_ok=True)\n"
        "pathlib.Path('results/report.json').write_text(json.dumps({'value': VALUE, 'args': sys.argv[2:]}))\n"
        "print('measured', flush=True)\nprint('diagnostic', file=sys.stderr)\n"
        "sys.exit(int(sys.argv[1]))\n"
    )
    return path


@pytest.mark.parametrize("code", [0, 7])
def test_local_preserves_failure_streams_arguments_and_artifacts(
    bundle, tmp_path, code
):
    out = tmp_path / "artifacts"
    literal = "x; $(touch should-not-exist) ' quoted"
    result = run(
        bundle,
        "--fetch",
        "results",
        "--out",
        str(out),
        "--",
        sys.executable,
        "evaluate.py",
        str(code),
        literal,
    )
    assert result.returncode == code, result.stderr
    assert result.stdout == "measured\n"
    assert result.stderr == "diagnostic\n"
    assert json.loads((out / bundle.name / "results/report.json").read_text()) == {
        "value": 42,
        "args": [literal],
    }
    assert not (bundle / "should-not-exist").exists()


def test_remote_delegates_to_official_cli_with_literal_argv(bundle, tmp_path):
    client = tmp_path / "kcoral"
    receipt = tmp_path / "argv.json"
    client.write_text(
        f"#!{sys.executable}\nimport json, pathlib, sys\n"
        f"pathlib.Path({str(receipt)!r}).write_text(json.dumps(sys.argv[1:]))\n"
        "sys.exit(7)\n"
    )
    client.chmod(0o700)
    out = tmp_path / "artifacts"
    result = run(
        bundle,
        "--url",
        "http://worker:8000/prefix",
        "--fetch",
        "results",
        "--out",
        str(out),
        "--timeout",
        "17",
        "--",
        "python",
        "evaluate.py",
        "7",
        "$literal",
        env={"PATH": str(tmp_path), "HMZ_BENCHMARK_BACKEND": "kcoral"},
    )
    assert result.returncode == 7, result.stderr
    assert json.loads(receipt.read_text()) == [
        "run",
        "shell",
        "--timeout",
        "17",
        "--send",
        str(bundle),
        "--url",
        "http://worker:8000/prefix",
        "--out",
        str(out),
        "--fetch",
        f"{bundle.name}/results",
        "--",
        "sh",
        "-c",
        'cd "./$1" && shift && exec "$@"',
        "kernel-benchmark",
        bundle.name,
        "python",
        "evaluate.py",
        "7",
        "$literal",
    ]


def test_missing_client_never_runs_evaluator_locally(bundle):
    result = run(
        bundle,
        "--backend",
        "kcoral",
        "--",
        sys.executable,
        "evaluate.py",
        "0",
        env={"PATH": "", "KCORAL_URL": "http://worker"},
    )
    assert result.returncode != 0
    assert not (bundle / "results").exists()


def test_missing_url_fails_before_invoking_the_client(bundle, tmp_path):
    client = tmp_path / "kcoral"
    client.write_text(f"#!{sys.executable}\nraise RuntimeError('client was invoked')\n")
    client.chmod(0o700)
    result = run(
        bundle,
        "--backend",
        "kcoral",
        "--",
        "python",
        "evaluate.py",
        "0",
        env={"PATH": str(tmp_path), "KCORAL_URL": ""},
    )
    assert result.returncode == 1
    assert "requires --url or KCORAL_URL" in result.stderr
    assert "client was invoked" not in result.stderr
    assert not (bundle / "results").exists()


def test_timeout_kills_local_descendants(bundle):
    child = "import time,pathlib; time.sleep(2); pathlib.Path('escaped').touch()"
    script = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(10)"
    result = run(bundle, "--timeout", "1", "--", sys.executable, "-c", script)
    assert result.returncode == 124, result.stderr
    time.sleep(1.5)
    assert not (bundle / "escaped").exists()


def test_flow_termination_kills_the_local_evaluator(bundle):
    script = "import pathlib,time; pathlib.Path('ready').touch(); time.sleep(20)"
    with subprocess.Popen(
        [
            sys.executable,
            str(RUNNER),
            "--backend",
            "local",
            "--bundle",
            str(bundle),
            "--",
            sys.executable,
            "-c",
            script,
        ]
    ) as process:
        deadline = time.monotonic() + 5
        while not (bundle / "ready").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert (bundle / "ready").exists()
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=5) == 143


@pytest.mark.parametrize("code", [0, 7])
def test_real_kcoral_execution_and_artifact_round_trip(bundle, tmp_path, code):
    """Opt in with a real client on PATH and a CPU or GPU KCoral endpoint."""
    url = os.environ.get("KCORAL_TEST_URL")
    if not url:
        pytest.skip("set KCORAL_TEST_URL to exercise a real KCoral server")
    out = tmp_path / "remote-results"
    result = run(
        bundle,
        "--backend",
        "kcoral",
        "--url",
        url,
        "--fetch",
        "results",
        "--out",
        str(out),
        "--timeout",
        "10",
        "--",
        "python",
        "evaluate.py",
        str(code),
        "a literal $argument",
    )
    assert result.returncode == code, result.stderr
    assert "measured" in result.stdout
    assert "diagnostic" in result.stderr
    assert json.loads((out / bundle.name / "results/report.json").read_text()) == {
        "value": 42,
        "args": ["a literal $argument"],
    }
    assert not (bundle / "results").exists()


@pytest.mark.parametrize(
    "options",
    [
        ["--fetch", "../outside", "--out", "new"],
        ["--fetch", "/absolute", "--out", "new"],
        ["--fetch", "results"],
        ["--out", "new"],
        ["--fetch", "results", "--fetch", "results/x", "--out", "new"],
        ["--timeout", "0"],
        ["--backend", "other"],
        ["--url", "http://worker"],
    ],
)
def test_bad_configuration_fails_before_execution(bundle, options):
    result = run(bundle, *options, "--", sys.executable, "evaluate.py", "0")
    assert result.returncode == 2
    assert not (bundle / "results").exists()


def test_artifact_failure_cannot_be_reported_as_success(bundle, tmp_path):
    result = run(
        bundle,
        "--fetch",
        "missing",
        "--out",
        str(tmp_path / "new"),
        "--",
        sys.executable,
        "evaluate.py",
        "0",
    )
    assert result.returncode != 0
    assert "missing" in result.stderr


def test_existing_artifacts_are_not_replaced(bundle, tmp_path):
    out = tmp_path / "existing"
    out.mkdir()
    result = run(
        bundle,
        "--fetch",
        "results",
        "--out",
        str(out),
        "--",
        sys.executable,
        "evaluate.py",
        "0",
    )
    assert result.returncode == 2
    assert not (bundle / "results").exists()


def test_artifact_symlinks_are_refused(bundle, tmp_path):
    (bundle / "linked").symlink_to(tmp_path, target_is_directory=True)
    result = run(
        bundle,
        "--fetch",
        "linked",
        "--out",
        str(tmp_path / "new"),
        "--",
        sys.executable,
        "evaluate.py",
        "0",
    )
    assert result.returncode != 0
    assert "symbolic link" in result.stderr
