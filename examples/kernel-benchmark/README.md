# Kernel benchmarks with KCoral

Choose where a kernel task's existing evaluator runs, independently of the agent
CLI and flow. `tools/kernel_benchmark.py` offers `local` and `kcoral` backends.
It works with `ralph_loop`, `flame_chase`, and a recorded evaluator command such
as `pfc evaluate -- ...`; it introduces no new agent or environment backend.

## Table of Contents

- [Install](#install)
- [Usage](#usage)
- [Server ownership](#server-ownership)
- [Contract](#contract)
- [Validation](#validation)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
- [License](#license)

## Install

The runner uses Python 3.12+ and the standard library. For remote execution,
install a [KCoral client](https://github.com/cmu-catalyst/kcoral) with
`kcoral run shell` support on the agent machine's `PATH`. No CUDA, PyTorch or
Triton is needed on that machine. `kcoral run shell --help` checks the client.

Install the KCoral server environment on the GPU machine, including the task's
compiler and runtime dependencies. Start it separately:

```sh
kcoral server --device gpu --gpus 0 --workers-per-gpu 1 \
  --host 127.0.0.1 --port 8000
curl http://127.0.0.1:8000/health
```

For another machine, use a reachable service address or an SSH tunnel. KCoral
executes uploaded code: keep the endpoint on a trusted network. `--url` also
accepts a KCoral Router address. A healthy response reports the target and
versions; run the evaluator smoke check below to verify that execution works.

## Usage

From the root of this checkout, run the included multi-file Triton vector-add
example. It checks correctness before measuring CUDA event time and writes a
JSON report with the device, versions, shape and mean latency in microseconds.

```sh
export KCORAL_URL=http://127.0.0.1:8000
python tools/kernel_benchmark.py --backend kcoral \
  --bundle examples/kernel-benchmark/experiment \
  --fetch results --out /tmp/kernel-trial-001 -- python evaluate.py
cat /tmp/kernel-trial-001/experiment/results/report.json
```

`--out` must be new and its parent must exist. Choose a new path per trial.
Run the same evaluator on a local GPU by changing only the backend:

```sh
python tools/kernel_benchmark.py --backend local \
  --bundle examples/kernel-benchmark/experiment \
  --fetch results --out /tmp/kernel-trial-002 -- python evaluate.py
```

For your task, make an `experiment/` directory containing the candidate,
evaluator and any inputs it imports. The command after `--` runs **inside that
directory** on either backend. Supply your own evaluator without changing its
correctness rules, reference implementation or timing method:

```sh
export HMZ_BENCHMARK_BACKEND=kcoral
export KCORAL_URL=http://127.0.0.1:8000
python /ABS/flowverse/tools/kernel_benchmark.py --bundle experiment \
  --timeout 300 --fetch results --out /tmp/my-trial-001 -- python evaluate.py
```

`--backend` overrides `HMZ_BENCHMARK_BACKEND`; the default is `local`. `--url`
overrides `KCORAL_URL`. `--timeout` is per evaluator invocation, independent of
the flow's budget. Only the KCoral backend requires the client and endpoint.

Give an existing flow the exact evaluator command in its task:

```sh
hmz exec -f /ABS/flowverse/flows/ralph_loop \
  -a agent=codex/gpt-5.6-sol:high -b duration=30m \
  'Optimize experiment/kernel.py. Keep evaluate.py and its correctness checks
   unchanged. Run every trial with:
   python /ABS/flowverse/tools/kernel_benchmark.py --backend kcoral
     --url http://127.0.0.1:8000 --bundle experiment -- python evaluate.py
   A nonzero exit is a failed trial. Keep the measured JSON and the tested source.'
```

Use absolute runner paths so each agent working directory can find it. With a
daemon already running, environment variables exported in a new terminal may
not reach it: put `--backend` and `--url` in the task as above. The client and
URL must be reachable from wherever the agent's shell runs, including its
container or SSH environment. `127.0.0.1` means that environment itself.

For a flow that freezes an official evaluator command, include the explicit
backend and URL in that command from the beginning. `pfc evaluate` can record
this runner's output and exit status. The runner does not define a score schema
or convert latency into a leaderboard score; retain the task's own score
adapter and receipt rules. Keep its evaluator outside candidate edits when
the task needs independent verification.

## Server ownership

The default is **bring an existing server or Router URL**. Humanize does not
start, restart, install dependencies on, or stop that service. This lets several
flows share its GPU scheduling, and lets a CPU-only agent use another machine's
GPU. KCoral manages workers, request workspaces, leases and execution deadlines.

| Responsibility | Owner |
| --- | --- |
| Agent sessions, worktrees, flow budget | humanize |
| Candidate, reference, tolerances, timing and score | task evaluator |
| Selecting local or KCoral execution | task's runner invocation |
| GPU selection, server environment, queue, workers and isolation | KCoral operator |
| Endpoint lifetime and shared-service shutdown | user or service manager |

Starting one server per flow would multiply workers on a GPU, require the flow
to know its CUDA environment and ports, and make detach/resume and shared
shutdown ambiguous. If managed startup is added later, make it explicit and
local-only, with an owned process handle, readiness deadline, logs, GPU/port
configuration and cleanup of only that process. Never infer permission to
terminate or restart a server from its URL. A service manager is sufficient
for automatic startup today.

KCoral is not a replacement for humanize's `Env`: its requests are stateless;
an `Env` provides persistent files and subprocesses to agent sessions. Keep the
agent workspace local/SSH/container-based and send only benchmark executions
to KCoral.

## Contract

- KCoral receives a fresh snapshot of the entire explicit bundle on every
  request; it retains the bundle name. Hidden files are included, except Python
  bytecode caches. Do not bundle a whole checkout, credentials, virtualenv or
  prior results unnecessarily. Symlinks and special input files are refused by
  the client. Local execution works in the original bundle and can modify it.
- The executable and dependencies must exist on the selected execution host.
  Use `python`, not a local interpreter's absolute path, for portable commands.
  Shell expansions are not implicit: arguments after `--` stay separate.
- Environment variables are not forwarded to the server automatically. The
  worker owns its assigned GPU. For custom environment overrides or profilers,
  use `kcoral run shell`, `ncu` or `compute-sanitizer` directly with their options.
- Normal evaluator exit status and stdout/stderr are preserved. Remote output
  arrives after completion; local output streams as it is written. Remote
  capture limits are KCoral's. Fetch files for complete reports.
- Repeat `--fetch` for disjoint files or directories relative to the bundle.
  Both backends save them under `OUT/BUNDLE_NAME/PATH`, including after an
  ordinary nonzero evaluator exit. Missing artifacts fail the invocation.
  Failed collection may leave partial files; use a new output path next time.
- A local timeout kills the evaluator process group and returns 124. Remote
  timeout limits and error exit codes are the client's/server's; the server
  may cap the requested deadline. There is no fallback to local execution and
  no automatic execution retry here. A transport failure can have an unknown
  execution outcome. Interrupting a remote client does not cancel the server
  request; it may run until its deadline. A hard timeout may lose artifacts.
- Exit zero means only that the evaluator succeeded. Correctness must be checked
  by the evaluator before reporting a timing. HTTP duration includes transfer,
  compilation and queueing; it is not kernel latency. The example's CUDA event
  mean is a smoke measurement, not a standardized performance leaderboard.

## Validation

Run the runner tests in an environment with pytest:

```sh
python -m pytest tests/test_kernel_benchmark.py
```

These exercise local execution, descendant cleanup on timeout, literal argument
passing, failure/stream/artifact preservation and delegation to a stand-in
KCoral executable. They do not claim to test the remote wire protocol. Use the
Triton command above for a real GPU smoke check through the official client.

To exercise real HTTP execution, file upload and artifact downloads in the
tests, put the client on `PATH` and provide a CPU or GPU server:

```sh
KCORAL_TEST_URL=http://127.0.0.1:8000 python -m pytest tests/test_kernel_benchmark.py
```

Two additional cases verify the same multi-file evaluator with successful and
nonzero exits. A CPU service verifies the transport; it does not verify CUDA
compilation, GPU correctness or timing.

## Maintainers

[humanfia](https://github.com/humanfia).

## Contributing

Keep evaluator behavior independent of the transport. Add a regression test for
changes to execution or artifact semantics.

## License

See the repository's licensing terms.
