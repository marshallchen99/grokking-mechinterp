#!/usr/bin/env python3
"""Run several training configurations concurrently.

The machine has twelve cores but the per-step work is small enough that a
single run saturates around six threads and then gets slower (memory
bandwidth, not compute).  So the throughput-optimal thing is several runs in
parallel with a few threads each, rather than one run with everything.

Jobs are given as a JSON list of dicts, each of which is passed to
`run_train.py` as command line flags:

    [{"tag": "op_sub", "op": "sub"}, {"tag": "op_mul", "op": "mul"}]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def to_flags(job: dict) -> list:
    out = []
    for k, v in job.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                out.append(flag)
        else:
            out += [flag, str(v)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True, help="path to a JSON list of job dicts")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--logdir", default=str(ROOT / "logs"))
    args = ap.parse_args()

    jobs = json.loads(Path(args.jobs).read_text())
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    queue = list(jobs)
    running = []           # (popen, tag, logfile, t0)
    done = []
    t_start = time.time()

    while queue or running:
        while queue and len(running) < args.parallel:
            job = dict(queue.pop(0))
            job.setdefault("threads", args.threads)
            tag = job["tag"]
            log = logdir / f"{tag}.log"
            fh = open(log, "w")
            cmd = [sys.executable, str(ROOT / "scripts" / "run_train.py")] + to_flags(job)
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
            running.append((proc, tag, fh, time.time()))
            print(f"[{time.time()-t_start:6.0f}s] START {tag}  ({len(queue)} queued)", flush=True)

        time.sleep(5)
        still = []
        for proc, tag, fh, t0 in running:
            if proc.poll() is None:
                still.append((proc, tag, fh, t0))
            else:
                fh.close()
                status = "OK" if proc.returncode == 0 else f"FAIL({proc.returncode})"
                print(f"[{time.time()-t_start:6.0f}s] {status:9s} {tag}  "
                      f"({time.time()-t0:.0f}s)", flush=True)
                done.append((tag, proc.returncode))
        running = still

    bad = [t for t, rc in done if rc != 0]
    print(f"\n{len(done)} jobs finished in {time.time()-t_start:.0f}s; "
          f"{len(bad)} failed" + (f": {bad}" if bad else ""), flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
