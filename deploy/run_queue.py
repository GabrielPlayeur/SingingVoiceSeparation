#!/usr/bin/env python3
"""Experiment queue: trains then evaluates each run, `--parallel N` runs at a time (a run = training THEN evaluation,
in the same location; while one evaluates, the others train).

    python deploy/run_queue.py --tag lr-v1 --group lr --parallel 3 [--dry-run] [--shutdown]

(This file is not called queue.py: launched as a script, it would shadow the standard library's `queue` module.)

- Works from a frozen checkout of the tag (git worktree in RUNS_DIR/<tag>/_code): the code can keep changing.
- Refuses to start if the working tree is not clean (only a warning with --dry-run).
- `ckpt: last` on a queue entry evaluates last.ckpt instead of the best-on-validation checkpoint (fixed-budget runs).
- A run = folder RUNS_DIR/<tag>/<name>_s<seed>/: train.log, eval.log, eval.json, metadata.json, hydra/ (resolved config),
  outputs/ (checkpoints, offline wandb folder). A failed run does not stop the others. Rerunning the same command resumes:
  the finished phases are skipped (--force to redo everything).
- End: SUMMARY.md / summary.json then a DONE file. `--shutdown` only shuts down if fetch_results.sh has set FETCHED (all the
  results are on your side); otherwise no shutdown, ever.
"""
import argparse
import datetime as dt
import glob
import json
import math
import os
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import DEPLOY_DIR, REPO_ROOT, cpu_cores, find_ckpt, load_env, mem_total_gb

from omegaconf import OmegaConf  # noqa: E402

EVAL_GB_PER_WORKER = 2.5   # cf. src/evaluate.py (default_workers)
TRAIN_GB = 3.0             # rough estimate per training (process + DataLoader workers): to be confirmed by benchmark_parallel.sh
PROCS: set = set()
PROCS_LOCK = threading.Lock()
STOP = threading.Event()


@dataclass
class Job:
    name: str
    seed: int
    overrides: list
    ckpt: str = "best"   # checkpoint to evaluate: "best" (best on validation) or "last" (final weights, fixed-budget runs)
    rundir: Path = None
    meta: dict = field(default_factory=dict)

    @property
    def key(self):
        return f"{self.name}_s{self.seed}"


def now():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def git(*args, check=True):
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], check=check, capture_output=True, text=True).stdout.strip()


def build_jobs(cfg, groups, only):
    jobs = []
    names = set()
    for e in cfg.experiments:
        if e.name in names:
            sys.exit(f"ERROR: duplicate experiment name in the queue: {e.name}")
        names.add(e.name)
        if groups and e.get("group") not in groups:
            continue
        if only and e.name not in only:
            continue
        for seed in e.get("seeds", cfg.get("seeds", [42])):
            ckpt = e.get("ckpt", "best")
            if ckpt not in ("best", "last"):
                sys.exit(f"ERROR: experiment {e.name}: ckpt={ckpt!r}, expected 'best' or 'last'")
            overrides = [str(o) for o in e.overrides]
            if any("<" in o for o in overrides):
                sys.exit(f"ERROR: experiment {e.name}: unfilled placeholder in {overrides} (replace <B> etc. in deploy/queue.yaml)")
            jobs.append(Job(e.name, int(seed), overrides, ckpt))
    unknown = (only or set()) - names
    if unknown:
        sys.exit(f"ERROR: --only: unknown experiments {sorted(unknown)} (available: {sorted(names)})")
    if not jobs:
        sys.exit("ERROR: no experiment selected (see --group / --only and deploy/queue.yaml)")
    return jobs


def resources(args, env, parallel):
    cores = args.cores or cpu_cores(env)
    mem = mem_total_gb()
    train_workers = max(1, min(4, cores // parallel))
    by_cpu = max(1, cores // parallel - 1)
    by_ram = int((mem - 2 - TRAIN_GB * parallel) // (EVAL_GB_PER_WORKER * parallel))
    eval_workers = args.eval_workers if args.eval_workers is not None else max(1, min(6, by_cpu, by_ram))
    return cores, mem, train_workers, eval_workers


def train_cmd(job, py, env, tag, train_workers):
    return [py, "-m", "src.train", *job.overrides, f"seed={job.seed}",
            f"data.dataset_path={env['MUSDB_PROCESSED']}", f"data.split_path={env['SPLIT_PATH']}",
            f"data.num_workers={train_workers}", f"output_dir={job.rundir / 'outputs'}",
            "training.evaluate_at_end=false",                      # the evaluation is launched afterwards, with JSON
            f"wandb.run_name={job.key}-{tag}", f"hydra.run.dir={job.rundir / 'hydra'}"]


def eval_cmd(job, py, env, ckpt, eval_workers, extra=()):
    return [py, "-m", "src.evaluate", ckpt, "--dataset-path", env["MUSDB_PROCESSED"], "-o", str(job.rundir / "eval.json"),
            "-w", str(eval_workers), "--oracle-cache", env["ORACLE_CACHE_DIR"], *extra]


def save_meta(job):
    tmp = job.rundir / "metadata.json.tmp"
    tmp.write_text(json.dumps(job.meta, indent=2))
    os.replace(tmp, job.rundir / "metadata.json")


def run_cmd(cmd, cwd, log_path, env):
    """Launches cmd (own process group, so that everything can be killed on Ctrl-C), log in append mode. Returns the return code."""
    with open(log_path, "ab") as log, PROCS_LOCK:
        if STOP.is_set():
            return -signal.SIGTERM
        log.write(f"\n$ {shlex.join(cmd)}\n".encode()); log.flush()
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        PROCS.add(p)
    rc = p.wait()
    with PROCS_LOCK:
        PROCS.discard(p)
    return rc


def kill_all():
    STOP.set()
    with PROCS_LOCK:
        for p in list(PROCS):
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def phase_done(job, phase):
    return job.meta.get(phase, {}).get("returncode") == 0


def run_job(job, ctx):
    job.rundir.mkdir(parents=True, exist_ok=True)
    meta_path = job.rundir / "metadata.json"
    if meta_path.exists() and not ctx["force"]:
        job.meta = json.loads(meta_path.read_text())
    job.meta.update(name=job.name, seed=job.seed, tag=ctx["tag"], commit=ctx["commit"], overrides=job.overrides, ckpt=job.ckpt)
    child_env = {**os.environ, "WANDB_MODE": ctx["env"]["WANDB_MODE"], "PYTHONUNBUFFERED": "1", "HYDRA_FULL_ERROR": "1"}
    py = ctx["py"]

    # --- training
    if phase_done(job, "train") and find_ckpt(job.rundir, job.ckpt):
        print(f"[{job.key}] training already finished, skipping", flush=True)
    else:
        shutil.rmtree(job.rundir / "outputs", ignore_errors=True)  # leftovers of an interrupted run
        cmd = train_cmd(job, py, ctx["env"], ctx["tag"], ctx["train_workers"])
        job.meta.pop("eval", None)
        job.meta.update(status="training", train={"cmd": cmd, "start": now()})
        save_meta(job)
        print(f"[{job.key}] training ...", flush=True)
        t0 = time.time()
        rc = run_cmd(cmd, ctx["code"], job.rundir / "train.log", child_env)
        job.meta["train"].update(end=now(), returncode=rc, seconds=round(time.time() - t0))
        if rc != 0:
            job.meta["status"] = "failed_train"
            save_meta(job)
            print(f"[{job.key}] training FAILED (code {rc}): see {job.rundir / 'train.log'}", flush=True)
            return job
    ckpt = find_ckpt(job.rundir, job.ckpt)
    job.meta["checkpoint"] = ckpt
    if ckpt:
        job.meta["wandb_run_id"] = Path(ckpt).parents[1].name  # outputs/VoiceSeparation/<id>/checkpoints/x.ckpt
        slim = job.rundir / "checkpoint_slim.ckpt"  # weights only (~39 MB): transferable; non-blocking
        if not slim.exists():
            subprocess.run([py, str(DEPLOY_DIR / "slim_ckpt.py"), ckpt, str(slim)], capture_output=True)
    save_meta(job)

    # --- evaluation
    if ctx["no_eval"]:
        job.meta["status"] = "trained"
    elif phase_done(job, "eval") and (job.rundir / "eval.json").exists():
        print(f"[{job.key}] evaluation already finished, skipping", flush=True)
    elif ckpt is None:
        job.meta["status"] = "failed_eval"
        job.meta["eval"] = {"returncode": -1, "error": "no checkpoint found"}
    else:
        cmd = eval_cmd(job, py, ctx["env"], ckpt, ctx["eval_workers"], ctx["eval_args"])
        job.meta.update(status="evaluating", eval={"cmd": cmd, "start": now()})
        save_meta(job)
        print(f"[{job.key}] evaluation ...", flush=True)
        t0 = time.time()
        rc = run_cmd(cmd, ctx["code"], job.rundir / "eval.log", child_env)
        job.meta["eval"].update(end=now(), returncode=rc, seconds=round(time.time() - t0))
        job.meta["status"] = "done" if rc == 0 else "failed_eval"
        if rc != 0:
            print(f"[{job.key}] evaluation FAILED (code {rc}): see {job.rundir / 'eval.log'}", flush=True)
    if job.meta.get("eval", {}).get("returncode") == 0:
        job.meta["status"] = "done"
        job.meta["results"] = summarize_eval(job.rundir / "eval.json")
    save_meta(job)
    print(f"[{job.key}] finished: {job.meta['status']}", flush=True)
    return job


def summarize_eval(path):
    rows = json.loads(Path(path).read_text())
    out = {"n_tracks": len(rows)}
    for k in ("sdr", "si_sdri", "sdr_oracle", "si_sdri_oracle"):
        v = [r[k] for r in rows if k in r and not math.isnan(r[k])]
        if v:
            out[k + "_median"] = round(statistics.median(v), 3)
    return out


def write_summary(tag_dir, jobs):
    rows = []
    for j in jobs:
        m = j.meta
        r = m.get("results", {})
        rows.append({"run": j.key, "status": m.get("status", "not_run"), "wandb_run_id": m.get("wandb_run_id"),
                     "train_min": round(m.get("train", {}).get("seconds", 0) / 60, 1),
                     "eval_min": round(m.get("eval", {}).get("seconds", 0) / 60, 1),
                     "sdr_median": r.get("sdr_median"), "si_sdri_median": r.get("si_sdri_median"),
                     "sdr_oracle_median": r.get("sdr_oracle_median")})
    (tag_dir / "summary.json").write_text(json.dumps(rows, indent=2))
    f = lambda x: "-" if x is None else f"{x:.2f}"  # noqa: E731
    lines = ["| run | status | W&B id | train (min) | eval (min) | median SDR | median SI-SDRi | oracle SDR |", "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['run']} | {r['status']} | {r['wandb_run_id'] or '-'} | {r['train_min']} | {r['eval_min']} | {f(r['sdr_median'])} | "
              f"{f(r['si_sdri_median'])} | {f(r['sdr_oracle_median'])} |" for r in rows]
    (tag_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    return rows, "\n".join(lines)


def ensure_worktree(tag, commit, code_dir):
    if code_dir.exists():
        head = subprocess.run(["git", "-C", str(code_dir), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        if head != commit:
            sys.exit(f"ERROR: {code_dir} is on {head[:8]} instead of {commit[:8]} (tag {tag}). Was the tag moved? "
                     f"Delete the folder: git -C {REPO_ROOT} worktree remove --force {code_dir}")
        return
    code_dir.parent.mkdir(parents=True, exist_ok=True)
    git("worktree", "add", "--detach", str(code_dir), tag)


def wait_fetched_then_shutdown(tag_dir, env):
    fetched = tag_dir / "FETCHED"
    limit = time.time() + float(env["SHUTDOWN_WAIT_MIN"]) * 60
    print(f"--shutdown: waiting for {fetched} (set by deploy/fetch_results.sh {tag_dir.name} from your PC), "
          f"{env['SHUTDOWN_WAIT_MIN']} min max. The instance stays billed as long as it is on.", flush=True)
    while time.time() < limit:
        if fetched.exists():
            print("FETCHED received: shutting down in 15 s.", flush=True)
            time.sleep(15)
            subprocess.run(["bash", "-c", env["SHUTDOWN_CMD"]])
            return 0
        time.sleep(30)
    print("Timeout without FETCHED: the instance is NOT shut down (your results are not all fetched). "
          "Run fetch_results.sh then shut down by hand.", flush=True)
    return 3


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    env = load_env()
    ap.add_argument("--tag", default=env["GIT_TAG"], help="git tag to run (default: GIT_TAG of env.sh)")
    ap.add_argument("--queue-file", type=Path, default=DEPLOY_DIR / "queue.yaml")
    ap.add_argument("--group", action="append", help="group(s) of queue.yaml to run (repeatable)")
    ap.add_argument("--only", help="experiment names separated by commas")
    ap.add_argument("--parallel", type=int, default=None, help="simultaneous runs (default: queue.yaml, otherwise DEFAULT_PARALLEL)")
    ap.add_argument("--eval-workers", type=int, default=None, help="metric processes per evaluation (default: according to cores and RAM)")
    ap.add_argument("--cores", type=int, default=None)
    ap.add_argument("--no-eval", action="store_true", help="training only")
    ap.add_argument("--eval-args", default="", help="arguments appended to `python -m src.evaluate` (test: \"--max-tracks 2\")")
    ap.add_argument("--force", action="store_true", help="redoes the finished phases")
    ap.add_argument("--dry-run", action="store_true", help="prints the commands without executing anything")
    ap.add_argument("--shutdown", action="store_true", help="shuts the instance down at the end, ONLY if FETCHED is present")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.queue_file)
    jobs = build_jobs(cfg, set(args.group or []), set(filter(None, (args.only or "").split(","))))
    parallel = max(1, min(args.parallel or cfg.get("parallel") or int(env["DEFAULT_PARALLEL"]), len(jobs)))
    cores, mem, train_workers, eval_workers = resources(args, env, parallel)
    py = str(REPO_ROOT / ".venv" / "bin" / "python")
    if not os.path.exists(py):
        py = sys.executable

    tag = args.tag
    tag_dir = Path(env["RUNS_DIR"]) / tag
    code_dir = tag_dir / "_code"
    dirty = git("status", "--porcelain")
    commit = git("rev-parse", "--verify", "--quiet", f"refs/tags/{tag}^{{commit}}", check=False)
    problems = []
    if dirty:
        problems.append("working tree not clean (git status): commit or stash before launching\n" + dirty)
    if not commit:
        problems.append(f"tag '{tag}' does not exist (git tag {tag} && git push origin {tag})")
    if problems and not args.dry_run:
        sys.exit("ERROR: " + "\nERREUR: ".join(problems))
    commit = commit or "<commit-du-tag>"
    for j in jobs:
        j.rundir = tag_dir / j.key

    print(f"Queue: {len(jobs)} run(s), tag {tag} ({commit[:8]}), {parallel} in parallel | {cores} cores, {mem:.0f} GiB RAM")
    print(f"       DataLoader: {train_workers} worker(s) per training | evaluation: {eval_workers} metric processes per run")
    est_ram = parallel * (TRAIN_GB + EVAL_GB_PER_WORKER * eval_workers)
    if est_ram > mem - 2:
        print(f"WARNING: estimated RAM {est_ram:.0f} GiB if the {parallel} runs evaluate at the same time, for {mem:.0f} GiB "
              f"available: risk of saturation (smaller --parallel, or validate with benchmark_parallel.sh)")
    for p in problems:
        print("WARNING (blocking outside --dry-run):", p)

    if args.dry_run:
        print(f"\n$ git -C {REPO_ROOT} worktree add --detach {code_dir} {tag}   # frozen checkout, cwd of all the commands")
        for j in jobs:
            print(f"\n# {j.key}  ->  {j.rundir}")
            print("  train:", shlex.join(train_cmd(j, py, env, tag, train_workers)))
            if not args.no_eval:
                print("  eval:", shlex.join(eval_cmd(j, py, env, "<best .ckpt of outputs/VoiceSeparation/*/checkpoints/>", eval_workers, shlex.split(args.eval_args))))
        print(f"\nWANDB_MODE={env['WANDB_MODE']} | end: SUMMARY.md + DONE in {tag_dir}" +
              (f" | then shutdown (`{env['SHUTDOWN_CMD']}`) only if {tag_dir / 'FETCHED'} exists" if args.shutdown else " | no shutdown"))
        return 0

    ensure_worktree(tag, commit, code_dir)
    tag_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("DONE", "FETCHED"):
        (tag_dir / stale).unlink(missing_ok=True)
    shutil.copy(args.queue_file, tag_dir / "queue.yaml")
    ctx = dict(tag=tag, commit=commit, env=env, py=py, code=code_dir, force=args.force, no_eval=args.no_eval,
               train_workers=train_workers, eval_workers=eval_workers, eval_args=shlex.split(args.eval_args))
    signal.signal(signal.SIGTERM, lambda *_: (kill_all(), sys.exit(143)))
    started = time.time()
    try:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = [pool.submit(run_job, j, ctx) for j in jobs]
            for fut, j in zip(futures, jobs):
                try:
                    fut.result()
                except Exception as e:  # a launcher error on one run must not stop the others
                    j.meta.update(status="failed_launcher", error=repr(e))
                    print(f"[{j.key}] launcher ERROR: {e!r}", flush=True)
    except KeyboardInterrupt:
        kill_all()
        print("\nInterrupted: child processes stopped, no DONE. Rerun the same command to resume.")
        return 130

    rows, table = write_summary(tag_dir, jobs)
    n_ok = sum(r["status"] in ("done", "trained") for r in rows)
    (tag_dir / "DONE").write_text(json.dumps({"finished": now(), "runs": len(rows), "ok": n_ok,
                                              "wall_minutes": round((time.time() - started) / 60, 1)}, indent=2))
    print(f"\n{table}\n\n{n_ok}/{len(rows)} runs finished. DONE written in {tag_dir}")
    print(f"Fetch from the PC: deploy/fetch_results.sh {tag}")
    if args.shutdown:
        return wait_fetched_then_shutdown(tag_dir, env)
    return 0 if n_ok == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
