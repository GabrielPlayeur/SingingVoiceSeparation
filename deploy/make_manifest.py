#!/usr/bin/env python3
"""Manifest of a dataset (relative paths, sizes, number of files) and verification of a copy.

    python3 deploy/make_manifest.py DIR -o deploy/manifest_musdb18.json            # writes
    python3 deploy/make_manifest.py DIR --check deploy/manifest_musdb18.json       # checks (exit code 1 on mismatch)

Only the `--pattern` files (default: *.stem.mp4, hence neither processed/ nor split.json) are listed. Standard library only.
"""
import argparse
import json
import sys
from pathlib import Path


def scan(root: Path, pattern: str) -> dict:
    files = {p.relative_to(root).as_posix(): p.stat().st_size for p in sorted(root.rglob(pattern)) if p.is_file()}
    return {"pattern": pattern, "n_files": len(files), "total_bytes": sum(files.values()), "files": files}


def check(root: Path, manifest: dict) -> list[str]:
    now = scan(root, manifest["pattern"])["files"]
    expected = manifest["files"]
    problems = [f"manquant : {p}" for p in sorted(expected.keys() - now.keys())]
    problems += [f"different size: {p} ({now[p]} bytes instead of {expected[p]})"
                 for p in sorted(expected.keys() & now.keys()) if now[p] != expected[p]]
    problems += [f"extra: {p}" for p in sorted(now.keys() - expected.keys())]
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", type=Path)
    ap.add_argument("-o", "--output", type=Path, help="writes the manifest here")
    ap.add_argument("--check", type=Path, help="compares the folder to this manifest")
    ap.add_argument("--pattern", default="*.stem.mp4")
    args = ap.parse_args()
    if not args.root.is_dir():
        sys.exit(f"ERROR: folder not found: {args.root}")

    if args.check:
        manifest = json.loads(args.check.read_text())
        problems = check(args.root, manifest)
        for p in problems[:30]:
            print(p)
        if problems:
            sys.exit(f"ERROR: {len(problems)} mismatch(es) with the manifest (incomplete transfer? rerun rsync / wget -c)")
        print(f"OK: {manifest['n_files']} files, {manifest['total_bytes'] / 1e9:.2f} GB conform to the manifest")
        return

    manifest = scan(args.root, args.pattern)
    if manifest["n_files"] == 0:
        sys.exit(f"ERROR: no {args.pattern} file in {args.root}")
    text = json.dumps(manifest, indent=1)
    if args.output:
        args.output.write_text(text + "\n")
    print(f"{manifest['n_files']} files, {manifest['total_bytes'] / 1e9:.2f} GB" + (f" -> {args.output}" if args.output else ""))


if __name__ == "__main__":
    main()
