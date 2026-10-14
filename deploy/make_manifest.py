#!/usr/bin/env python3
"""Manifeste d'un dataset (chemins relatifs, tailles, nombre de fichiers) et vérification d'une copie.

    python3 deploy/make_manifest.py DOSSIER -o deploy/manifest_musdb18.json            # écrit
    python3 deploy/make_manifest.py DOSSIER --check deploy/manifest_musdb18.json       # vérifie (code 1 si écart)

Seuls les fichiers `--pattern` (défaut : *.stem.mp4, donc ni processed/ ni split.json) sont listés. Bibliothèque standard seule.
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
    problems += [f"taille différente : {p} ({now[p]} octets au lieu de {expected[p]})"
                 for p in sorted(expected.keys() & now.keys()) if now[p] != expected[p]]
    problems += [f"en trop : {p}" for p in sorted(now.keys() - expected.keys())]
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", type=Path)
    ap.add_argument("-o", "--output", type=Path, help="écrit le manifeste ici")
    ap.add_argument("--check", type=Path, help="compare le dossier à ce manifeste")
    ap.add_argument("--pattern", default="*.stem.mp4")
    args = ap.parse_args()
    if not args.root.is_dir():
        sys.exit(f"ERREUR: dossier introuvable : {args.root}")

    if args.check:
        manifest = json.loads(args.check.read_text())
        problems = check(args.root, manifest)
        for p in problems[:30]:
            print(p)
        if problems:
            sys.exit(f"ERREUR: {len(problems)} écart(s) avec le manifeste (transfert incomplet ? relancer rsync / wget -c)")
        print(f"OK : {manifest['n_files']} fichiers, {manifest['total_bytes'] / 1e9:.2f} Go conformes au manifeste")
        return

    manifest = scan(args.root, args.pattern)
    if manifest["n_files"] == 0:
        sys.exit(f"ERREUR: aucun fichier {args.pattern} dans {args.root}")
    text = json.dumps(manifest, indent=1)
    if args.output:
        args.output.write_text(text + "\n")
    print(f"{manifest['n_files']} fichiers, {manifest['total_bytes'] / 1e9:.2f} Go" + (f" -> {args.output}" if args.output else ""))


if __name__ == "__main__":
    main()
