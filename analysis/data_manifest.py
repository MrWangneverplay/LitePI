"""Which result files back which paper table (results/MANIFEST.json).

A number may enter the paper only through a manifest entry.  Each entry names a
table or dataset, the files behind it and their status:

  valid     use it
  partial   usable as far as the reason says
  stale     must be regenerated, do not cite
  pending   not produced yet
  archive   superseded; kept on disk, moved by `archive`

Commands (nothing moves unless `archive --yes` is given):

  status                     one line per entry, files found and code versions
  check [--freeze REF]       valid entries have their files; with REF, each
                             record must come from clean code at or after REF
  set ID --status S [--reason TEXT]    record a new state of an entry
  orphans                    result folders no entry mentions
  archive [--yes]            move `archive` entries to results/_archive/<date>/

    python analysis/data_manifest.py status
    python analysis/data_manifest.py set T7-13 --status valid --reason "Pi5 two hosts"
    python analysis/data_manifest.py check --freeze data-freeze-2
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
import subprocess
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "results" / "MANIFEST.json"
STATUSES = ("valid", "partial", "stale", "pending", "archive")


def git_sha(root: Path = ROOT) -> str | None:
    """HEAD, with -dirty when the protocol or analysis code has local changes."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True,
                              check=False, timeout=30).stdout.strip()
    sha = git("rev-parse", "HEAD")
    if not sha:
        return None
    return sha + ("-dirty" if git("status", "--porcelain", "--", "analysis", "seal") else "")


def load() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def save(data: dict) -> None:
    MANIFEST.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def files_of(entry: dict) -> list[Path]:
    out = []
    for pattern in entry.get("sources", []):
        out += [Path(p) for p in glob.glob(str(ROOT / pattern))]
    return sorted({p for p in out if p.is_file()})


def shas_of(paths: list[Path]) -> set[str]:
    shas = set()
    for path in paths:
        if path.suffix != ".json":
            continue
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(rec, dict):
            shas.add(rec.get("git_sha") or rec.get("rerun_git_sha") or "no-sha")
    return shas


def cmd_status(data: dict) -> int:
    counts: dict[str, int] = {}
    print(f"{'id':<14}{'status':<9}{'files':>7}/{'need':<6}{'versions':<26}name")
    for e in data["entries"]:
        counts[e["status"]] = counts.get(e["status"], 0) + 1
        paths = files_of(e)
        versions = ",".join(sorted(s[:7] + ("*" if s.endswith("-dirty") else "")
                                   for s in shas_of(paths)))
        print(f"{e['id']:<14}{e['status']:<9}{len(paths):>7}/{e.get('min_files', 0):<6}"
              f"{versions[:25]:<26}{e['name']}")
        if e.get("reason") and e["status"] != "valid":
            print(f"{'':<14}-> {e['reason']}")
    print(counts)
    return 0


def is_ancestor(ref: str, sha: str) -> bool:
    return subprocess.run(["git", "merge-base", "--is-ancestor", ref, sha], cwd=ROOT,
                          capture_output=True, check=False).returncode == 0


def cmd_check(data: dict, freeze: str | None) -> int:
    problems = []
    for e in data["entries"]:
        paths = files_of(e)
        if e["status"] == "pending" and len(paths) >= max(e.get("min_files", 1), 1):
            print(f"note  {e['id']}: files exist, update its status")
        if e["status"] != "valid":
            continue
        if len(paths) < e.get("min_files", 1):
            problems.append(f"{e['id']}: {len(paths)} files, needs {e.get('min_files', 1)}")
        if freeze and not e.get("freeze_exempt"):
            for sha in sorted(shas_of(paths)):
                if sha == "no-sha":
                    problems.append(f"{e['id']}: a record has no git_sha")
                elif sha.endswith("-dirty"):
                    problems.append(f"{e['id']}: produced from uncommitted code ({sha[:7]})")
                elif not is_ancestor(freeze, sha):
                    problems.append(f"{e['id']}: {sha[:7]} is older than {freeze}")
    for line in problems:
        print("FAIL ", line)
    print("OK" if not problems else f"{len(problems)} problems")
    return 1 if problems else 0


def cmd_set(data: dict, entry_id: str, status: str, reason: str | None) -> int:
    entry = next((e for e in data["entries"] if e["id"] == entry_id), None)
    if entry is None:
        raise SystemExit(f"no entry {entry_id!r}")
    entry["status"] = status
    if reason is not None:
        entry["reason"] = reason
    save(data)
    print(f"{entry_id}: {status}")
    return 0


def leaf_dirs() -> list[Path]:
    out = set()
    for base in ("results/raw_results", "results"):
        for path in (ROOT / base).rglob("*"):
            if path.is_file() and "_archive" not in path.parts and path.name != MANIFEST.name:
                out.add(path.parent)
    return sorted(out)


def cmd_orphans(data: dict) -> int:
    known = {p.parent for e in data["entries"] for p in files_of(e)}
    orphans = [d for d in leaf_dirs() if d not in known and d != ROOT / "results"]
    for d in orphans:
        print(d.relative_to(ROOT).as_posix())
    print(f"{len(orphans)} folders are in no entry")
    return 0


def cmd_archive(data: dict, yes: bool) -> int:
    target = ROOT / "results" / "_archive" / date.today().isoformat()
    moved = 0
    for e in data["entries"]:
        if e["status"] != "archive":
            continue
        for path in files_of(e):
            dest = target / path.relative_to(ROOT / "results")
            print(("move " if yes else "would move ") + path.relative_to(ROOT).as_posix())
            moved += 1
            if yes:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(dest))
    print(f"{moved} files {'moved to ' + str(target) if yes else 'listed; pass --yes to move'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    c = sub.add_parser("check")
    c.add_argument("--freeze", default=None)
    s = sub.add_parser("set")
    s.add_argument("id")
    s.add_argument("--status", choices=STATUSES, required=True)
    s.add_argument("--reason", default=None)
    sub.add_parser("orphans")
    a = sub.add_parser("archive")
    a.add_argument("--yes", action="store_true")
    args = ap.parse_args()
    data = load()
    if args.cmd == "status":
        return cmd_status(data)
    if args.cmd == "check":
        return cmd_check(data, args.freeze)
    if args.cmd == "set":
        return cmd_set(data, args.id, args.status, args.reason)
    if args.cmd == "orphans":
        return cmd_orphans(data)
    return cmd_archive(data, args.yes)


if __name__ == "__main__":
    raise SystemExit(main())
