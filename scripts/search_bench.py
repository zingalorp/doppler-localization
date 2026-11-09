"""A fixed benchmark of the search: projects frozen as they are, searched again after every change.

    uv run python scripts/search_bench.py freeze [project[=case] ...]   # copy projects with a known spot in as cases
    uv run python scripts/search_bench.py list
    uv run python scripts/search_bench.py run [--label NAME] [--quick | --wide] [--fast] [--max-gb N] [case ...]
    uv run python scripts/search_bench.py report [run ...]       # the last runs side by side

A case is a copy of a project's settings and passes, taken once, with its spectrogram linked and its
flights left where they are (data/adsb). So retracing a pass in the app changes nothing here: the
benchmark keeps asking the same question, and a different answer means the code changed it. Each
run searches every case, one at a time, in a scratch data folder of its own, with the app's
settings (half the threads, below normal priority), and records for each:

- how far the top candidate is from the case's true spot (`truth_lat`, `truth_lon`), and its confidence;
- where the first candidate within RIGHT_M of the true spot ranks, how far it is and its confidence;
- how long the search took, and the most memory it used (the search and its workers together).

The top candidate counts as right within RIGHT_M: pins are only good to a few hundred metres.
`--quick` runs the cases over areas under 500 km each way (a known start or a small box, minutes
each), `--wide` the rest (country-wide, up to an hour each). `--fast` searches them in the fast
mode, whatever the case says. A search going over `--max-gb` of
memory is stopped, and the case marked so. Cases and runs live in data/bench/search, which stays
on this computer like the projects themselves.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from doppler_loc.project import DATA, WIDE_HALF_KM, Project  # noqa: E402

BENCH = DATA / "bench" / "search"
CASES, RUNS = BENCH / "cases", BENCH / "runs"
RIGHT_M = 1000.0


def _region(state, name) -> str | None:
    """The flights a project searched with, as a full path; a country-wide project whose settings
    changed since keeps none, so the country's file of that day is found instead."""
    reg = (state.get("adsb") or {}).get("region")
    if reg:
        path = Path(reg) if Path(reg).is_absolute() else REPO / reg
        return str(path) if path.exists() else None
    date = name.split("-whole-us")[0][-10:]
    found = sorted((DATA / "adsb" / date).glob("region_38.5*")) if (DATA / "adsb" / date).exists() else []
    return str(found[0]) if found else None


def _reference(folder: Path) -> dict | None:
    """What the project's own last search found (its candidates, not spots kept by hand)."""
    path = folder / "results.json"
    if not path.exists():
        return None
    cands = [c for c in json.loads(path.read_text(encoding="utf-8"))["candidates"] if c.get("pair") != "map" and not c.get("kept")]
    return _score(cands)


def _score(cands) -> dict:
    out = dict(candidates=len(cands), top_m=None, top_conf=None, right_rank=None, right_m=None, right_conf=None)
    if cands:
        out.update(top_m=cands[0].get("truth_m"), top_conf=cands[0].get("confidence"))
    for k, c in enumerate(cands):
        if c.get("truth_m") is not None and c["truth_m"] <= RIGHT_M:
            out.update(right_rank=k + 1, right_m=c["truth_m"], right_conf=c.get("confidence"))
            break
    return out


def freeze(names):
    projects = DATA / "projects"
    names = names or sorted(p.name for p in projects.iterdir() if (p / "project.json").exists())
    CASES.mkdir(parents=True, exist_ok=True)
    for spec in names:  # "project" or "project=case", to freeze a project again under a new name
        name, _, as_case = spec.partition("=")
        src = projects / name
        state = json.loads((src / "project.json").read_text(encoding="utf-8"))
        s = state["settings"]
        accepted = [p for p in state["passes"] if p["status"] == "accepted"]
        why = (None if s.get("truth_lat") is not None else "no true spot") or (None if (src / "spec.npz").exists() else "no spectrogram") \
            or (None if len(accepted) >= 2 or s.get("start_unix") is not None else "fewer than 2 passes accepted") \
            or (None if _region(state, name) else "no flights")
        if why:
            print(f"  {name}: skipped, {why}")
            continue
        case = CASES / (as_case or name)
        if case.exists():
            print(f"  {name}: already a case (remove {case} to freeze it again)")
            continue
        case.mkdir(parents=True)
        state["adsb"] = dict(state.get("adsb") or {}, region=_region(state, name))
        (case / "project.json").write_text(json.dumps(state), encoding="utf-8")
        os.link(src / "spec.npz", case / "spec.npz")
        meta = dict(source=name, frozen_at=time.strftime("%Y-%m-%d %H:%M"), half_km=s["half_km"], spacing_m=s["spacing_m"],
                    start_known=s.get("start_unix") is not None, accepted=sorted(p["id"] for p in accepted), reference=_reference(src))
        (case / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print(f"  {name}: frozen, {len(accepted)} passes accepted, {s['half_km']:g} km each way at {s['spacing_m']:g} m")


def _cases(args):
    quick, wide = "--quick" in args, "--wide" in args
    names = [a for a in args if not a.startswith("--")]
    out = []
    for case in sorted(CASES.iterdir()) if CASES.exists() else []:
        meta = json.loads((case / "meta.json").read_text(encoding="utf-8"))
        big = meta["half_km"] >= WIDE_HALF_KM
        if (names and case.name not in names) or (quick and big) or (wide and not big):
            continue
        out.append((case, meta))
    if names:  # in the order asked for
        out.sort(key=lambda x: names.index(x[0].name))
    return out


def list_cases(args):
    for case, meta in _cases(args):
        r = meta.get("reference") or {}
        print(f"  {case.name:40s} {meta['half_km']:6g} km at {meta['spacing_m']:5g} m, {'known start' if meta['start_known'] else 'no start':11s}"
              f" {len(meta['accepted'])} passes | when frozen: top {r.get('top_m')} m at {r.get('top_conf')}")


# Run in a process of its own, so that its memory can be measured and a runaway stopped.
_CHILD = """
import json, sys, time
from pathlib import Path
from doppler_loc.project import Project, lower_priority
lower_priority()
folder = Path(sys.argv[1])
t = time.time()
Project(folder).search(lambda m, f: print(f"{time.time() - t:7.0f} s  {m}", flush=True))
"""


def _tree_memory(proc: psutil.Process) -> float:
    total = 0
    try:
        tree = [proc, *proc.children(recursive=True)]
    except psutil.Error:  # a worker ended while its parent's children were being listed
        return 0.0
    for p in tree:
        try:
            total += p.memory_info().private if sys.platform == "win32" else p.memory_info().rss
        except psutil.Error:
            pass
    return total / 1e9


def run(args):
    label = args[args.index("--label") + 1] if "--label" in args else "run"
    max_gb = float(args[args.index("--max-gb") + 1]) if "--max-gb" in args else 0.6 * psutil.virtual_memory().total / 1e9
    rest = [a for k, a in enumerate(args) if a not in ("--label", "--max-gb") and (k == 0 or args[k - 1] not in ("--label", "--max-gb"))]
    cases = _cases(rest)
    stamp = time.strftime("%Y%m%d-%H%M")
    out_dir = RUNS / f"{stamp}-{label}"
    work = out_dir / "work"
    (work / "projects").mkdir(parents=True)
    shutil.copytree(DATA / "types", work / "types")
    _link_dir(DATA / "dem", work / "dem")
    env = dict(os.environ, DOPPLER_DATA=str(work), PYTHONIOENCODING="utf-8")
    rows = []
    print(f"{len(cases)} cases, results in {out_dir}")
    try:
        for case, meta in cases:
            folder = work / "projects" / case.name
            folder.mkdir()
            state = json.loads((case / "project.json").read_text(encoding="utf-8"))
            if "--fast" in args:
                state["settings"]["search_mode"] = "fast"
            (folder / "project.json").write_text(json.dumps(state), encoding="utf-8")
            os.link(case / "spec.npz", folder / "spec.npz")
            t = time.time()
            log = open(out_dir / f"{case.name}.log", "w", encoding="utf-8")
            proc = psutil.Popen([sys.executable, "-c", _CHILD, str(folder)], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
            peak, stopped = 0.0, None
            while proc.poll() is None:
                peak = max(peak, _tree_memory(proc))
                if peak > max_gb:
                    stopped = f"over {max_gb:.0f} GB"
                    for p in [*proc.children(recursive=True), proc]:
                        try:
                            p.kill()
                        except psutil.Error:
                            pass
                    break
                time.sleep(0.5)
            proc.wait()
            log.close()
            row = dict(case=case.name, seconds=round(time.time() - t), peak_gb=round(peak, 1), stopped=stopped)
            res = folder / "results.json"
            if not stopped and proc.returncode == 0 and res.exists():
                shutil.copy(res, out_dir / f"{case.name}.results.json")
                cands = [c for c in json.loads(res.read_text(encoding="utf-8"))["candidates"] if c.get("pair") != "map"]
                row.update(_score(cands))
            elif not stopped:
                row["stopped"] = f"failed (exit {proc.returncode}): see {case.name}.log"
            rows.append(row)
            (out_dir / "run.json").write_text(json.dumps(dict(label=label, started=stamp, rows=rows), indent=1), encoding="utf-8")
            print(_line(row, meta), flush=True)
            shutil.rmtree(folder)
    finally:
        _unlink_dir(work / "dem")
        shutil.rmtree(work, ignore_errors=True)
    report([out_dir.name])


def _line(row, meta=None) -> str:
    if row.get("stopped"):
        return f"  {row['case']:40s} stopped: {row['stopped']} after {row['seconds']} s, {row['peak_gb']} GB"
    right = "RIGHT" if row.get("top_m") is not None and row["top_m"] <= RIGHT_M else "wrong"
    first = f"first right #{row['right_rank']} ({row['right_m']} m, {row['right_conf']})" if row.get("right_rank") else "none right"
    return (f"  {row['case']:40s} {right} top {row.get('top_m')} m at {row.get('top_conf')} | {first} | "
            f"{row['seconds']} s, {row['peak_gb']} GB")


def report(names):
    runs = sorted(RUNS.iterdir()) if RUNS.exists() else []
    chosen = [RUNS / n for n in names] if names else runs[-2:]
    data = [json.loads((r / "run.json").read_text(encoding="utf-8")) for r in chosen if (r / "run.json").exists()]
    cases = sorted({row["case"] for d in data for row in d["rows"]})
    for d, r in zip(data, chosen):
        right = sum(1 for row in d["rows"] if row.get("top_m") is not None and row["top_m"] <= RIGHT_M)
        print(f"\n{r.name}: right first for {right} of {len(d['rows'])}")
    if len(data) < 2:
        return
    a, b = ({row["case"]: row for row in d["rows"]} for d in data[-2:])
    print(f"\nchanges from {chosen[-2].name} to {chosen[-1].name}:")
    for c in cases:
        if c in a and c in b and (a[c].get("top_m"), a[c].get("top_conf")) != (b[c].get("top_m"), b[c].get("top_conf")):
            print(f"  {c}: top {a[c].get('top_m')} m at {a[c].get('top_conf')} -> {b[c].get('top_m')} m at {b[c].get('top_conf')};"
                  f" {a[c]['seconds']} -> {b[c]['seconds']} s, {a[c]['peak_gb']} -> {b[c]['peak_gb']} GB")


def _link_dir(target: Path, link: Path):
    """The terrain tiles are shared, not copied: a junction on Windows, a symlink elsewhere."""
    if sys.platform == "win32":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)
    else:
        link.symlink_to(target, target_is_directory=True)


def _unlink_dir(link: Path):
    """Remove the link alone, never what it points to."""
    if not os.path.lexists(link):
        return
    if sys.platform == "win32":
        subprocess.run(["cmd", "/c", "rmdir", str(link)], check=True, capture_output=True)
    else:
        link.unlink()


if __name__ == "__main__":
    cmd, args = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else ("list", [])
    {"freeze": freeze, "list": list_cases, "run": run, "report": report}[cmd](args)
