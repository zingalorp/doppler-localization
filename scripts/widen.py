"""Copy projects with a wider search area, to test how the search does as the area grows.

    uv run python scripts/widen.py 50 [project ...]     # 100 x 100 km copies, named <project>-100km

Each copy keeps the original's recording, spectrogram, passes (hand traces too), centre and known
start; only `half_km` changes, and its flight data must be cut again (step 3: `fetch`), wider.
Projects already widened (named ...km) are skipped.
"""

from __future__ import annotations

import json
import shutil
import sys

from doppler_loc.project import DATA, to_json, write_atomic

PROJECTS = DATA / "projects"


def widen(src, half_km: float):
    name = f"{src.name}-{2 * half_km:g}km"
    dst = PROJECTS / name
    if dst.exists():
        print(f"{name}: exists")
        return dst
    state = json.loads((src / "project.json").read_text(encoding="utf-8"))
    state["settings"].update(name=name, half_km=half_km)
    state.update(adsb=None, verdicts={})
    dst.mkdir()
    if (src / "spec.npz").exists():
        shutil.copy2(src / "spec.npz", dst / "spec.npz")
    write_atomic(dst / "project.json", to_json(state, indent=1))
    print(f"{name}: made")
    return dst


if __name__ == "__main__":
    half = float(sys.argv[1])
    names = sys.argv[2:] or sorted(p.name for p in PROJECTS.iterdir() if (p / "project.json").exists()
                                   and not p.name.endswith("km"))
    for n in names:
        widen(PROJECTS / n, half)
