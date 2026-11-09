"""Local web GUI for locating a recording step by step.

    uv run python -m doppler_loc.gui        # then open http://127.0.0.1:8765

Long steps run as separate processes (`python -m doppler_loc.project <step> <folder>`) and the
page polls their progress; everything else (spectrogram images, pass edits) is answered here.
"""

from __future__ import annotations

from dataclasses import asdict, fields
from functools import lru_cache
import io
import json
import mimetypes
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from matplotlib import colormaps
import numpy as np
from PIL import Image
from pydantic import BaseModel
from starlette.datastructures import Headers

from ..adsb import Archive, coverage_gaps, is_jet, load_type_table, low_coverage
from ..area import centre_and_half, clean_rings
from ..dem import TerrainTiles, zoom_for_spacing
from ..project import DATA, STEPS, Project, Settings, _retry, fit_key, rank_candidates, read_json, to_json, write_atomic
from ..visible import curve_visibility
from ..wide import QuietIndex

PROJECTS = DATA / "projects"
STATIC = Path(__file__).parent / "static"
LUT = (colormaps["magma"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def _is_local(url) -> bool:
    try:
        return urlsplit(url).hostname in ("127.0.0.1", "localhost")
    except ValueError:
        return False


class OnlyThisComputer:
    """Refuse requests a web page elsewhere makes the browser send. The server listens only on this
    computer, but a page on any site can still reach it through the browser: by pointing its own
    domain at 127.0.0.1 (the Host header gives that away) or by asking directly (the Origin header)."""

    def __init__(self, inner):
        self.inner = inner

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = Headers(scope=scope)
            origin = headers.get("origin")
            if not _is_local("//" + headers.get("host", "")) or (origin is not None and not _is_local(origin)):
                await Response("only pages from this computer can use this app", 403, media_type="text/plain")(scope, receive, send)
                return
        await self.inner(scope, receive, send)


app = FastAPI(title="doppler-loc")
app.add_middleware(OnlyThisComputer)
_running: dict[tuple[str, str], subprocess.Popen] = {}


def _project(name) -> Project:
    if not NAME.match(name) or not (PROJECTS / name / "project.json").exists():
        raise HTTPException(404, f"no project {name!r}")
    return Project(PROJECTS / name)


def _job(name, step) -> dict | None:
    path = PROJECTS / name / "jobs" / f"{step}.json"
    if not path.exists():
        return None
    status = read_json(path)
    proc = _running.get((name, step))
    if status.get("state") == "running" and (proc is None or proc.poll() is not None):
        # The process is gone without saying it finished (killed, or the server restarted), unless
        # the status now names another process that is still running (the step was started again
        # from the command line, say).
        if not _pid_alive(status.get("pid")):
            status.update(state="error", message="the step stopped unexpectedly")
    return status


def _pid_alive(pid) -> bool:
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if sys.platform == "win32":
        # os.kill(pid, 0) would send Ctrl+C there (signal 0 is CTRL_C_EVENT), so ask the kernel.
        import ctypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = ctypes.c_void_p
        handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5  # access denied: it exists but isn't ours
        try:
            code = ctypes.c_ulong()
            ok = k32.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code))
            return bool(ok) and code.value == 259  # STILL_ACTIVE
        finally:
            k32.CloseHandle(ctypes.c_void_p(handle))
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _busy(name) -> str | None:
    for step in STEPS:
        status = _job(name, step)
        if status and status["state"] == "running":
            return step
    return None


# --- projects --------------------------------------------------------------------------


@lru_cache(maxsize=256)
def _summary(folder: str, mtimes: tuple) -> dict:
    """What the project list shows for one project: how far it got and, when the location is
    known, how well the search did (cached until project.json or results.json change)."""
    folder = Path(folder)
    state = read_json(folder / "project.json")
    s, passes = state["settings"], state["passes"]
    out = dict(name=folder.name, date=s["date"], duration=state["duration"], passes=len(passes),
               accepted=sum(p["status"] == "accepted" for p in passes),
               aircraft=(state.get("adsb") or {}).get("aircraft"), start_known=s.get("start_unix") is not None,
               truth=s.get("truth_lat") is not None, candidates=None, best=None, verdicts=len(state.get("verdicts", {})))
    if (folder / "results.json").exists():
        cands = rank_candidates(read_json(folder / "results.json"), state)["candidates"]
        out["candidates"] = len(cands)
        if cands:
            out["best"] = dict(truth_m=cands[0].get("truth_m"), explained=cands[0].get("explained"),
                               of=len(cands[0].get("explains") or []))
            # The first candidate within 1 km of the known location, for tests.
            near = [c["rank"] for c in cands if c.get("truth_m") is not None and c["truth_m"] < 1000]
            out["truth_rank"] = near[0] if near else None
    return out


@app.get("/api/projects")
def list_projects():
    out = []
    for path in sorted(PROJECTS.glob("*/project.json")):
        res = path.parent / "results.json"
        out.append(_summary(str(path.parent), (path.stat().st_mtime, res.stat().st_mtime if res.exists() else 0)))
    return out


MEDIA = {".aac", ".aif", ".aiff", ".avi", ".flac", ".m4a", ".m4v", ".mka", ".mkv", ".mov", ".mp3", ".mp4", ".mpeg", ".mpg", ".oga", ".ogg",
         ".opus", ".ts", ".wav", ".webm", ".wma", ".wmv"}


def _roots() -> list[str]:
    if sys.platform != "win32":
        return ["/"]
    return [f"{d}:\\" for d in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if os.path.exists(f"{d}:\\")]


@app.get("/api/browse")
def browse(path: str = ""):
    """The folders and audio or video files in a folder of this computer, for picking a recording
    (a web page can't ask the browser for a file's path). `path` empty: the data folder's tests."""
    start = Path(path).expanduser() if path else (DATA / "tests" if (DATA / "tests").is_dir() else Path.home())
    try:
        here = start.resolve()
        entries = sorted(os.scandir(here), key=lambda e: e.name.lower())
    except OSError as e:
        raise HTTPException(400, f"can't open {start}: {e.strerror or e}") from e
    dirs, files = [], []
    for e in entries:
        if e.name.startswith("."):
            continue
        try:
            if e.is_dir():
                dirs.append(e.name)
            elif Path(e.name).suffix.lower() in MEDIA:
                files.append(dict(name=e.name, path=str(here / e.name), size=e.stat().st_size))
        except OSError:  # a broken link, or a file the system won't let us look at
            continue
    return dict(path=str(here), parent=str(here.parent) if here.parent != here else None, sep=os.sep, roots=_roots(), dirs=dirs, files=files)


class NewProject(BaseModel):
    name: str
    audio: str
    date: str
    lat: float
    lon: float
    half_km: float = 25.0
    mic_height: float = 2.0
    truth_lat: float | None = None
    truth_lon: float | None = None


@app.post("/api/projects")
def create_project(body: NewProject):
    if not NAME.match(body.name):
        raise HTTPException(400, "name: letters, digits, - and _ only")
    if (PROJECTS / body.name).exists():
        raise HTTPException(400, f"a project called {body.name!r} already exists")
    if not Path(body.audio).expanduser().is_file():
        raise HTTPException(400, f"no audio file at {body.audio}")
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", body.date):
        raise HTTPException(400, "date: YYYY-MM-DD")
    settings = Settings(**{**body.model_dump(), "audio": str(Path(body.audio).expanduser().resolve())})
    Project.create(settings, PROJECTS)
    return get_project(body.name)


@app.get("/api/projects/{name}")
def get_project(name: str):
    p = _project(name)
    return dict(name=name, settings=asdict(p.settings), duration=p.state["duration"], passes=p.state["passes"],
                adsb=p.state["adsb"], adsb_gaps=_adsb_gaps(p), adsb_low=_adsb_low(p), verdicts=p.state.get("verdicts", {}),
                jobs={s: _job(name, s) for s in STEPS}, has_results=(p.folder / "results.json").exists(),
                has_spec=(p.folder / "spec.npz").exists())


@app.delete("/api/projects/{name}")
def delete_project(name: str):
    """Delete a project's folder: settings, passes, spectrogram, results. The recording and the flight
    data (which other projects may share) stay."""
    p = _project(name)
    if busy := _busy(name):
        raise HTTPException(409, f"{busy} is still running: wait for it to finish")
    for step in STEPS:
        _running.pop((name, step), None)
    _retry(lambda: shutil.rmtree(p.folder))
    return {"ok": True}


@app.patch("/api/projects/{name}/settings")
def update_settings(name: str, body: dict):
    p = _project(name)
    if _busy(name):
        raise HTTPException(409, "wait for the running step to finish")
    allowed = {f.name for f in fields(Settings)} - {"name", "audio"}
    changed = set()
    if body.get("area"):  # the square around a shape comes from the shape
        try:
            body["area"] = clean_rings(body["area"])
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        body["lat"], body["lon"], body["half_km"] = centre_and_half(body["area"])
    elif "area" in body:
        body["area"], body["area_name"] = None, None
    for key, value in body.items():
        if key not in allowed:
            raise HTTPException(400, f"cannot change {key}")
        if getattr(p.settings, key) != value:
            changed.add(key)
        setattr(p.settings, key, value)
    if {"lat", "lon", "half_km", "audible_km", "date"} & changed:  # saving a form resends what hasn't changed
        p.state["adsb"] = None  # the flight data must be cut out again for a new area or day
    p.save()
    return get_project(name)


# --- steps -----------------------------------------------------------------------------


@app.post("/api/projects/{name}/jobs/{step}")
def start_job(name: str, step: str):
    p = _project(name)
    if step not in STEPS:
        raise HTTPException(404, f"no step {step!r}")
    if busy := _busy(name):
        raise HTTPException(409, f"{busy} is still running")
    (p.folder / "jobs").mkdir(exist_ok=True)
    # Written before the step starts, so it can't overwrite the step's own first report.
    write_atomic(p.folder / "jobs" / f"{step}.json", json.dumps(dict(step=step, state="running", message="starting",
                                                                     progress=0.0)))
    with open(p.folder / "jobs" / f"{step}.log", "w") as log:
        _running[(name, step)] = subprocess.Popen([sys.executable, "-m", "doppler_loc.project", step, str(p.folder)],
                                                  stdout=log, stderr=subprocess.STDOUT,
                                                  start_new_session=sys.platform != "win32")
    return _job(name, step)


def _kill_tree(pid):
    """End a step's process and its workers."""
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except OSError:
            pass


@app.post("/api/projects/{name}/jobs/{step}/stop")
def stop_job(name: str, step: str):
    """Stop a running step. What it had saved before stays: each step writes its results only when
    it finishes, so a stopped search leaves the last one's candidates, and it can simply be run again."""
    _project(name)
    status = _job(name, step)
    if not status or status["state"] != "running":
        raise HTTPException(409, f"{step} isn't running")
    proc = _running.get((name, step))
    for pid in {proc.pid if proc else None, status.get("pid")} - {None}:
        _kill_tree(pid)
    if proc:
        proc.wait(timeout=10)
    write_atomic(PROJECTS / name / "jobs" / f"{step}.json", json.dumps(dict(step=step, state="stopped", message="stopped", progress=0.0)))
    return _job(name, step)


@app.get("/api/projects/{name}/jobs")
def jobs(name: str):
    _project(name)
    return {s: _job(name, s) for s in STEPS}


# --- spectrogram -----------------------------------------------------------------------


@lru_cache(maxsize=2)
def _spec(folder: str, mtime: float):
    d = np.load(Path(folder) / "spec.npz")
    return d["t"], d["f"], d["score"].astype(np.float32)


def _resample(a, n, axis):
    """Resample `a` to n samples along axis: the brightest of each block when shrinking, so thin
    lines stay visible, and nearest when stretching."""
    m = a.shape[axis]
    if m == 0:
        return np.zeros(a.shape[:axis] + (n,) + a.shape[axis + 1:], a.dtype)
    if m >= n:
        starts = np.floor(np.arange(n) * m / n).astype(int)
        return np.maximum.reduceat(a, starts, axis=axis)
    return np.take(a, np.minimum((np.arange(n) * m / n).astype(int), m - 1), axis=axis)


@app.get("/api/projects/{name}/spectrogram.png")
def spectrogram(name: str, t0: float, t1: float, f0: float, f1: float, w: int = 1200, h: int = 400,
                vmax: float = 14.0):
    p = _project(name)
    path = p.folder / "spec.npz"
    if not path.exists():
        raise HTTPException(404, "prepare the spectrogram first")
    t, f, score = _spec(str(p.folder), path.stat().st_mtime)
    w, h = max(1, min(w, 4000)), max(1, min(h, 2000))
    # Columns and rows cover equal slices of [t0, t1] x [f0, f1]; outside the recording is black.
    hop, df = t[1] - t[0], f[1] - f[0]
    ti = np.floor((t0 + (np.arange(w) + 0.5) * (t1 - t0) / w - t[0]) / hop + 0.5).astype(int)
    fi = np.floor((f1 - (np.arange(h) + 0.5) * (f1 - f0) / h - f[0]) / df + 0.5).astype(int)
    # Brightest within each output pixel's span, so lines thinner than a pixel survive.
    span_t = max(int(np.ceil((t1 - t0) / w / hop)), 1)
    span_f = max(int(np.ceil((f1 - f0) / h / df)), 1)
    img = np.zeros((h, w), np.float32)
    okt = (ti >= 0) & (ti < len(t))
    okf = (fi >= 0) & (fi < len(f))
    if okt.any() and okf.any():
        lo_t, hi_t = ti[okt].min(), min(ti[okt].max() + span_t, len(t))
        lo_f, hi_f = max(fi[okf].min() - span_f + 1, 0), fi[okf].max() + 1
        crop = score[lo_f:hi_f, lo_t:hi_t]
        crop = _resample(_resample(crop, int(okf.sum()), 0)[::-1], int(okt.sum()), 1)
        img[np.ix_(np.flatnonzero(okf), np.flatnonzero(okt))] = crop
    rgb = LUT[np.clip(img / vmax * 255, 0, 255).astype(np.uint8)]
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG", optimize=False)
    return Response(buf.getvalue(), media_type="image/png", headers={"Cache-Control": "max-age=3600"})


# --- passes ----------------------------------------------------------------------------


class HandTrace(BaseModel):
    points: list[tuple[float, float]]
    replace: str | None = None


@app.post("/api/projects/{name}/passes/measure")
def measure_pass(name: str, body: HandTrace):
    p = _project(name)
    if _busy(name):
        raise HTTPException(409, "wait for the running step to finish")
    try:
        return p.measure_by_hand(body.points, body.replace)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.patch("/api/projects/{name}/passes/{pid}")
def set_pass_status(name: str, pid: str, body: dict):
    p = _project(name)
    status = body.get("status")
    if status not in ("accepted", "rejected", "undecided"):
        raise HTTPException(400, "status: accepted, rejected or undecided")
    for rec in p.state["passes"]:
        if rec["id"] == pid:
            rec["status"] = status
            p.save()
            return rec
    raise HTTPException(404, f"no pass {pid}")


@app.delete("/api/projects/{name}/passes/{pid}")
def delete_pass(name: str, pid: str):
    p = _project(name)
    before = len(p.state["passes"])
    p.state["passes"] = [r for r in p.state["passes"] if not (r["id"] == pid and r["source"] == "manual")]
    if len(p.state["passes"]) == before:
        raise HTTPException(400, "only passes traced by hand can be deleted; reject detected ones")
    p.save()
    return {"ok": True}


# --- flights ---------------------------------------------------------------------------


@lru_cache(maxsize=4)
def _archive(path: str, mtime: float) -> Archive:
    return Archive(path)


@lru_cache(maxsize=1)
def _types() -> dict:
    return load_type_table(DATA / "types" / "icao_aircraft_types2.js")


@lru_cache(maxsize=16)
def _gaps(path: str, mtime: float) -> list:
    arch = _archive(path, mtime)
    who = np.repeat(np.arange(len(arch)), np.diff(arch.offsets))
    return [[a, b] for a, b in coverage_gaps(arch.cols["t"], who)]


def _adsb_gaps(p: Project) -> list:
    """Outages in the project's flight data (`adsb.coverage_gaps`), [start, end] unix s."""
    region = (p.state.get("adsb") or {}).get("region")
    if not region or not Path(region).exists():
        return []
    return _gaps(region, Path(region).stat().st_mtime)


@lru_cache(maxsize=16)
def _low(path: str, mtime: float, lat: float, lon: float, half_km: float) -> dict | None:
    arch = _archive(path, mtime)
    who = np.repeat(np.arange(len(arch)), np.diff(arch.offsets))
    c = arch.cols
    dem = TerrainTiles(DATA / "dem", zoom_for_spacing(10_000))
    args = c["lat"], c["lon"], c["alt_baro"], who, lat, lon, half_km
    try:
        return low_coverage(*args, dem.elevation)
    except OSError:  # no terrain tiles (offline): heights above sea level will do
        return low_coverage(*args, lambda la, lo: np.zeros(np.shape(la)))


def _adsb_low(p: Project) -> dict | None:
    """How well the project's flight data covers low flying over its area (`adsb.low_coverage`)."""
    if p.wide:  # mostly sea and other countries in a square that big: it says nothing about the recording
        return None
    region = (p.state.get("adsb") or {}).get("region")
    if not region or not Path(region).exists():
        return None
    s = p.settings
    return _low(region, Path(region).stat().st_mtime, float(s.lat), float(s.lon), float(s.half_km))


@lru_cache(maxsize=2)
def _flight_setup(name: str, region: str, mtime: float, settings: tuple):
    """The project's site and the flights its search uses, kept between clicks on the map."""
    p = _project(name)
    site = p.site()
    if p.wide:  # a whole country's flights only as arrays, made into tracks near each spot checked
        legs = p._legs(site)
        return site, (legs, QuietIndex(legs.samples()))
    return site, p._flights(site)


@app.post("/api/projects/{name}/spots")
def try_spot(name: str, body: dict):
    """Check a spot clicked on the map like any candidate (`Project.try_spot`): body {lat, lon}."""
    p = _project(name)
    if busy := _busy(name):
        raise HTTPException(409, f"{busy} is still running")
    if not p.state.get("adsb"):
        raise HTTPException(404, "get the flight data first")
    region = p.state["adsb"]["region"]
    s = p.settings
    site, keep = _flight_setup(name, region, Path(region).stat().st_mtime,
                               (s.lat, s.lon, s.half_km, s.audible_km, s.max_alt_m, s.mic_height))
    try:
        if p.wide:
            keep = p.near_tracks(*keep, site, float(body["lat"]), float(body["lon"]))
        results = p.try_spot(float(body["lat"]), float(body["lon"]), keep=keep, site=site)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return Response(to_json(results), media_type="application/json")


@app.post("/api/projects/{name}/spots/keep")
def keep_spot(name: str, body: dict):
    """Keep a spot tried on the map or, with keep false, drop it: body {key, keep}."""
    p = _project(name)
    if busy := _busy(name):
        raise HTTPException(409, f"{busy} is still running")
    try:
        results = p.keep_spot(str(body.get("key")), bool(body.get("keep", True)))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return Response(to_json(results), media_type="application/json")


@app.get("/api/projects/{name}/audio")
def audio(name: str):
    """The recording's file, as it is, for the browser's player (which asks for ranges to seek)."""
    path = Path(_project(name).settings.audio)
    if not path.is_file():
        raise HTTPException(404, f"no audio file at {path}")
    return FileResponse(path, media_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream")


@app.get("/api/projects/{name}/aircraft")
def aircraft(name: str, t0: float, t1: float, step: float = 5.0):
    """Every aircraft over the area searched plus the distance a plane is heard from, between unix
    times t0 and t1 (at most 6 h), as reported: [t, lat, lon, altitude m, ground speed m/s] about
    every `step` s, for the Flights page to replay."""
    p = _project(name)
    if not p.state.get("adsb"):
        raise HTTPException(404, "get the flight data first")
    if not 0 < t1 - t0 <= 6 * 3600:
        raise HTTPException(400, "t0 < t1, at most 6 h apart")
    path = Path(p.state["adsb"]["region"])
    arch, types, s = _archive(str(path), path.stat().st_mtime), _types(), p.settings
    reach = s.half_km + s.audible_km
    dlat, dlon = reach / 111.2, reach / (111.2 * np.cos(np.radians(s.lat)))
    out = []
    for i, meta in enumerate(arch.meta):
        pts = arch.points(i)
        t = pts["t"]
        if not t.size or t.max() < t0 or t.min() > t1:
            continue
        ok = ((t >= t0) & (t <= t1) & np.isfinite(pts["lat"]) & (np.abs(pts["lat"] - s.lat) <= dlat)
              & (np.abs(pts["lon"] - s.lon) <= dlon))
        if ok.sum() < 2:
            continue
        q = {k: v[ok] for k, v in pts.items()}
        order = np.argsort(q["t"], kind="stable")
        q = {k: v[order] for k, v in q.items()}
        b = np.floor(q["t"] / step)
        q = {k: v[np.r_[True, b[1:] != b[:-1]]] for k, v in q.items()}  # one report per step
        alt = np.where(np.isfinite(q["alt_geom"]), q["alt_geom"], q["alt_baro"]) * 0.3048
        alt = np.where(np.nan_to_num(q["alt_baro"], nan=1.0) <= 0, 0.0, np.nan_to_num(alt))  # on the ground
        gs = np.nan_to_num(q["gs"]) * 0.514444
        out.append(dict(icao=meta.get("icao"), reg=meta.get("r"), type=meta.get("t"), desc=meta.get("desc"),
                        jet=is_jet(meta.get("t"), types),
                        pts=[[round(float(a), 1), round(float(b_), 5), round(float(c), 5), round(float(d)), round(float(e))]
                             for a, b_, c, d, e in zip(q["t"], q["lat"], q["lon"], alt, gs)]))
    return dict(t0=t0, t1=t1, aircraft=out)


# --- results ---------------------------------------------------------------------------


@app.get("/api/projects/{name}/results")
def results(name: str):
    p = _project(name)
    return Response(to_json(_ranked_results(p)), media_type="application/json")


def _ranked_results(p: Project) -> dict:
    """The project's results, its heard flights graded (results from before grading get it now, and
    keep it) and the candidates ranked with the verdicts given on them (`project.rank_candidates`)."""
    path = p.folder / "results.json"
    if not path.exists():
        raise HTTPException(404, "run the search first")
    res = read_json(path)
    ungraded = [h for c in res["candidates"] for h in c.get("heard") or [] if "grade" not in h and h.get("curve_t")]
    if ungraded:
        spec = p.spectrogram()
        for h in ungraded:
            h.update(curve_visibility(spec, h["curve_t"], h["curve_f"]))
        write_atomic(path, to_json(res))
    return rank_candidates(res, p.state)


@app.post("/api/projects/{name}/heard")
def heard_verdict(name: str, body: dict):
    """Whether a flight heard at a candidate shows on the spectrogram, judged by hand: body {key
    (the candidate's), flight, status: "accepted" | "undecided" | "rejected" | null (back to its
    automatic grade)}. Returns the results, ranked again."""
    return _hand_verdict(_project(name), "heard_verdicts", str(body.get("key")), str(body.get("flight")), body.get("status"))


@app.post("/api/projects/{name}/fit")
def fit_verdict(name: str, body: dict):
    """Whether a flight's modelled curve fits a pass at a candidate, judged by hand: body {key (the
    candidate's), pass_id, flight, status: "accepted" | "undecided" | "rejected" | null (back to the
    share of the line it follows)}. Only that candidate changes. Returns the results, ranked again."""
    item = fit_key(body.get("pass_id"), body.get("flight"))
    return _hand_verdict(_project(name), "fit_verdicts", str(body.get("key")), item, body.get("status"))


def _hand_verdict(p: Project, table: str, key: str, item: str, status) -> Response:
    if status not in ("accepted", "undecided", "rejected", None):
        raise HTTPException(400, "status: accepted, undecided, rejected or null")
    mine = p.state.setdefault(table, {}).setdefault(key, {})
    if status is None:
        mine.pop(item, None)
    else:
        mine[item] = status
    if not mine:
        p.state[table].pop(key)
    p.save()
    return Response(to_json(_ranked_results(p)), media_type="application/json")


@app.post("/api/projects/{name}/verdicts")
def verdict(name: str, body: dict):
    p = _project(name)
    key, value = str(body.get("key")), body.get("verdict")
    if value not in ("right", "wrong", None):
        raise HTTPException(400, "verdict: right, wrong or null")
    verdicts = p.state.setdefault("verdicts", {})
    if value is None:
        verdicts.pop(key, None)
    else:
        verdicts[key] = value
    p.save()
    return verdicts


class _Revalidated(StaticFiles):
    """Static files the browser must check before reusing (a quick 304 when unchanged): without it,
    it guesses from their age and kept running an old app.js for a while after an update."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


app.mount("/static", _Revalidated(directory=STATIC), name="static")
