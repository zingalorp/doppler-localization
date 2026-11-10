"""A recording being located: its settings, measured passes and results, kept in one folder.

Each long step (prepare, detect, fetch, search) can run in its own process and reports progress
to jobs/<step>.json, which is how the GUI drives it:

    uv run python -m doppler_loc.project search data/projects/oyster-bay
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import partial
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from .adsb import Archive, download_day, extract_region, is_jet, load_type_table
from .area import AreaMask
from .geo import FT
from .audio import load_audio
from .detect import assess, close_to_found, detect, harmonic_groups, measure
from .measure import Event, fit_flyby, sharpen_with_harmonics, straight_flyby, trace_by_hand
from .physics import closest_approach, doppler_factor
from .search import (
    Passes, explain_passes, observer_grid, refine_many, search_event_pairs, search_known_starts, vote_passes,
)
from .site import Site
from .spectro import Spectrogram, compute_spectrogram
from .stack import refine_place, spectrogram_peaks, stack_at, stack_search
from .visible import curve_visibility
from .wide import Legs, QuietIndex, search_pairs_wide

STACK_CAP, STACK_FLOOR = 6.0, 2.0  # each heard flight adds at most cap - floor (see `stack.contribution`)
# How many flight pairs (or flights, for one pass) each search keeps for refining. Without a start, a
# wide area has thousands of pairs that fit by chance: over 500 km on a 1 km grid the video test's true
# pair ranked 2nd by evidence but 5th by curve score, and on a 500 m grid it fell outside the top 3 either way.
KEEP_START_KNOWN, KEEP_NO_START = 10, 30
REFINE_START_TOL = 10.0  # s either side of the start a pass's search found, when refining it
# Candidates resting on the same evidence are one answer (`Project._group`) when this close, in m and s.
GROUP_DIST_M, GROUP_START_S = 1500.0, 8.0
# From this far each way an area is wide: flight data is cut down to what can be heard (a day's
# flying worldwide would not fit), and with a known start only pairs of passes are searched, flights
# first, since the searches that need a grid over the whole area take hours.
WIDE_HALF_KM = 500.0
WIDE_PASSES = 4  # passes whose pairs search a wide area, the most reliable: 10 would be 45 pairs. All are still checked.
MAX_SURVIVORS = 20000  # of a pair's surviving points, sent to the map: tens of thousands of dots slow it down

# Worker processes for a step: half the logical cores, so a search leaves the machine usable;
# DOPPLER_WORKERS overrides it (e.g. the cores' count, for a machine left alone).
WORKERS = max(1, int(os.environ.get("DOPPLER_WORKERS", 0)) or (os.cpu_count() or 2) // 2)


def lower_priority():
    """Run this process (and the workers it starts, which inherit it) below normal priority, so
    whatever the user is doing comes first."""
    try:
        if sys.platform == "win32":
            import ctypes
            BELOW_NORMAL_PRIORITY_CLASS = 0x4000
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.GetCurrentProcess.restype = ctypes.c_void_p  # a 64-bit handle, which ctypes would cut to 32
            k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            if not k32.SetPriorityClass(k32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS):
                raise ctypes.WinError(ctypes.get_last_error())
        else:
            os.nice(10)
    except OSError:
        pass

DATA = Path(os.environ.get("DOPPLER_DATA", Path(__file__).resolve().parent.parent / "data"))
STEPS = ("prepare", "detect", "fetch", "search", "explain")


@dataclass
class Settings:
    name: str
    audio: str  # path to any audio or video file FFmpeg reads
    date: str  # UTC day of the recording, YYYY-MM-DD (the start time is found by the search)
    lat: float  # centre of the area to search
    lon: float
    half_km: float = 25.0  # the area is a square this far from the centre each way
    spacing_m: float = 500.0
    mic_height: float = 2.0
    audible_km: float = 10.0  # farthest a flight is looked for
    max_alt_m: float = 6000.0  # flights above this are not heard
    truth_lat: float | None = None  # where it was really recorded, for tests
    truth_lon: float | None = None
    start_unix: float | None = None  # when the recording started (UTC, seconds since 1970), if known
    start_tol_s: float = 120.0  # ...to within this many seconds (recorder clocks drift: 75 s on eclipse-281)
    area: list | None = None  # a shape to search instead of the whole square: rings of [lat, lon] (`area.py`)
    area_name: str | None = None  # what it is, when it came from a map of countries
    search_mode: str = "thorough"  # "fast" leaves out the pairs of a straight pass and a turning one: minutes instead of an hour over a country


def _free_id(taken, prefix) -> str:
    n = 1
    while f"{prefix}{n}" in taken:
        n += 1
    return f"{prefix}{n}"


def utc(t) -> str:
    return datetime.fromtimestamp(float(t), timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def _finite(obj):
    """`obj` with every NaN or infinity replaced by None."""
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    if isinstance(obj, np.generic):
        return _finite(obj.item())
    return obj


def to_json(obj, **kw) -> str:
    """Strict JSON: Python would write NaN and Infinity, which browsers refuse to parse (an empty
    pair of passes has evidence -inf, for one), so those become null."""
    return json.dumps(_finite(obj), allow_nan=False, **kw)


def _retry(fn, tries=50):
    """Windows refuses to replace, and briefly to open, a file another process has open (the GUI
    polls a step's status while the step rewrites it); such a clash lasts milliseconds, so wait."""
    for i in range(tries):
        try:
            return fn()
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(0.02 + 0.01 * i)


def write_atomic(path, text: str):
    """Write `text` to `path` so that readers see the old file or the new one, never half of it."""
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    _retry(lambda: os.replace(tmp, path))


def read_json(path):
    return json.loads(_retry(lambda: Path(path).read_text(encoding="utf-8")))


def candidate_key(c) -> str:
    """A candidate's name across rankings (the GUI's verdicts on candidates use it too)."""
    return f"{c['pair']}@{c['lat']},{c['lon']}"


# A heard flight's curve, graded (`visible.curve_visibility`) or judged by hand, as a pass's status.
GRADE_STATUS = {"visible": "accepted", "unsure": "undecided", "absent": "rejected"}
EXPLAINED_AT = 0.6  # a pass is explained when a flight follows this share of its traced line
# A flight heard on the spectrogram but following no traced line is weaker evidence than one that does.
# At half, against a whole, every search with a known spot keeps its answer but for the spot's own
# place within it, while at a whole, unchecked flights outvote traced passes (seen on one test site,
# once a flight may not count twice).
HEARD_WEIGHT = 0.5
# How a candidate's confidence grows with its lead and the share of accepted passes it explains
# (`_add_confidence`). Fitted on all candidates of 19 searches of 13 recordings with known spots; with
# each recording held out in turn, all 41 candidates above 90% were right and none of 529 below 10%,
# but three of the 19 top candidates were wrong, at 23%, 47% and 89%.
CONF_BASE, CONF_LEAD, CONF_EXPLAINED = -3.3, 3.0, 5.9


def fit_key(pass_id, flight) -> str:
    """A hand verdict's name within a candidate: it is on this flight's curve for this pass, so it
    lapses if a re-check puts another flight on the pass."""
    return f"{pass_id}|{flight}"


def rank_candidates(results, state: dict):
    """Rank candidates by passes explained plus flights heard there (`_explain`).

    A pass counts the share of its line the flight follows, unless its curve was judged by hand at
    this candidate: in full when it fits, half that share when unsure, nothing when it doesn't. A
    heard flight counts `HEARD_WEIGHT` by whether its curve shows on the spectrogram: all of that
    when it does (graded "visible"), half its stacked evidence when unsure, nothing when it doesn't,
    and in full if a person says it shows. Results from before grading count each by its stacked
    evidence. One flight is one piece of evidence: one that explains a pass, or whose curve for a
    pass has been judged by hand, counts as that alone.
    From the project's `state`, verdicts being "accepted" | "undecided" | "rejected":
    `heard_verdicts`: {candidate_key: {flight id: verdict}}, `fit_verdicts`: {candidate_key: {`fit_key`: verdict}}."""
    full = STACK_CAP - STACK_FLOOR
    heard_verdicts, fit_verdicts = state.get("heard_verdicts", {}), state.get("fit_verdicts", {})
    for c in results["candidates"]:
        c["key"] = candidate_key(c)
        c.setdefault("rank_found", c.get("rank"))
        if "support" not in c:
            continue
        fits = fit_verdicts.get(c["key"], {})
        for x in c.get("explains") or []:
            share = x.get("share") or 0.0
            x["verdict"] = fits.get(fit_key(x["pass_id"], x["flight"])) if x.get("flight") else None
            auto = "accepted" if share >= EXPLAINED_AT else "undecided" if share > 0 else "rejected"
            x["status"] = (x["verdict"] or auto) if x.get("flight") else None
            x["counts"] = {"accepted": 1.0, "undecided": 0.5 * share, "rejected": 0.0}.get(x["verdict"], share)
        c["fit_counted"] = round(sum(x["counts"] for x in c.get("explains") or []), 2)
        c["explained"] = sum(x["status"] == "accepted" for x in c.get("explains") or [])
        mine = heard_verdicts.get(c["key"], {})
        as_pass = {x["flight"] for x in c.get("explains") or []
                   if x.get("flight") and (x["verdict"] or x["status"] == "accepted")}
        for h in c.get("heard") or []:
            h["verdict"] = mine.get(h["flight"])
            h["status"] = h["verdict"] or GRADE_STATUS.get(h.get("grade"))
            z = h.get("z", 0.0)
            w = {"accepted": full, "undecided": 0.5 * z, "rejected": 0.0}.get(h["status"], z)
            h["counts"] = (0.0 if h["flight"] in as_pass else 1.0 if h["verdict"] == "accepted"
                           else round(HEARD_WEIGHT * w / full, 2))
        c["heard_counted"] = round(sum(h["counts"] for h in c.get("heard") or []), 2)
        c["combined"] = round(c["fit_counted"] + c["heard_counted"], 2)
    results["candidates"].sort(key=lambda c: (-(c.get("combined") or 0), -(c.get("pair_evidence") or 0),
                                              -(c.get("score") or 0)))
    for k, c in enumerate(results["candidates"]):
        c["rank"] = k + 1
    _add_confidence(results["candidates"], state)
    return results


def _add_confidence(cands, state):
    """How likely each candidate (ranked, best first) is to be within about a kilometre of the right
    spot. Two things tell right from wrong on the test recordings: its lead over the best candidate
    elsewhere, and the share of the accepted passes it explains. How closely the curves fit adds
    nothing to those. Spots can't all be right, so each is then discounted by the chance that one
    ranked above it, elsewhere, is. Hand verdicts count through both, so they change it at once."""
    scored = [c for c in cands if c.get("combined") is not None]
    if not scored:
        return
    accepted = {p["id"] for p in state.get("passes", []) if p["status"] == "accepted"}
    lat, lon = np.array([c["lat"] for c in scored]), np.array([c["lon"] for c in scored])
    km_lon = 111.2 * np.cos(np.radians(lat.mean()))
    apart = np.hypot((lat[:, None] - lat) * 111.2, (lon[:, None] - lon) * km_lon) > GROUP_DIST_M / 1000
    score = np.array([c["combined"] for c in scored])
    lead = score - (score * apart).max(axis=1)
    share = []
    for c in scored:
        mine = [x for x in c.get("explains") or [] if x["pass_id"] in accepted]
        share.append(sum(x["status"] == "accepted" for x in mine) / max(1, len(mine)))
    p = 1.0 / (1.0 + np.exp(-(CONF_BASE + CONF_LEAD * lead + CONF_EXPLAINED * np.array(share))))
    for k, c in enumerate(scored):
        c["confidence"] = round(float(p[k] * np.prod(1.0 - p[:k][apart[k, :k]])), 3)


def _r(x, nd=2):
    return [round(float(v), nd) for v in x]


def _stored(path: str) -> str:
    """How project.json keeps a path: "data/..." for one under DATA, so a checkout can move."""
    try:
        return "data/" + Path(path).resolve().relative_to(DATA.resolve()).as_posix()
    except ValueError:
        return path


def _local(path: str) -> str:
    """A path from project.json on this computer: "data/..." under DATA, and an absolute path from
    another computer's checkout (".../data/...") moved under this one's DATA when it is there."""
    parts = Path(path.replace("\\", "/")).parts
    if parts[:1] == ("data",):
        return str(DATA.joinpath(*parts[1:]))
    if not Path(path).exists() and "data" in parts:
        moved = DATA.joinpath(*parts[len(parts) - parts[::-1].index("data"):])
        if moved.exists():
            return str(moved)
    return path


class Project:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.state = read_json(self.folder / "project.json")
        self.state["settings"]["audio"] = _local(self.state["settings"]["audio"])
        if self.state.get("adsb"):
            self.state["adsb"]["region"] = _local(self.state["adsb"]["region"])
            if not Path(self.state["adsb"]["region"]).exists():  # a fresh checkout: step 3 cuts it again
                self.state["adsb"] = None
        if self.state["settings"].pop("fast", False):  # a project from when a 1 km grid had its own switch
            self.state["settings"]["spacing_m"] = 1000.0
        self.settings = Settings(**self.state["settings"])
        self._renumber_duplicates()

    def _renumber_duplicates(self):
        """Find passes used to reuse the names of passes retraced by hand. A pass's name is how
        everything finds it, so a detected one sharing a name gets a new one, and groups are made anew."""
        passes, taken = self.state.get("passes", []), set()
        for p in passes:
            if p["source"] == "manual":
                taken.add(p["id"])
        changed = False
        for p in passes:
            if p["source"] == "manual":
                continue
            if p["id"] in taken:
                p["id"] = _free_id(taken, "P")
                changed = True
            taken.add(p["id"])
        if changed:
            self.group_harmonics()

    @classmethod
    def create(cls, settings: Settings, root=DATA / "projects") -> Project:
        folder = Path(root) / settings.name
        folder.mkdir(parents=True, exist_ok=False)
        state = dict(settings=asdict(settings), duration=None, passes=[], adsb=None, verdicts={})
        state["settings"]["audio"] = _stored(settings.audio)
        (folder / "project.json").write_text(to_json(state, indent=1))
        return cls(folder)

    def save(self):
        self.state["settings"] = asdict(self.settings)
        state = {**self.state, "settings": {**self.state["settings"], "audio": _stored(self.settings.audio)}}
        if state.get("adsb"):
            state["adsb"] = {**state["adsb"], "region": _stored(state["adsb"]["region"])}
        write_atomic(self.folder / "project.json", to_json(state, indent=1))

    # --- shared pieces -------------------------------------------------------------------

    def spectrogram(self) -> Spectrogram:
        d = np.load(self.folder / "spec.npz")
        return Spectrogram(d["t"], d["f"], d["db"], d["score"])

    def site(self) -> Site:
        s = self.settings
        return Site(s.lat, s.lon, DATA / "dem", mic_height=s.mic_height, wide=self.wide)

    @property
    def wide(self) -> bool:
        return self.settings.half_km >= WIDE_HALF_KM

    def _area_mask(self, site) -> AreaMask | None:
        """The shape searched, if it is not the whole square."""
        s = self.settings
        return AreaMask(s.area, site.frame, s.half_km * 1000) if s.area else None

    def pass_event(self, p) -> Event:
        return Event(t_heard=p["t_heard"], f0=p["f0"], speed=p["speed"], dist=p["dist"], speed_tol=p["speed_tol"],
                     dist_tol=p["dist_tol"], line=(np.array(p["line_t"]), np.array(p["line_f"])))

    def _pass_record(self, ident, source, ev: Event, strength=None, harmonics=(), t_range=None, status=None):
        # A turning flight is fitted to lines traced by hand: that is how turning planes get measured
        # (on the labelled set, detected lines a turn fitted and no straight pass did were all false).
        a = assess(ev, strength, harmonics, max_dist=self.settings.audible_km * 1000, turns=source == "manual")
        t_line = ev.line[0]
        lo, hi = t_range if t_range else (float(t_line.min()), float(t_line.max()))
        t = np.linspace(lo - 20, hi + 20, 200)
        if a.turn is not None:  # draw the turning flight that follows the line, not the straight pass,
            t = np.linspace(lo - 5, hi + 5, 200)  # and only where it was traced: a turn can't be extrapolated
            curve = a.turn.curve(t)
            t, curve = t[np.isfinite(curve)], curve[np.isfinite(curve)]
        else:
            curve = straight_flyby(t, ev.f0, ev.speed, ev.dist, ev.t_heard)
        default = {"likely aircraft": "accepted", "not a pass": "rejected"}.get(a.kind, "undecided")
        return dict(
            id=ident, source=source, status=status or default, kind=a.kind, assessed=a.kind,
            confidence=round(a.confidence, 2), notes=a.notes, t_heard=ev.t_heard,
            f0=a.turn.params[0] if a.turn else ev.f0, speed=ev.speed, dist=ev.dist, speed_tol=a.speed_tol,
            dist_tol=a.dist_tol, strength=strength, harmonics=list(harmonics), t_range=[lo, hi],
            line_t=_r(ev.line[0]), line_f=_r(ev.line[1]),
            curve_t=_r(t), curve_f=_r(curve),
        )

    def measure_by_hand(self, points, replace=None) -> dict:
        """Trace and fit a pass along hand-placed (t, f) points, as a new pass or replacing one."""
        pts = sorted((float(t), float(f)) for t, f in points)
        if len({t for t, _ in pts}) < 2:
            raise ValueError("place at least two points along the curve, at different times")
        spec = self.spectrogram()
        t_range = (pts[0][0], pts[-1][0])
        line_t, line_f, strength = trace_by_hand(spec, pts)
        line_f, ratios = sharpen_with_harmonics(spec, line_t, line_f, strength)
        ev = fit_flyby(line_t, line_f, strength)
        ident = replace or _free_id({p["id"] for p in self.state["passes"]}, "M")
        rec = self._pass_record(ident, "manual", ev, harmonics=ratios, t_range=t_range, status="accepted")
        rec["guide"] = pts
        rec["traced_at"] = time.time()  # newer than the results' `checked_at`: they want a re-check
        self.state["passes"] = [p for p in self.state["passes"] if p["id"] != ident] + [rec]
        self.group_harmonics()
        self.save()
        return next(p for p in self.state["passes"] if p["id"] == ident)

    def group_harmonics(self):
        """Mark lines that are harmonics of another pass, so one aircraft is not counted twice.

        In each group (see `detect.harmonic_groups`) the main pass is one you accepted, else the most
        confident; the others become kind "harmonic", rejected, with a note saying whose and at what
        pitch ratio. Run again after any change: earlier marks are cleared first.
        """
        passes = self.state["passes"]
        for p in passes:
            if p["kind"] == "harmonic":
                p["kind"] = p.get("assessed", "possible")
                p["status"] = "undecided"
            p["notes"] = [n for n in p["notes"] if not n.startswith(("harmonic of ", "also heard at ", "also traced as "))]
            p.pop("harmonic_of", None)
        trust = sorted(range(len(passes)), key=lambda i: (passes[i]["status"] != "accepted", -passes[i]["confidence"],
                                                          passes[i]["f0"]))
        for main_i, *others in harmonic_groups([(p["line_t"], p["line_f"]) for p in passes], trust):
            main = passes[main_i]
            for i in others:
                p = passes[i]
                ratio = _line_ratio(main, p)
                p.update(kind="harmonic", status="rejected", harmonic_of=main["id"])
                if abs(np.log(ratio)) < 0.02:
                    p["notes"].insert(0, f"harmonic of {main['id']}: the same line traced again")
                    main["notes"].append(f"also traced as {p['id']}")
                else:
                    p["notes"].insert(0, f"harmonic of {main['id']}: {ratio:.3g}x its pitch, the same aircraft")
                    main["notes"].append(f"also heard at {ratio:.3g}x its pitch ({p['id']})")

    # --- steps ---------------------------------------------------------------------------

    def prepare(self, report):
        report("decoding audio", 0.05)
        audio = load_audio(self.settings.audio)
        report(f"{len(audio) / 8000 / 60:.0f} min of audio; computing spectrogram", 0.5)
        spec = compute_spectrogram(audio, 8000, fmin=20, fmax=1000)
        part = self.folder / "spec.part.npz"  # whole or not at all, should the step be stopped while saving
        np.savez(part, t=spec.t, f=spec.f, db=spec.db, score=spec.score)
        _retry(lambda: os.replace(part, self.folder / "spec.npz"))
        self.state["duration"] = float(spec.t[-1])
        self.save()
        report("spectrogram ready", 1.0)

    def detect(self, report, max_events=15):
        report("matching flyby shapes", 0.05)
        spec = self.spectrogram()
        dets = detect(spec, max_events=max_events, workers=WORKERS)
        out, taken = [], {p["id"] for p in self.state["passes"] if p["source"] == "manual"}  # retraced ones keep their names
        for k, det in enumerate(dets):
            report(f"measuring pass {k + 1} of {len(dets)}", 0.5 + 0.5 * k / max(len(dets), 1))
            ev = measure(spec, det)
            line_f, ratios = sharpen_with_harmonics(spec, *ev.line)
            if len(ratios) >= 2:
                ev = fit_flyby(ev.line[0], line_f)
            ident = _free_id(taken, "P")
            taken.add(ident)
            out.append(self._pass_record(ident, "detected", ev, det.strength, sorted({*det.harmonics, *ratios}), det.t_range))
        report("looking for curves hidden by those found", 0.8)
        close = close_to_found(spec, [(p["line_t"], p["line_f"]) for p in out], workers=WORKERS)
        for det, ev, i in close:
            ident = _free_id(taken, "P")
            taken.add(ident)
            line_f, ratios = sharpen_with_harmonics(spec, *ev.line)
            if len(ratios) >= 2:
                ev = fit_flyby(ev.line[0], line_f)
            rec = self._pass_record(ident, "detected", ev, det.strength, sorted({*det.harmonics, *ratios}), det.t_range)
            if rec["status"] == "accepted":  # hidden until now, so not one to search with unchecked
                rec["status"] = "undecided"
            rec["notes"].append(f"close to {out[i]['id']}: found on what that line left, so check it by eye")
            out.append(rec)
        self.state["passes"] = out + [p for p in self.state["passes"] if p["source"] == "manual"]
        self.group_harmonics()
        self.save()
        same = sum(p["kind"] == "harmonic" for p in self.state["passes"])
        report(f"{len(out)} passes found" + (f", {same} of them harmonics of another" if same else "") + (f", {len(close)} close to another" if close else ""), 1.0)

    def fetch(self, report):
        s = self.settings
        day_dir = DATA / "adsb" / s.date
        report("downloading the day's ADS-B archive", 0.0)
        parts = download_day(s.date, day_dir, progress=lambda done, total: report(
            f"downloading ADS-B: {done / 1e9:.1f} of {total / 1e9:.1f} GB", 0.6 * done / total))
        radius = s.half_km * 2**0.5 + s.audible_km + 15
        out = day_dir / f"region_{s.lat:.3f}_{s.lon:.3f}_{radius:.0f}km.npz"
        if not out.exists():
            types = load_type_table(DATA / "types" / "icao_aircraft_types2.js") if self.wide else None
            part = out.with_name(out.stem + ".part.npz")  # whole or not at all, should the step be stopped
            extract_region(parts, s.lat, s.lon, radius, part, progress=lambda n, kept: report(
                f"cutting out the area: {n} aircraft scanned, {kept} kept", 0.6 + 0.4 * min(n / 60000, 0.99)),
                audible_only=(types, (s.max_alt_m + 300.0) / FT) if self.wide else None)
            _retry(lambda: os.replace(part, out))
        arch = Archive(out)
        self.state["adsb"] = dict(region=str(out), radius_km=radius, aircraft=len(arch))
        self.save()
        report(f"{len(arch)} aircraft came within {radius:.0f} km that day", 1.0)

    def search(self, report, n_refine=None):
        """Search the area with the passes.

        Without a known start, every pair of accepted passes is searched (the time between them is
        what constrains the place). With one (`start_unix` +- `start_tol_s`), pairs must also have
        been heard at that time, each accepted pass is searched on its own as well, and every pass
        not rejected votes for the places that explain it (`search.vote_passes`), so that real
        passes left undecided count too and false ones scatter, and the spectrogram itself is
        stacked along every flight's curve (`stack.stack_search`), which needs no passes at all.
        Candidates are then ranked by how many of the passes they explain (the accepted ones, or all
        voters when there was a vote) plus how many flights are heard there on the spectrogram.
        """
        s = self.settings
        chosen = [p for p in self.state["passes"] if p["status"] == "accepted"]
        start = (s.start_unix, s.start_tol_s) if s.start_unix is not None else None
        if n_refine is None:
            n_refine = KEEP_START_KNOWN if start else KEEP_NO_START
        path = self.folder / "results.json"
        kept = [c for c in read_json(path)["candidates"] if c.get("kept")] if path.exists() else []
        scored = self._scorable() if start else []  # matched with flights; the ones not rejected vote
        if len(chosen) < 2 and not start:
            raise ValueError("accept at least two passes first, or set the recording's start time")
        report("loading flights", 0.02)
        spec, site = self.spectrogram(), self.site()
        events = [self.pass_event(p) for p in chosen]
        if start is None or self.wide:
            return self._search_flights_first(report, spec, site, chosen, events, start, n_refine, kept)
        keep = self._flights(site)
        report(f"{len(keep)} propeller or unknown flights near the area; placing the grid", 0.05)
        def terrain(done, total):
            report(f"downloading terrain: {done} of {total} tiles", 0.05)

        site.on_download(terrain)
        grid = observer_grid((0.0, 0.0), s.half_km * 1000, s.spacing_m, place=site.for_spacing(s.spacing_m).place)

        t_start = time.time()
        passes = Passes(keep, grid, s.audible_km * 1000, workers=WORKERS)
        report("closest approaches done", 0.3)
        found = []
        if len(chosen) >= 2:
            def progress(done, total):
                if done:
                    report(f"searched {done} of {total} pairs of passes", 0.3 + 0.3 * done / total)

            found = search_event_pairs(spec, keep, grid, events, spec.t[-1], audible_dist=s.audible_km * 1000,
                                       offset_tol=8.0, shift_tol=6.0, silence_dist=2500.0, silence_between_only=True,
                                       workers=WORKERS, progress=progress, passes=passes, start=start)
        turning = [p["kind"] == "possibly turning" for p in chosen]
        all_points = grid
        if s.search_mode != "fast" and any(turning) and not all(turning):
            # A turning pass's straight-pass numbers rule out its real flight above: judged by its line instead.
            report("searching the pairs of a straight pass and a turning one", 0.6)
            legs = Legs.from_tracks(keep)
            site.on_download(None)  # the site goes to the workers, which can't take a local function
            anchored, extra, _ = search_pairs_wide(
                spec, legs, events, site.for_spacing(s.spacing_m), half=s.half_km * 1000, spacing=s.spacing_m, workers=WORKERS, start=start, progress=terrain,
                audible_dist=s.audible_km * 1000, f0_tols=[0.3 if t else 0.15 for t in turning], area=self._area_mask(site), turning=turning, anchored_only=True)
            site.on_download(terrain)
            for ep in anchored:
                ep.result.point = ep.result.point + len(grid)
            found += anchored
            all_points = np.concatenate([grid, extra])
        singles = []
        if start and events:
            report(f"searching each of the {len(events)} passes on its own at the known time", 0.6)
            trust_fit = [p["kind"] != "possibly turning" for p in chosen]
            singles = list(enumerate(search_known_starts(spec, keep, grid, events, start, passes=passes,
                                                         use_fit=trust_fit, workers=WORKERS)))
        votes = None
        if scored:
            voting = [p["status"] != "rejected" for p in scored]
            report(f"{sum(voting)} passes voting for places at the known start, {len(scored)} matched with flights",
                   0.7)
            scored_events = [self.pass_event(p) for p in scored]
            # A turning plane's straight-pass numbers are unreliable: look for its flight more widely.
            turning = np.array([p["kind"] == "possibly turning" for p in scored])
            votes = (vote_passes(keep, grid, scored_events, start, passes=passes, voting=voting, use_fit=~turning,
                                 f0_tol=np.where(turning, 0.3, 0.15), measure_tol=np.where(turning, 20.0, 8.0),
                                 workers=WORKERS), scored, scored_events)
        del passes  # the most memory a search holds, and nothing below needs it
        stack = None
        if start:
            report(f"stacking the spectrogram along each of {len(keep)} flights' curves", 0.72)
            peaks = spectrogram_peaks(spec)
            coarse = observer_grid((0.0, 0.0), s.half_km * 1000, 1000.0, place=site.for_spacing(1000.0).place)
            sm = stack_search(keep, coarse, peaks, start, passes=Passes(keep, coarse, s.audible_km * 1000,
                                                                         workers=WORKERS),
                              cap=STACK_CAP, floor=STACK_FLOOR, workers=WORKERS)
            stack = (sm, coarse, peaks)
        report("refining the best candidates", 0.8)
        results = self._results(spec, site, all_points, keep, chosen, events, found, n_refine, singles, votes, stack, anchored_apart=True)
        report("checking how many passes each candidate explains", 0.92)
        self._explain(results, site, keep, *self._explain_set(results, chosen))
        self._group(results, spec, site, keep, *self._explain_set(results, chosen), report=report)
        if kept:  # spots kept by hand (`try_spot`) stay, checked against the passes as they are now
            self._explain(dict(candidates=kept, stack=results.get("stack")), site, keep, *self._explain_set(results, chosen))
            results["candidates"] += kept
            rank_candidates(results, self.state)
        results["seconds"] = round(time.time() - t_start)
        write_atomic(self.folder / "results.json", to_json(results))
        report(f"{len(results['candidates'])} candidate locations", 1.0)

    def _search_flights_first(self, report, spec, site, chosen, events, start, n_refine, kept):
        """Search with pairs of passes alone, flights first (`wide`). Its time and memory follow the
        flights that could have made the passes, not the area or the grid."""
        s = self.settings
        t_start = time.time()
        if len(chosen) < 2:
            raise ValueError("over an area this wide, accept at least two passes first")
        legs = self._legs(site)
        report(f"{len(legs)} propeller or unknown flights near the area; comparing them", 0.1)

        def step(done, total):
            report(f"searching pair {done + 1} of {total} of passes", 0.1 + 0.6 * done / total)

        def terrain(done, total):
            report(f"downloading terrain: {done} of {total} tiles", 0.1)

        searched = sorted(range(len(chosen)), key=lambda k: (chosen[k]["kind"] == "possibly turning", -chosen[k]["confidence"]))
        pairs_of = sorted(searched[:WIDE_PASSES]) if self.wide else None
        found, grid, (samples, index) = search_pairs_wide(
            spec, legs, events, site.for_spacing(s.spacing_m), half=s.half_km * 1000, spacing=s.spacing_m, workers=WORKERS,
            report=step, progress=terrain, start=start, audible_dist=s.audible_km * 1000,
            f0_tols=[0.3 if p["kind"] == "possibly turning" else 0.15 for p in chosen], pairs_of=pairs_of, area=self._area_mask(site),
            turning=None if s.search_mode == "fast" else [p["kind"] == "possibly turning" for p in chosen])
        # From here on only the flights of the best pairs matter, and those near what they find.
        best = sorted({int(f) for ep in found for f in np.r_[ep.result.a[ep.rows[:n_refine]], ep.result.b[ep.rows[:n_refine]]]})
        number = np.full(len(legs), -1)
        number[best] = np.arange(len(best))
        for ep in found:
            ep.result.a, ep.result.b = number[ep.result.a], number[ep.result.b]
        report("refining the best candidates", 0.75)
        results = self._results(spec, site, grid, [legs.track(i) for i in best], chosen, events, found, n_refine)
        report("checking how many passes each candidate explains", 0.9)
        keep = self._near_candidates(legs, index, results["candidates"] + kept, best)
        ps, evs = self._explain_set(results, chosen)
        self._explain(results, site, keep, ps, evs)
        self._group(results, spec, site, keep, ps, evs, report=report)
        if kept:
            self._explain(dict(candidates=kept), site, keep, ps, evs)
            results["candidates"] += kept
            rank_candidates(results, self.state)
        results["seconds"] = round(time.time() - t_start)
        write_atomic(self.folder / "results.json", to_json(results))
        report(f"{len(results['candidates'])} candidate locations", 1.0)

    def _near_candidates(self, legs: Legs, index: QuietIndex, candidates, also=()) -> list:
        """The flights within reach of any of `candidates` while its recording ran, as tracks."""
        reach, near = self.settings.audible_km * 1000, set(also)
        for c in candidates:
            x, y = c["xyz"][:2]
            near.update(index.near(x - reach, x + reach, y - reach, y + reach, c["start_unix"] - 300.0,
                                   c["start_unix"] + (self.state.get("duration") or 0.0) + 300.0).tolist())
        return [legs.track(i) for i in sorted(near)]

    def _legs(self, site) -> Legs:
        """The flights `_flights` keeps, without their splines."""
        s = self.settings
        types = load_type_table(DATA / "types" / "icao_aircraft_types2.js")
        reach = (s.half_km + s.audible_km) * 1000
        parts = []
        for name, meta, t, pos, vel in Archive(self.state["adsb"]["region"]).legs(site.frame, t_range=self.time_window()):
            if not is_jet(meta.get("t") or "", types) and (
                    (np.abs(pos[:, 0]) < reach) & (np.abs(pos[:, 1]) < reach) & (pos[:, 2] < s.max_alt_m)).sum() >= 5:
                parts.append((name, meta, t, pos, vel))
        return Legs.from_parts(parts)

    def _group(self, results, spec, site, keep, ps, events, report=None):
        """Merge candidates that rest on the same evidence into one, placed by all of it together.

        The searches (pairs of passes, single passes, voting, stacking) often find the same answer
        separately, a few hundred metres apart. Two candidates are one when they share most of their
        evidence flights (`_evidence`), never put a pass on different flights, agree on the start
        within GROUP_START_S and lie within GROUP_DIST_M of each other on the same side of every
        flight they share: a straight pass fits both sides of its track equally, and those must stay
        apart (between them, on the track, is wrong). Greedily, best first, each joins the best one
        it matches.

        Each group of two or more is then refined once, on a fine grid around its best member, with
        a shared start: every evidence flight's modelled curve is scored on the spectrogram (as a pass
        is refined) and the scores added. The group keeps that spot unless it explains less than its
        best member did (`_explain` decides), when the member's spot stands. A group lists its
        members and the methods that found it: independent methods agreeing is itself evidence."""
        cands = results["candidates"]
        if len(cands) < 2:
            return
        ev = {id(c): _evidence(c) for c in cands}
        left, groups = list(cands), []  # cands are ranked, best first
        while left:
            seed, rest = left[0], left[1:]
            members = [seed] + [c for c in rest if self._same_answer(seed, c, ev, keep)]
            groups.append(members)
            left = [c for c in rest if not any(c is m for m in members)]
        merged, jobs, plans = [], [], []
        for members in groups:
            if len(members) == 1:
                continue
            seed = members[0]
            items = {}
            for m in members:  # (flight, f0) for every evidence flight, the seed's f0 first
                for fl, f0 in ev[id(m)][2].items():
                    items.setdefault(fl, f0)
            index = {tr.flight_id: i for i, tr in enumerate(keep)}
            items = {fl: f0 for fl, f0 in items.items() if fl in index}
            xy = np.array(seed["xyz"][:2])
            spread = max(float(np.hypot(*(np.array(m["xyz"][:2]) - xy))) for m in members)
            half = float(np.clip(spread + 400.0, 600.0, 2000.0))
            obs = observer_grid(tuple(xy), half, 25.0 if half <= 800 else 50.0, place=site.place)
            dt = max(abs(m["start_unix"] - seed["start_unix"]) for m in members) + 3.0
            n = int(np.ceil(min(dt, 10.0) / spec.hop))
            shifts = np.arange(-n, n + 1)
            first = len(jobs)
            for fl, f0 in items.items():
                jobs.append(("curve", (keep[index[fl]], obs, seed["start_unix"], f0 * (1 + np.arange(-0.015, 0.0151, 0.0025)),
                                       shifts)))
            plans.append((members, obs, shifts, first, len(jobs), spread))
        if not plans:
            return
        if report:
            report(f"placing {len(plans)} groups of candidates that share their evidence", 0.95)
        scored = refine_many(spec, jobs, WORKERS)
        truth = site.to_xy(self.settings.truth_lat, self.settings.truth_lon) if self.settings.truth_lat is not None else None
        for members, obs, shifts, a, b, spread in plans:
            seed = members[0]
            total = np.sum(scored[a:b], axis=0)  # (points, shifts)
            k, j = np.unravel_index(int(np.argmax(total)), total.shape)
            best, t0 = obs[k], float(seed["start_unix"] + shifts[j] * spec.hop)
            lat, lon = site.to_latlon(best)
            merged.append(dict(
                pair="group", pair_evidence=None, lat=round(float(lat), 6), lon=round(float(lon), 6),
                score=round(float(total[k, j]), 2), score1=None, score2=None, followed=None, start_utc=utc(t0),
                start_unix=t0, xyz=_r(best, 1),
                truth_m=None if truth is None else round(float(np.hypot(*(best[:2] - truth)))), flights=[],
                group=dict(
                    methods=sorted({_method(m["pair"]) for m in members}), spread_m=round(spread),
                    members=[dict(pair=m["pair"], lat=m["lat"], lon=m["lon"], start_utc=m["start_utc"],
                                  truth_m=m["truth_m"], combined=m.get("combined"), xyz=m["xyz"],
                                  start_unix=m["start_unix"], score=m["score"]) for m in members]),
            ))
        sub = dict(results, candidates=merged)
        self._explain(sub, site, keep, ps, events)
        for g in merged:  # the joint spot, unless it explains less than the best member did
            best = g["group"]["members"][0]
            if (g.get("combined") or 0) < (best.get("combined") or 0) - 1e-6:
                g.update(lat=best["lat"], lon=best["lon"], xyz=best["xyz"], start_unix=best["start_unix"],
                         start_utc=best["start_utc"], truth_m=best["truth_m"], placed_by="best member")
            else:
                g["placed_by"] = "all evidence"
        self._explain(sub, site, keep, ps, events)
        grouped = {id(m) for members, *_ in plans for m in members}
        results["candidates"] = merged + [c for c in cands if id(c) not in grouped]
        rank_candidates(results, self.state)

    @staticmethod
    def _same_answer(a, b, ev, keep):
        """Whether candidates a and b rest on the same evidence at much the same place (`_group`)."""
        pa, fa, _ = ev[id(a)]
        pb, fb, _ = ev[id(b)]
        if not fa or not fb or len(fa & fb) < max(1, 0.5 * len(fa | fb)):
            return False
        if any(pa[p] != pb[p] for p in pa.keys() & pb.keys()):
            return False
        if abs(a["start_unix"] - b["start_unix"]) > GROUP_START_S:
            return False
        xa, xb = np.array(a["xyz"]), np.array(b["xyz"])
        if np.hypot(*(xa[:2] - xb[:2])) > GROUP_DIST_M:
            return False
        tracks = {tr.flight_id: tr for tr in keep}
        for fl in fa & fb:
            tr = tracks.get(fl)
            if tr is not None and _side(tr, xa) * _side(tr, xb) < 0:
                return False
        return True

    def _scorable(self, min_change=0.05):
        """Passes worth matching with flights: every one not a harmonic of another whose line changes
        pitch by at least `min_change` (any flight's flat stretch would follow a flat line)."""
        return [p for p in self.state["passes"] if p["kind"] != "harmonic"
                and np.log(max(p["line_f"]) / min(p["line_f"])) >= min_change]

    def _voters(self):
        """Passes that vote: the scorable ones not rejected."""
        return [p for p in self._scorable() if p["status"] != "rejected"]

    def _explain_set(self, results, chosen):
        """(passes, events) the candidates are ranked by: the voters if the search voted, else the accepted."""
        ps = self._voters() if results.get("votes") else chosen
        return ps, [self.pass_event(p) for p in ps]

    def explain(self, report):
        """Re-rank existing results by how many of the accepted passes each candidate explains."""
        path = self.folder / "results.json"
        if not path.exists():
            raise ValueError("run the search first")
        report("loading flights", 0.1)
        site = self.site()
        results = json.loads(path.read_text())
        ps, events = self._explain_set(results, [p for p in self.state["passes"] if p["status"] == "accepted"])
        if self.wide:
            legs = self._legs(site)
            keep = self._near_candidates(legs, QuietIndex(legs.samples()), results["candidates"])
        else:
            keep = self._flights(site)
        self._explain(results, site, keep, ps, events)
        write_atomic(path, to_json(results))
        best = results["candidates"][0] if results["candidates"] else None
        report("best candidate explains " + (f"{best['explained']} of {len(ps)} passes" if best else "nothing"), 1.0)

    def near_tracks(self, legs: Legs, index: QuietIndex, site, lat, lon) -> list:
        """Flights that come within reach of a place, at any time (`Project.try_spot`, for an area
        too wide to keep every flight's spline)."""
        x, y = site.to_xy(lat, lon)
        reach = self.settings.audible_km * 1000
        return [legs.track(int(i)) for i in index.near(x - reach, x + reach, y - reach, y + reach, legs.t.min(), legs.t.max())]

    def try_spot(self, lat, lon, keep=None, site=None) -> dict:
        """Check a spot picked by hand exactly as the search's candidates are checked, and add it to
        the results as one, replacing the last one tried unless that was kept. Returns the results,
        ranked, with the new candidate's key as `tried`."""
        path = self.folder / "results.json"
        if not path.exists():
            raise ValueError("run the search first")
        results = read_json(path)
        s = self.settings
        site = site or self.site()
        x, y = site.to_xy(lat, lon)
        if max(abs(x), abs(y)) > s.half_km * 1000:
            raise ValueError("that spot is outside the area searched: widen the area in step 1 first")
        keep = keep if keep is not None else self._flights(site)
        xyz = site.place(*site.to_xy(lat, lon)[:, None])[0]
        ps, events = self._explain_set(results, [p for p in self.state["passes"] if p["status"] == "accepted"])
        start = self._best_start(keep, xyz, ps, events)
        truth = site.to_xy(s.truth_lat, s.truth_lon) if s.truth_lat is not None else None
        c = dict(pair="map", pair_evidence=None, lat=round(float(lat), 6), lon=round(float(lon), 6), score=0.0,
                 start_unix=start, start_utc=utc(start), xyz=_r(xyz, 1), flights=[], kept=False,
                 truth_m=None if truth is None else round(float(np.hypot(x - truth[0], y - truth[1]))))
        self._explain(dict(candidates=[c], stack=results.get("stack")), site, keep, ps, events)
        results["candidates"] = [o for o in results["candidates"]
                                 if not (o["pair"] == "map" and not o.get("kept")) and o.get("key") != c["key"]] + [c]
        rank_candidates(results, self.state)
        write_atomic(path, to_json(results))
        results["tried"] = c["key"]
        return results

    def _best_start(self, keep, xyz, ps, events, agree_s=8.0) -> float:
        """The recording start that explains most of the passes from `xyz`. Each flight heard there
        is laid against each pass, around the start that puts the pass at the flight's closest
        approach, and the start with the most of the passes' lines followed within `agree_s` wins.
        With a known start, only flights it allows are tried."""
        s = self.settings
        known = s.start_unix is not None
        tried = []
        for tr in keep:
            ca = closest_approach(tr, xyz[None], max_dist=s.audible_km * 1000)
            if not ca.valid[0]:
                continue
            for n, (p, ev) in enumerate(zip(ps, events)):
                t0 = float(ca.t_heard[0] - ev.t_heard)
                turning = p["kind"] == "possibly turning"  # its measured closest moment is unreliable
                reach = 30.0 if turning else 8.0
                if known and abs(t0 - s.start_unix) > s.start_tol_s + reach:
                    continue
                e = explain_passes([tr], xyz, t0, [ev], max_dist=s.audible_km * 1000, f0_tol=0.3 if turning else 0.15,
                                   shifts_s=np.arange(-reach, reach + 0.5, 1.0))[0]
                if e.flight >= 0 and e.share > 0:
                    tried.append((t0 + e.shift, n, e.share))
        if not tried:
            if known:
                return float(s.start_unix)
            raise ValueError("no flight heard from that spot follows any of the passes")

        def around(t):
            best = {}
            for x in tried:
                if abs(x[0] - t) <= agree_s and x[2] > best.get(x[1], (0, 0, 0))[2]:
                    best[x[1]] = x
            return list(best.values())

        t = max((x[0] for x in tried), key=lambda t: sum(x[2] for x in around(t)))
        return float(np.median([x[0] for x in around(t)]))

    def keep_spot(self, key, keep=True) -> dict:
        """Keep a spot tried by hand (it then outlasts another try, and new searches), or drop it."""
        path = self.folder / "results.json"
        results = read_json(path)
        spots = [c for c in results["candidates"] if c["pair"] == "map" and candidate_key(c) == key]
        if not spots:
            raise ValueError("no such spot")
        if keep:
            spots[0]["kept"] = True
        else:
            results["candidates"].remove(spots[0])
        rank_candidates(results, self.state)
        write_atomic(path, to_json(results))
        return results

    def time_window(self, margin_s=900.0):
        """When a flight heard on the recording can have flown (unix s), or None if the start isn't
        known: the recording, at its earliest and latest possible start, and `margin_s` either side
        (sound travels a minute or so, and curves are scored a few minutes around a pass)."""
        s = self.settings
        if s.start_unix is None:
            return None
        return (s.start_unix - s.start_tol_s - margin_s,
                s.start_unix + (self.state.get("duration") or 0.0) + s.start_tol_s + margin_s)

    def _flights(self, site):
        """Propeller (or unknown) aircraft that come near the area, low enough to hear, and (with
        the start known) only as flown while the recording ran: the rest of the day can't be heard
        on it, and over a wide area it is most of the work (3.5x the flights over 500 km)."""
        s = self.settings
        tracks = Archive(self.state["adsb"]["region"]).tracks(site.frame, t_range=self.time_window())
        types = load_type_table(DATA / "types" / "icao_aircraft_types2.js")
        reach = (s.half_km + s.audible_km) * 1000
        return [tr for tr in tracks if not is_jet(tr.typecode, types) and (
            (np.abs(tr.pos[:, 0]) < reach) & (np.abs(tr.pos[:, 1]) < reach) & (tr.pos[:, 2] < s.max_alt_m)).sum() >= 5]

    def _explain(self, results, site, keep, chosen, events):
        """Add to each candidate which flight best explains each accepted pass at its place and start
        time (`search.explain_passes`), and rank by the total: the right answer should account for
        most real passes, not just the two it was found from. When the search stacked the
        spectrogram, each candidate also gets its stacked evidence there (`stack`, with the flights
        `heard`), each heard flight counting up to one more, as a pass explained does."""
        cands = results["candidates"]
        xyzs = [np.array(c["xyz"]) if "xyz" in c else site.place(*site.to_xy(c["lat"], c["lon"])[:, None])[0]
                for c in cands]
        stacked = None
        if (results.get("stack") or {}).get("stats") and cands:
            # All candidates at once, each at its own start: the flights' closest approaches are shared.
            index = {tr.flight_id: i for i, tr in enumerate(keep)}
            stats = {index[f]: tuple(v) for f, v in results["stack"]["stats"].items() if f in index}
            spec = self.spectrogram()
            peaks, taus = spectrogram_peaks(spec), np.arange(-4.0, 4.1, 1.0)
            total, heard = stack_at(keep, np.array(xyzs), np.array([c["start_unix"] for c in cands]), peaks, stats,
                                    taus=taus, cap=STACK_CAP, floor=STACK_FLOOR,
                                    max_dist=self.settings.audible_km * 1000)
            stacked = (total, heard, taus)
        for n, c in enumerate(cands):
            xyz = xyzs[n]
            # A turning plane's straight-line fit gets its pitch wrong too, so allow it more slack.
            f0_tol = [0.3 if p["kind"] == "possibly turning" else 0.15 for p in chosen]
            found = explain_passes(keep, xyz, c["start_unix"], events, max_dist=self.settings.audible_km * 1000,
                                   f0_tol=f0_tol)
            c["explains"] = []
            for p, e in zip(chosen, found):
                entry = dict(pass_id=p["id"], flight=None, registration=None, type=None)
                if e.flight >= 0:
                    # Who it was and its modelled curve, so the GUI can show every explained pass.
                    tr = keep[e.flight]
                    entry = _flight_info(site, tr, closest_approach(tr, xyz[None]), xyz, c["start_unix"] + e.shift,
                                         e.f0, p)
                c["explains"].append(dict(entry, share=round(e.share, 2)))
            c["support"] = round(sum(x["share"] for x in c["explains"]), 2)
            c["stack"], c["heard"] = 0.0, []
            if stacked is not None:
                total, heard, taus = stacked
                j = int(total[n].argmax())
                c["stack"] = round(float(total[n, j]), 2)
                c["heard"] = self._heard_info(site, keep, xyz, c["start_unix"] + taus[j], heard[n])
                for h in c["heard"]:  # does its curve follow a line one can see?
                    h.update(curve_visibility(spec, h["curve_t"], h["curve_f"]))
        results["explained_over"] = [p["id"] for p in chosen]
        results["checked_at"] = time.time()
        results["ranked_by"] = "support"
        rank_candidates(results, self.state)

    def _results(self, spec, site, grid, keep, chosen, events, found, n_refine, singles=(), votes=None, stack=None, anchored_apart=False):
        """`anchored_apart`: `found` holds anchored pairs after pairs searched another way, whose evidence
        is on another scale: the best three of each are used, instead of the best three of all."""
        s = self.settings
        truth = (site.to_xy(s.truth_lat, s.truth_lon) if s.truth_lat is not None else None)
        pairs, candidates, survivors = [], [], {}
        apart = 0  # anchored pairs among `survivors` that don't count towards the limits
        # Refinements wait in `pending` (candidate slot, job for `refine_many`, how to describe its
        # result) and then run together on the workers: one at a time they were most of the end.
        pending = []
        for ep in found:
            res, (pa, pb) = ep.result, (chosen[ep.i], chosen[ep.j])
            key = f"{pa['id']}-{pb['id']}"
            entry = dict(key=key, a=pa["id"], b=pb["id"], evidence=round(ep.best, 1), flight_pairs=len(ep.rows),
                         funnel=res.funnel)
            pairs.append(entry)
            if not len(ep.rows) or (apart >= 3 if anchored_apart and ep.anchored else len(survivors) - apart >= 3):
                continue
            apart += bool(anchored_apart and ep.anchored)
            # Points that survived the filters, coloured by score, for the map.
            pts = np.full(len(grid), np.nan)
            np.fmax.at(pts, res.point, res.total)
            idx = np.flatnonzero(np.isfinite(pts))
            idx = idx[np.argsort(-pts[idx])[:MAX_SURVIVORS]]
            lat, lon = site.to_latlon(grid[idx])
            survivors[key] = [[round(float(a), 5), round(float(b), 5), round(float(v), 2)]
                              for a, b, v in zip(lat, lon, pts[idx])]
            ea, eb = events[ep.i], events[ep.j]
            # By evidence (how much of both traced lines the flights follow), not curve score: anything
            # bright near a modelled curve lifts its score, and in a wide area some wrong pair always scores.
            for r in ep.rows[:n_refine]:
                fa, fb = keep[res.a[r]], keep[res.b[r]]
                if ep.anchored:  # already where the curves follow the lines: spectrogram scores can't place a turning pass
                    candidates.append(self._candidate_placed(site, grid[res.point[r]], float(res.t0[r]), fa, fb, res.f0_1[r], res.f0_2[r], pa, pb, ep,
                                                             ep_row(ep, r), key, truth))
                    continue
                obs = observer_grid(tuple(grid[res.point[r], :2]), 600.0, 25.0, place=site.place)
                describe = partial(self._candidate, spec, site, obs, fa, fb, ea, pa, pb, ep, ep_row(ep, r), key, truth)
                pending.append((len(candidates), ("pair", (fa, fb, obs, ea, eb, 6.0)), describe))
                candidates.append(None)
        single_entries = []
        for k, res in singles:
            p, key = chosen[k], chosen[k]["id"]
            rows = res.best_per_flight()
            single_entries.append(dict(key=key, flights=len(rows), funnel=res.funnel))
            if not len(rows):
                continue
            if len(survivors) - apart < 6:
                pts = np.full(len(grid), np.nan)
                np.fmax.at(pts, res.point, res.score)
                idx = np.flatnonzero(np.isfinite(pts))
                lat, lon = site.to_latlon(grid[idx])
                survivors[key] = [[round(float(a), 5), round(float(b), 5), round(float(v), 2)]
                                  for a, b, v in zip(lat, lon, pts[idx])]
            for r in rows[:n_refine]:
                # The search found the recording's start to a frame for this flight: a point a grid
                # step away can't move it much, so only REFINE_START_TOL s either side is tried, not the
                # whole uncertainty again (that was most of the refining with a wide one).
                fl, found_start = keep[res.flight[r]], float(res.start[r])
                obs = observer_grid(tuple(grid[res.point[r], :2]), 600.0, 25.0, place=site.place)
                describe = partial(self._candidate_single, spec, site, obs, fl, p, key, truth, found_start)
                pending.append((len(candidates), ("start", (fl, obs, events[k], (found_start, REFINE_START_TOL))),
                                describe))
                candidates.append(None)
        for (slot, _, describe), scored in zip(pending, refine_many(spec, [j for _, j, _ in pending], WORKERS)):
            candidates[slot] = describe(scored)
        vote_entry = matches = None
        if votes is not None:
            v, voters, voter_events = votes
            vote_entry = dict(passes=[p["id"] for p in voters if p["status"] != "rejected"], places=len(v.point),
                              seconds=round(v.seconds))
            matches = self._matches(site, grid, keep, voters, v)
            if len(v.point):
                lat, lon = site.to_latlon(grid[v.point])
                survivors["votes"] = [[round(float(a), 5), round(float(b), 5), round(float(x), 2)]
                                      for a, b, x in zip(lat, lon, v.votes)]
                candidates += self._vote_candidates(site, grid, keep, voters, voter_events, v, truth)
        stack_entry = heat = None
        if stack is not None:
            sm, coarse, peaks = stack
            places = sm.places(coarse)
            # Each flight's spread, so that later checks (`explain`) score candidates the same way.
            stack_entry = dict(places=len(places), flights=len(sm.stats), seconds=round(sm.seconds),
                               stats={keep[i].flight_id: [round(a, 4), round(b, 4)] for i, (a, b) in sm.stats.items()})
            heat = _heat_grid(sm.score, s.half_km * 1000, 1000.0)
            heat["floor"] = STACK_CAP - STACK_FLOOR  # one flight's most: more takes flights agreeing
            top = np.argsort(-sm.score)[:2000]
            top = top[sm.score[top] > 0]
            if top.size:
                lat, lon = site.to_latlon(coarse[top])
                survivors["stack"] = [[round(float(a), 5), round(float(b), 5), round(float(x), 2)]
                                      for a, b, x in zip(lat, lon, sm.score[top])]
            for k in places:
                obs, t0, score, heard = refine_place(keep, coarse[k, :2], sm.start[k], peaks, sm.stats,
                                                     place=site.place, cap=STACK_CAP, floor=STACK_FLOOR,
                                                     max_dist=s.audible_km * 1000)
                lat, lon = site.to_latlon(obs)
                candidates.append(dict(
                    pair="stack", pair_evidence=None, lat=round(float(lat), 6), lon=round(float(lon), 6),
                    score=round(score, 2), score1=None, score2=None, followed=None, start_utc=utc(t0), start_unix=t0,
                    xyz=_r(obs, 1), truth_m=None if truth is None else round(float(np.hypot(*(obs[:2] - truth)))),
                    flights=[],
                ))
        area = self._area_mask(site)
        if area is not None:  # the grid and flights cover the square around a shape: keep what is in it
            candidates = [c for c in candidates if area.contains(c["xyz"][0], c["xyz"][1])]
        candidates.sort(key=lambda c: (-(c["pair_evidence"] or 0), -c["score"]))
        for k, c in enumerate(candidates):
            c["rank"] = k + 1
        h = s.half_km * 1000
        box = [list(map(float, site.to_latlon(np.array([x * h, y * h, 0.0]))))
               for x, y in ((-1, -1), (-1, 1), (1, 1), (1, -1))]
        return dict(event_pairs=pairs, singles=single_entries, votes=vote_entry, stack=stack_entry,
                    candidates=candidates, survivors=survivors, box=box, start_known=s.start_unix is not None,
                    matches=matches, heat=heat)

    def _candidate(self, spec, site, obs, fa, fb, ea, pa, pb, ep, row, key, truth, scored):
        """Describe the best spot of a flight pair's fine grid `obs` (`refine_many`'s (s1, s2)) for the GUI."""
        s1, s2 = scored
        tot = s1.score + s2.score
        k = int(np.argmax(tot))
        best = obs[k]
        ca, cb = closest_approach(fa, best[None]), closest_approach(fb, best[None])
        t0 = float(ca.t_heard[0] - ea.t_heard + s1.shift[k] * spec.hop)
        lat, lon = site.to_latlon(best)
        flights = [_flight_info(site, tr, ca_, best, t0, f0, p)
                   for tr, ca_, f0, p in ((fa, ca, s1.f0[k], pa), (fb, cb, s2.f0[k], pb))]
        return dict(
            pair=key, pair_evidence=round(ep.best, 1), lat=round(float(lat), 6), lon=round(float(lon), 6),
            score=round(float(tot[k]), 2), score1=round(float(s1.score[k]), 2), score2=round(float(s2.score[k]), 2),
            followed=None if row is None else _r(ep.followed[row], 0), start_utc=utc(t0), start_unix=t0,
            xyz=_r(best, 1),
            truth_m=None if truth is None else round(float(np.hypot(*(best[:2] - truth)))), flights=flights,
        )


    def _candidate_placed(self, site, xyz, t0, fa, fb, f0_a, f0_b, pa, pb, ep, row, key, truth):
        """As `_candidate`, for a pair already placed by how well its curves follow the traced lines."""
        best = np.asarray(xyz, dtype=float)
        lat, lon = site.to_latlon(best)
        flights = [_flight_info(site, tr, closest_approach(tr, best[None]), best, t0, f0, p) for tr, f0, p in ((fa, f0_a, pa), (fb, f0_b, pb))]
        return dict(
            pair=key, pair_evidence=round(ep.best, 1), lat=round(float(lat), 6), lon=round(float(lon), 6), score=0.0, score1=None, score2=None,
            followed=None if row is None else _r(ep.followed[row], 0), start_utc=utc(t0), start_unix=t0, xyz=_r(best, 1),
            truth_m=None if truth is None else round(float(np.hypot(*(best[:2] - truth)))), flights=flights,
        )

    def _candidate_single(self, spec, site, obs, fl, p, key, truth, found_start, sc):
        """As `_candidate`, for one flight and one pass at the known start (`sc` its EventScore on
        `obs`, shifts counted from `found_start`)."""
        k = int(np.argmax(sc.score))
        best = obs[k]
        t0 = float(found_start + sc.shift[k] * spec.hop)
        ca = closest_approach(fl, best[None])
        lat, lon = site.to_latlon(best)
        return dict(
            pair=key, pair_evidence=None, lat=round(float(lat), 6), lon=round(float(lon), 6),
            score=round(float(sc.score[k]), 2), score1=round(float(sc.score[k]), 2), score2=None, followed=None,
            start_utc=utc(t0), start_unix=t0, xyz=_r(best, 1),
            truth_m=None if truth is None else round(float(np.hypot(*(best[:2] - truth)))),
            flights=[_flight_info(site, fl, ca, best, t0, sc.f0[k], p)],
        )


    @staticmethod
    def _heard_info(site, keep, xyz, t0, heard) -> list[dict]:
        """The flights stacking finds heard at `xyz` (from `stack.stack_at`), as `_flight_info` describes
        a pass's flight, named S1, S2... with their share of the evidence (`z`, at most cap - floor)."""
        out = []
        for n, (i, part, f0, th) in enumerate(heard):
            tr = keep[i]
            info = _flight_info(site, tr, closest_approach(tr, xyz[None]), xyz, t0, f0,
                                dict(id=f"S{n + 1}", t_heard=th - t0))
            out.append(dict(info, z=round(part, 2), heard_rec=round(th - t0, 1)))
        return out

    @staticmethod
    def _matches(site, grid, keep, scored, v) -> dict:
        """For each scored pass, the flight that follows most of its line anywhere in the area at the
        known start (`Votes.best`): whether any aircraft could have made it, turns and all, which a
        straight-pass fit cannot tell. Share 0 when none follows half of it."""
        out = {}
        for p, b in zip(scored, v.best):
            if b is None:
                out[p["id"]] = dict(share=0.0)
                continue
            share, f, point, t0 = b
            m = getattr(keep[f], "meta", None) or {}
            lat, lon = site.to_latlon(grid[point])
            out[p["id"]] = dict(share=round(share, 2), flight=keep[f].flight_id, registration=m.get("r"),
                                type=m.get("t"), lat=round(float(lat), 5), lon=round(float(lon), 5), start_utc=utc(t0))
        return out

    def _vote_candidates(self, site, grid, keep, voters, events, v, truth, n=5, apart=1000.0, fine=(400.0, 25.0)):
        """The `n` best-voted places at least `apart` m from each other and voted for by different
        passes and flights (the same ones elsewhere are the same answer, less well placed), each voted
        on again on a fine grid by the passes and flights that voted for it there."""
        out, taken, seen = [], [], set()
        for k in range(len(v.point)):
            xy = grid[v.point[k], :2]
            sig = frozenset(zip(v.passes[k].tolist(), v.flights[k].tolist()))
            if sig in seen or any(np.hypot(*(xy - t)) < apart for t in taken):
                continue
            taken.append(xy)
            seen.add(sig)
            if len(taken) > n:
                break
            ev_ids, fl_ids = list(v.passes[k]), sorted(set(v.flights[k].tolist()))
            sub = [keep[i] for i in fl_ids]
            obs = observer_grid(xy, *fine, place=site.place)
            fv = vote_passes(sub, obs, [events[e] for e in ev_ids], (float(v.start[k]), 6.0),
                             passes=Passes(sub, obs, self.settings.audible_km * 1000))
            if len(fv.point):
                best, t0, total = obs[fv.point[0]], float(fv.start[0]), float(fv.votes[0])
                pairs = [(ev_ids[e], fl_ids[f]) for e, f in zip(fv.passes[0], fv.flights[0])]
            else:
                best, t0, total = grid[v.point[k]], float(v.start[k]), float(v.votes[k])
                pairs = list(zip(v.passes[k].tolist(), v.flights[k].tolist()))
            pairs.sort(key=lambda x: voters[x[0]]["t_heard"])
            flights = []
            for e, f in pairs:
                tr, (times, freqs) = keep[f], events[e].line
                g = doppler_factor(tr, best, t0 + times)
                flights.append(_flight_info(site, tr, closest_approach(tr, best[None]), best, t0,
                                            float(np.nanmedian(freqs / g)), voters[e]))
            lat, lon = site.to_latlon(best)
            out.append(dict(
                pair="votes", voted=[voters[e]["id"] for e, _ in pairs], pair_evidence=None,
                lat=round(float(lat), 6), lon=round(float(lon), 6), score=round(total, 2), score1=None, score2=None,
                followed=None, start_utc=utc(t0), start_unix=t0, xyz=_r(best, 1),
                truth_m=None if truth is None else round(float(np.hypot(*(best[:2] - truth)))), flights=flights,
            ))
        return out


def _flight_info(site, tr, ca, best, t0, f0, p) -> dict:
    """A flight as a candidate explains one pass: who it was, its closest approach, its modelled
    curve on the recording and its path, for the GUI."""
    m = getattr(tr, "meta", {}) or {}
    t_rec = np.linspace(p["t_heard"] - 180, p["t_heard"] + 180, 361)
    curve = f0 * doppler_factor(tr, best, t0 + t_rec)
    ok = np.isfinite(curve)
    ts = np.arange(max(tr.t_start, ca.t[0] - 1200), min(tr.t_end, ca.t[0] + 1200), 10.0)
    path_lat, path_lon = site.to_latlon(tr.position(ts))
    return dict(
        pass_id=p["id"], flight=tr.flight_id, registration=m.get("r"), icao=m.get("icao"), type=m.get("t"),
        desc=m.get("desc"), cpa_m=round(float(ca.dist[0])), speed=round(float(ca.speed[0])),
        alt_m=round(float(site.frame.height(tr.position(ca.t[0])))), heard_utc=utc(ca.t_heard[0]), f0=round(float(f0), 2),
        curve_t=_r(t_rec[ok]), curve_f=_r(curve[ok]),
        path=[[round(float(a), 5), round(float(b), 5)] for a, b in zip(path_lat, path_lon)],
    )


def _heat_grid(score, half_m, spacing_m, max_side=400) -> dict:
    """A score over a square grid (`observer_grid` order: rows south to north, columns west to
    east) for the map's heatmap: blocks of cells merged to their highest so that a side has at most
    `max_side` cells (a 500 km area at 1 km would be 250,000), values to 0.1."""
    n = int(round(np.sqrt(len(score))))
    grid = np.asarray(score, dtype=float).reshape(n, n)
    k = int(np.ceil(n / max_side))
    if k > 1:
        m = -(-n // k) * k
        padded = np.full((m, m), 0.0)
        padded[:n, :n] = grid
        grid = padded.reshape(m // k, k, m // k, k).max(axis=(1, 3))
    return dict(kind="stacking", rows=grid.shape[0], cols=grid.shape[1], half_m=half_m, spacing_m=spacing_m,
                cell_m=spacing_m * k, values=[round(float(v), 1) for v in np.clip(grid, 0, None).ravel()])


def _evidence(c, explained_at=EXPLAINED_AT):
    """A candidate's evidence (`_explain`): {pass id: flight} for the passes it explains, the set of
    flights it rests on (those, and heard flights whose curve is visible) and {flight: emitted pitch}."""
    passes, f0 = {}, {}
    for x in c.get("explains") or []:
        if x.get("flight") and x.get("share", 0) >= explained_at:
            passes[x["pass_id"]] = x["flight"]
            f0.setdefault(x["flight"], x["f0"])
    for h in c.get("heard") or []:
        if h.get("grade") == "visible":
            f0.setdefault(h["flight"], h["f0"])
    return passes, set(f0), f0


def _side(tr, xyz, min_m=150.0) -> int:
    """Which side of flight `tr`'s track (+1 left, -1 right, looking along it) the point is, at its
    closest approach; 0 within `min_m` of the track, where either side will do."""
    t = closest_approach(tr, np.asarray(xyz, dtype=float)[None]).t[0]
    p, v = tr.position(t), tr.velocity(t)
    d = (v[0] * (xyz[1] - p[1]) - v[1] * (xyz[0] - p[0])) / max(float(np.hypot(v[0], v[1])), 1e-9)
    return 0 if abs(d) < min_m else int(np.sign(d))


def _method(pair) -> str:
    """Which search found a candidate, from its `pair` field."""
    return {"stack": "stacking", "votes": "voting", "group": "group"}.get(pair) or (
        "pair of passes" if "-" in pair else "single pass")


def _line_ratio(a, b, hop=0.5) -> float:
    """Pitch of pass b's traced line over pass a's where both were traced (median)."""
    ta, tb = np.rint(np.array(a["line_t"]) / hop), np.rint(np.array(b["line_t"]) / hop)
    _, ia, ib = np.intersect1d(ta, tb, return_indices=True)
    return float(np.exp(np.median(np.log(np.array(b["line_f"])[ib]) - np.log(np.array(a["line_f"])[ia]))))


def ep_row(ep, r):
    """Position of result row r in an EventPair's evidence ranking, or None."""
    hit = np.flatnonzero(ep.rows == r)
    return int(hit[0]) if hit.size else None


def run(step, folder):
    """Run one step, writing progress to jobs/<step>.json, below normal priority."""
    lower_priority()
    project = Project(folder)
    status_path = project.folder / "jobs" / f"{step}.json"
    status_path.parent.mkdir(exist_ok=True)
    started = time.time()

    def write(**kw):
        write_atomic(status_path, to_json(dict(step=step, pid=os.getpid(), started=started, **kw)))

    def report(message, fraction):
        write(state="running", message=message, progress=round(float(fraction), 3))

    report("starting", 0.0)
    try:
        getattr(project, step)(report)
    except Exception as e:  # reported to the GUI rather than lost
        write(state="error", message=f"{type(e).__name__}: {e}", trace=traceback.format_exc(), progress=1.0)
        raise
    write(state="done", message=read_json(status_path).get("message", "done"), progress=1.0,
          seconds=round(time.time() - started))


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in STEPS:
        sys.exit(f"usage: python -m doppler_loc.project {{{','.join(STEPS)}}} <project folder>")
    run(sys.argv[1], sys.argv[2])
