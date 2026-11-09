"""Historical ADS-B from adsb.lol's daily readsb archives (github.com/adsblol/globe_history_YYYY).

A day is a split tar of gzipped `traces/xx/trace_full_<icao>.json` files, one per aircraft. Each
trace point is [seconds after `timestamp`, lat, lon, baro alt ft | "ground" | null, ground speed
kt, track deg, flags, vertical rate fpm, details | null, source, geometric alt ft, geometric rate
fpm, ias, roll].
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import gzip
import io
import json
from multiprocessing import Pool
from pathlib import Path
import tarfile
import urllib.error
import urllib.request

import numpy as np

from .flight import FlightTrack
from .geo import FT, KT, LocalFrame

COLUMNS = ("t", "lat", "lon", "alt_baro", "alt_geom", "gs", "track", "vrate")


class _Chain(io.RawIOBase):
    def __init__(self, paths):
        self._files = [open(p, "rb") for p in paths]
        self._i = 0

    def readable(self):
        return True

    def readinto(self, b):
        while self._i < len(self._files):
            n = self._files[self._i].readinto(b)
            if n:
                return n
            self._files[self._i].close()
            self._i += 1
        return 0


def _num(v):
    return float(v) if isinstance(v, (int, float)) else np.nan


_audible_only = None  # in an `extract_region` worker: (type table, highest altitude in feet)


def _init_extract(audible_only):
    global _audible_only
    _audible_only = audible_only


def _parse(args):
    raw, lat0, lon0, radius_km = args
    try:
        d = json.loads(gzip.decompress(raw))
    except (OSError, ValueError):
        return None
    tr = d.get("trace") or []
    if not tr or (_audible_only and is_jet(d.get("t") or "", _audible_only[0])):
        return None
    lat = np.array([_num(p[1]) for p in tr])
    lon = np.array([_num(p[2]) for p in tr])
    dy = (lat - lat0) * 111.2
    dx = (lon - lon0) * 111.2 * np.cos(np.radians(lat0))
    keep = np.hypot(dx, dy) <= radius_km
    if _audible_only:
        geom = np.array([_num(p[10]) if len(p) > 10 else np.nan for p in tr])
        baro = np.array([np.nan if p[3] == "ground" else _num(p[3]) for p in tr])
        alt = np.where(np.isfinite(geom), geom, baro)
        keep &= (alt > 0) & (alt < _audible_only[1])
        if keep.sum() < 5:
            return None
    if not keep.any():
        return None
    rows = [p for p, k in zip(tr, keep) if k]
    cols = {
        "t": np.array([d["timestamp"] + p[0] for p in rows]),
        "lat": lat[keep],
        "lon": lon[keep],
        "alt_baro": np.array([0.0 if p[3] == "ground" else _num(p[3]) for p in rows]),
        "alt_geom": np.array([_num(p[10]) if len(p) > 10 else np.nan for p in rows]),
        "gs": np.array([_num(p[4]) for p in rows]),
        "track": np.array([_num(p[5]) for p in rows]),
        "vrate": np.array([_num(p[11]) if len(p) > 11 and p[11] is not None else _num(p[7]) for p in rows]),
    }
    meta = {k: d.get(k) for k in ("icao", "r", "t", "desc")}
    return meta, cols


def extract_region(tar_parts, lat0, lon0, radius_km, out_path, *, workers=4, progress=None, audible_only=None):
    """Keep every trace point within `radius_km` of (lat0, lon0) and save them to one .npz.

    `progress(aircraft_scanned, aircraft_kept)` is called every few thousand aircraft. With
    `audible_only` = (type table, highest altitude in feet) only low points of aircraft that aren't
    jets are kept, unpacked: a day over the whole US is then 1.7 GB, and a minute to cut.
    """
    stream = io.BufferedReader(_Chain(tar_parts), buffer_size=1 << 22)
    metas, cols = [], {k: [] for k in COLUMNS}

    def jobs():
        with tarfile.open(fileobj=stream, mode="r|") as tar:
            for m in tar:
                if m.isfile() and "trace_full_" in m.name:
                    yield tar.extractfile(m).read(), lat0, lon0, radius_km

    n_files = 0
    with Pool(workers, _init_extract, (audible_only,)) as pool:
        for res in pool.imap_unordered(_parse, jobs(), chunksize=64):
            n_files += 1
            if progress and n_files % 5000 == 0:
                progress(n_files, len(metas))
            if res is None:
                continue
            meta, c = res
            metas.append(meta)
            for k in COLUMNS:
                cols[k].append(c[k])

    lengths = np.array([len(c) for c in cols["t"]])
    (np.savez if audible_only else np.savez_compressed)(
        out_path,
        meta=json.dumps(metas),
        offsets=np.concatenate([[0], np.cumsum(lengths)]),
        **{k: np.concatenate(v) if v else np.zeros(0) for k, v in cols.items()},
    )
    return n_files, len(metas)


class Archive:
    """Region extract written by `extract_region`."""

    def __init__(self, path):
        z = np.load(path)
        self.meta = json.loads(str(z["meta"]))
        self.offsets = z["offsets"]
        self.cols = {k: z[k] for k in COLUMNS}

    def __len__(self):
        return len(self.meta)

    def points(self, i) -> dict:
        s = slice(self.offsets[i], self.offsets[i + 1])
        return {k: v[s] for k, v in self.cols.items()}

    def tracks(self, frame: LocalFrame, *, t_range=None, max_gap=60.0, min_points=5) -> list[FlightTrack]:
        """Airborne legs as FlightTracks in `frame`, split wherever reports stop for `max_gap` s.

        Positions use geometric (GNSS, ellipsoidal) altitude. Where only barometric altitude is
        reported it is shifted by that aircraft's median geometric-minus-baro difference, since
        pressure altitude can be off by hundreds of feet. Velocities from ground speed, track and
        vertical rate make the interpolation Hermite, so sparse reports still give smooth motion.

        With `t_range` (unix s) only the legs' parts within it are kept. The day's legs are split,
        numbered and their altitudes corrected first, so a leg keeps its name (N123AB#2) and, the
        interpolation being local, its positions inside the range, whatever range is asked for.
        """
        out = []
        for name, meta, t, pos, vel in self.legs(frame, t_range=t_range, max_gap=max_gap, min_points=min_points):
            tr = FlightTrack(name, t, pos, vel=vel, typecode=meta.get("t") or "")
            tr.meta = meta
            out.append(tr)
        return out

    def legs(self, frame: LocalFrame, *, t_range=None, max_gap=60.0, min_points=5):
        """What `tracks` makes tracks of: (name, meta, t, pos, vel) of each leg, as it goes."""
        for i, meta in enumerate(self.meta):
            p = self.points(i)
            ok = np.isfinite(p["lat"]) & np.isfinite(p["gs"]) & np.isfinite(p["track"]) & (p["alt_baro"] > 0)
            if ok.sum() < min_points:
                continue
            if t_range is not None and not ((p["t"][ok] >= t_range[0]) & (p["t"][ok] <= t_range[1])).any():
                continue
            p = {k: v[ok] for k, v in p.items()}
            diff = p["alt_geom"] - p["alt_baro"]
            offset = np.nanmedian(diff) if np.isfinite(diff).any() else 0.0
            alt = np.where(np.isfinite(p["alt_geom"]), p["alt_geom"], p["alt_baro"] + offset) * FT

            trk = np.radians(p["track"])
            vr = np.nan_to_num(p["vrate"]) * FT / 60.0
            v_local = np.stack([p["gs"] * KT * np.sin(trk), p["gs"] * KT * np.cos(trk), vr], axis=1)
            pos = frame.to_enu(p["lat"], p["lon"], alt)
            vel = frame.vel_to_enu(p["lat"], p["lon"], v_local)

            t = p["t"]
            order = np.argsort(t, kind="stable")
            t, pos, vel = t[order], pos[order], vel[order]
            uniq = np.concatenate([[True], np.diff(t) > 0.05])
            t, pos, vel = t[uniq], pos[uniq], vel[uniq]
            cuts = np.flatnonzero(np.diff(t) > max_gap) + 1
            name = meta.get("r") or meta.get("icao")
            for leg, (a, b) in enumerate(zip(np.r_[0, cuts], np.r_[cuts, len(t)])):
                if b - a < min_points:
                    continue
                if t_range is not None:  # its reports within the range, and one either side to interpolate to its ends
                    a2 = max(a, int(np.searchsorted(t, t_range[0], "left")) - 1)
                    b2 = min(b, int(np.searchsorted(t, t_range[1], "right")) + 1)
                    if b2 - a2 < min_points:
                        continue
                    a, b = a2, b2
                yield f"{name}#{leg}", meta, t[a:b], pos[a:b], vel[a:b]


def coverage_gaps(t, aircraft, *, min_gap_s=180.0, context_s=1800.0, min_rate=3.0) -> list[tuple[float, float]]:
    """Stretches of at least `min_gap_s` with no position reported at all, between half hours
    (`context_s`) in which at least `min_rate` different aircraft were seen a minute on average: an
    outage of the archive (adsb.lol lost 14:21-15:35 UTC on 2026-08-19, in prod and staging alike),
    not a quiet sky. Aircraft rather than positions are counted, since one aircraft alone reports
    dozens a minute. `t`: every position's time in an area, unix s; `aircraft`: whose each is (any
    ids). Returns (start, end) pairs, unix s."""
    t, aircraft = np.asarray(t, dtype=float), np.asarray(aircraft)
    ok = np.isfinite(t)
    t, aircraft = t[ok], aircraft[ok]
    if t.size < 2:
        return []
    edges = np.arange(np.floor(t.min() / 60) * 60, t.max() + 61, 60.0)
    minute = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, len(edges) - 2)
    _, ids = np.unique(aircraft, return_inverse=True)
    pairs = np.unique(minute.astype(np.int64) * (ids.max() + 1) + ids)
    per_min = np.bincount(pairs // (ids.max() + 1), minlength=len(edges) - 1)
    n_ctx = int(context_s // 60)
    out, k = [], 0
    while k < len(per_min):
        if per_min[k]:
            k += 1
            continue
        j = k
        while j < len(per_min) and not per_min[j]:
            j += 1
        before, after = per_min[max(0, k - n_ctx):k], per_min[j:j + n_ctx]
        if (j - k) * 60 >= min_gap_s and before.size and after.size \
                and before.mean() >= min_rate and after.mean() >= min_rate:
            out.append((float(edges[k]), float(edges[j])))
        k = j
    return out


def low_coverage(lat, lon, alt_ft, aircraft, lat0, lon0, half_km, ground, *, cell_km=10.0, below_m=1500.0,
                 min_aircraft=3) -> dict:
    """How well the archive covers low flying over the square area (`half_km` around lat0, lon0).

    Receivers see a plane only above their radio horizon, so where none is near, low traffic (light
    aircraft, the ones heard close by) goes unlogged while airliners overhead are logged as usual:
    on 2024-04-08 nothing below 1000 m was logged all day within 25 km of a site in rural Missouri,
    while 120 km away had hundreds of positions. The area is cut into squares about `cell_km` on a
    side, and a square counts as covered when at least `min_aircraft` different aircraft were seen in
    it below `below_m` above its ground at some point that day. `ground(lat, lon)`: ground height
    above sea level, m, at the squares' centres (pressure altitude is compared with it).

    Returns n (squares a side), cell_km, below_m, min_aircraft, covered (share of the squares on
    land), and n x n lists, row 0 the southernmost, column 0 the westernmost: counts (aircraft per
    square) and water (squares whose centre is at sea, left out of the share).
    """
    n = max(1, int(np.ceil(2 * half_km / cell_km)))
    size = 2 * half_km / n
    kx = 111.2 * np.cos(np.radians(lat0))
    centres = (np.arange(n) + 0.5) * size - half_km
    c_lat, c_lon = np.meshgrid(lat0 + centres / 111.2, lon0 + centres / kx, indexing="ij")
    floor = np.asarray(ground(c_lat, c_lon), dtype=float)
    water = floor <= 0  # elevation tiles read the sea as 0 or below; nobody records out there
    floor = np.maximum(floor, 0.0)
    lat, lon, alt = (np.asarray(v, dtype=float) for v in (lat, lon, alt_ft))
    row = np.floor(((lat - lat0) * 111.2 + half_km) / size)
    col = np.floor(((lon - lon0) * kx + half_km) / size)
    inside = (row >= 0) & (row < n) & (col >= 0) & (col < n) & (alt > 0)  # 0: on the ground
    row, col = row[inside].astype(np.int64), col[inside].astype(np.int64)
    low = alt[inside] * FT < floor[row, col] + below_m
    cell = (row * n + col)[low]
    _, who = np.unique(np.asarray(aircraft)[inside][low], return_inverse=True)
    pairs = np.unique(cell * (who.max(initial=0) + 1) + who)
    counts = np.bincount(pairs // (who.max(initial=0) + 1), minlength=n * n).reshape(n, n)
    land = ~water if (~water).any() else np.ones_like(water)
    return dict(n=n, cell_km=size, below_m=below_m, min_aircraft=min_aircraft,
                covered=float((counts >= min_aircraft)[land].mean()), counts=counts.tolist(), water=water.tolist())


def default_parts(day_dir) -> list[Path]:
    return sorted(Path(day_dir).glob("*.tar.a?"))


RELEASES = "https://api.github.com/repos/adsblol/globe_history_{year}/releases/tags/v{tag}-planes-readsb-prod-0"


def download_day(date, day_dir, *, progress=None) -> list[Path]:
    """Fetch adsb.lol's archive for `date` ("YYYY-MM-DD", UTC) into `day_dir` as prod.tar.aa, .ab, ...

    About 4 GB a day, published the following day. Parts already there at full size are kept.
    `progress(done_bytes, total_bytes)` is called as it goes.
    """
    year = date[:4]
    try:
        with urllib.request.urlopen(RELEASES.format(year=year, tag=date.replace("-", ".")), timeout=60) as r:
            assets = sorted((a for a in json.load(r)["assets"] if ".tar.a" in a["name"]), key=lambda a: a["name"])
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        assets = []
    if not assets:
        ready = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1, hours=4)
        if datetime.now(timezone.utc) < ready:
            raise FileNotFoundError(f"adsb.lol hasn't published {date} yet: each day appears around 03:30 UTC the "
                                    f"next day, so try again after {ready:%Y-%m-%d %H:%M} UTC")
        raise FileNotFoundError(f"adsb.lol has no archive for {date}")
    day_dir = Path(day_dir)
    day_dir.mkdir(parents=True, exist_ok=True)
    total = sum(a["size"] for a in assets)
    done = 0
    out = []
    for a in assets:
        path = day_dir / f"prod.tar.{a['name'].rsplit('.', 1)[1]}"
        out.append(path)
        if path.exists() and path.stat().st_size == a["size"]:
            done += a["size"]
            continue
        tmp = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(a["browser_download_url"], timeout=120) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 22):
                f.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
        tmp.replace(path)  # over a part left short by an earlier run (rename refuses that on Windows)
    return out


TYPES_URL = "https://raw.githubusercontent.com/wiedehopf/tar1090-db/master/db/icao_aircraft_types2.js"


def load_type_table(path) -> dict:
    """ICAO type designator -> [description, engine code like 'L2J', wake class], from tar1090-db's
    db/icao_aircraft_types2.js (gzipped JSON despite the name), downloaded to `path` if missing."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(TYPES_URL, timeout=60) as r:
            path.write_bytes(r.read())
    raw = path.read_bytes()
    try:
        raw = gzip.decompress(raw)
    except OSError:
        pass
    return json.loads(raw)


def is_jet(typecode, table) -> bool:
    """True only for types known to be jets: unknown types are kept, as the video did."""
    entry = table.get(typecode or "")
    return bool(entry) and len(entry[1]) == 3 and entry[1][2] == "J"
