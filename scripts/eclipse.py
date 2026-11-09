"""Test recordings from Eclipse Soundscapes (zenodo.org, CC BY): AudioMoth recorders at published
coordinates, clocks set by time chime, one-minute WAVs named by their UTC start, around the
8 April 2024 eclipse. adsb.lol has those days, so they can be searched at once.

    uv run python scripts/eclipse.py fetch <record> [--date 2024-04-08 --start 14:00 --hours 3]
    uv run python scripts/eclipse.py projects

`fetch` reads the site's day ZIP (about 6 GB) with HTTP range requests: only the members (WAVs)
that overlap the window, and of each only as far as the window's end, in pieces over 8 connections
(Zenodo gives about 3 MB/s to one). Some recorders wrote one-minute files, others one file until it
reached 4 GB (12 h); compressed members can't be entered midway, so those are read from their start.
Each is laid at its own time (gaps stay silent) in one 8 kHz WAV in data/eclipse/<ESID>/, with a
.json of the site and start. `projects` makes a project of each,
centred 6-8 km from the recorder in a seeded random direction, start
known to +-120 s (a recorder clock can drift a minute or more: 75 s at ESID 281).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timedelta, timezone
import io
import json
from math import gcd
from pathlib import Path
import time
import urllib.request
import zipfile
import zlib

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from doppler_loc.geo import LocalFrame
from doppler_loc.project import DATA, Project, Settings

FOLDER = DATA / "eclipse"
PROJECTS = DATA / "projects"
SR = 8000
ZENODO = "https://zenodo.org/records/{record}/files/{name}?download=1"


class RangeFile(io.RawIOBase):
    """A remote file read with HTTP range requests, for zipfile to find and read single members."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Range"].rsplit("/", 1)[1])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                break
            except OSError:
                if attempt == 4:
                    raise
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def site_info(record) -> dict:
    with urllib.request.urlopen(ZENODO.format(record=record, name="total_eclipse_data.csv"), timeout=60) as r:
        d = next(csv.DictReader(io.StringIO(r.read().decode("utf-8-sig"))))
    return dict(esid=int(d["ESID"]), record=str(record), lat=float(d["Latitude"]), lon=float(d["Longitude"]),
                clock=d["WAV Files Time & Date Settings"], clock_notes=d["Data Collector Start Time Notes"],
                source=f"https://zenodo.org/records/{record}")


def get_range(url, a, b, tries=6) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b - 1}"})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                data = r.read()
            if len(data) == b - a:
                return data
        except OSError:
            if attempt == tries - 1:
                raise
        time.sleep(2 * (attempt + 1))
    raise OSError(f"short read of bytes {a}-{b}")


def member_stream(path, compress_type):
    """The uncompressed bytes of a member saved (compressed) at `path`, in pieces."""
    dec = zlib.decompressobj(-15) if compress_type == zipfile.ZIP_DEFLATED else None
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            yield dec.decompress(chunk) if dec else chunk


def wav_format(head: bytes):
    """(sample rate, bytes a second, where the samples start) from a WAV's first bytes."""
    i = 12
    while i + 8 <= len(head):
        cid, size = head[i:i + 4], int.from_bytes(head[i + 4:i + 8], "little")
        if cid == b"fmt ":
            sr, bps = int.from_bytes(head[i + 12:i + 16], "little"), int.from_bytes(head[i + 16:i + 20], "little")
        if cid == b"data":
            return sr, bps, i + 8
        i += 8 + size + (size & 1)
    raise ValueError("no data chunk in the WAV header")


def wav_samples(chunks):
    """(sample rate, generator of int16 sample blocks) from a stream of a 16-bit mono WAV's bytes."""
    buf = b""
    for c in chunks:
        buf += c
        if len(buf) >= 1 << 16:
            break
    i, sr = 12, None
    while True:
        cid, size = buf[i:i + 4], int.from_bytes(buf[i + 4:i + 8], "little")
        if cid == b"fmt ":
            ch, sr, bits = (int.from_bytes(buf[i + 10:i + 12], "little"), int.from_bytes(buf[i + 12:i + 16], "little"),
                            int.from_bytes(buf[i + 22:i + 24], "little"))
            assert ch == 1 and bits == 16, (ch, bits)
        if cid == b"data":
            break
        i += 8 + size + (size & 1)
    rest = buf[i + 8:]

    def blocks():
        pending = rest
        for c in chunks:
            pending += c
            n = len(pending) // 2 * 2
            yield np.frombuffer(pending[:n], dtype="<i2")
            pending = pending[n:]
        if len(pending) >= 2:
            yield np.frombuffer(pending[:len(pending) // 2 * 2], dtype="<i2")
    return sr, blocks()


def resample_long(x, sr, piece_s=600, pad_s=2):
    """`x` (int16 at `sr`) at SR Hz, in pieces of `piece_s` with `pad_s` either side, trimmed, so the
    filter's edges leave no seams and hours of audio need no more than a piece's memory at once."""
    g = gcd(SR, sr)
    up, down = SR // g, sr // g
    piece, pad = int(piece_s * sr) // down * down, int(pad_s * sr) // down * down  # whole output samples
    out = []
    for s in range(0, len(x), piece):
        a, b = max(0, s - pad), min(len(x), s + piece + pad)
        y = resample_poly(x[a:b].astype(np.float32), up, down)
        y0 = (s - a) * up // down
        out.append(y[y0:y0 + (min(s + piece, len(x)) - s) * up // down])
    return np.concatenate(out)


def cmd_fetch(args):
    site = site_info(args.record)
    t0 = datetime.strptime(f"{args.date} {args.start}", "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    t1 = t0 + timedelta(hours=args.hours)
    out_dir = FOLDER / str(site["esid"])
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"esid{site['esid']}_{t0:%Y-%m-%dT%H%MZ}"
    if (out_dir / f"{stem}.json").exists():
        print(f"{stem}: already fetched")
        return
    day = f"ESID_{site['esid']}_{t0:%Y_%m_%d}.zip"
    url = ZENODO.format(record=args.record, name=day)
    rf = RangeFile(url)
    zf = zipfile.ZipFile(io.BufferedReader(rf, 1 << 20))
    members = []
    for info in zf.infolist():
        name = Path(info.filename).name
        try:
            t = datetime.strptime(name[:15], "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if name.lower().endswith(".wav") and info.file_size > 1000:
            members.append((t, info))
    members.sort(key=lambda x: x[0])
    # Each member's length, from its WAV header: its compressed data starts after its local header
    # (30 bytes, name, extra), and the first 256 kB of it hold the header.
    wanted = []
    for t, info in members:
        if t >= t1:
            continue
        h = get_range(url, info.header_offset, info.header_offset + 30)
        start = info.header_offset + 30 + int.from_bytes(h[26:28], "little") + int.from_bytes(h[28:30], "little")
        head = get_range(url, start, start + min(info.compress_size, 1 << 18))
        sr, bytes_per_s, data_at = wav_format(zlib.decompressobj(-15).decompress(head)
                                              if info.compress_type == zipfile.ZIP_DEFLATED else head)
        end = t + timedelta(seconds=(info.file_size - data_at) / bytes_per_s)
        if end > t0:
            # Of a long member, only as far as the window's end: by time, with room for compression
            # varying over the day; more is fetched below if that falls short.
            frac = 1.0 if end <= t1 else min(1.0, (t1 - t) / (end - t) * 1.15 + 0.02)
            wanted.append((t, info, start, frac))
    total = sum(int(i.compress_size * f) for _, i, _, f in wanted)
    print(f"ESID {site['esid']}: {len(wanted)} file(s) overlap {t0:%H:%M}-{t1:%H:%M} UTC; "
          f"downloading {total / 1e9:.2f} GB", flush=True)
    done = [0]

    def fetch(part, start, a_from, n):
        """Bytes a_from..n of a member's compressed data into `part`, in 50 MB pieces over 8 connections."""
        with open(part, "ab") as f:
            f.truncate(n)

        def piece(a):
            data = get_range(url, start + a, start + min(a + 50_000_000, n))
            with open(part, "r+b") as f:
                f.seek(a)
                f.write(data)
            done[0] += len(data)
            print(f"  {done[0] / 1e9:.2f} GB", flush=True)

        with ThreadPoolExecutor(8) as ex:
            list(ex.map(piece, range(a_from, n, 50_000_000)))

    audio = np.zeros(int(args.hours * 3600 * SR), dtype=np.int16)
    covered = 0.0
    for k, (t, info, start, frac) in enumerate(wanted):
        part = out_dir / f"{stem}.{k}.part"
        part.unlink(missing_ok=True)
        n = max(1, int(info.compress_size * frac))
        fetch(part, start, 0, n)
        while True:
            stream = member_stream(part, info.compress_type)
            sr, blocks = wav_samples(stream)
            keep_a = max(0, int(round((t0 - t).total_seconds() * sr)))  # this member's samples in the window
            keep_b = int(round((t1 - t).total_seconds() * sr))
            kept, pos = [], 0
            try:
                for blk in blocks:
                    a, b = pos, pos + len(blk)
                    pos = b
                    lo, hi = max(a, keep_a), min(b, keep_b)
                    if hi > lo:
                        kept.append(blk[lo - a:hi - a])
                    if pos >= keep_b:
                        break
            except zlib.error:
                pass  # a member read only partly ends mid-stream
            finally:
                blocks.close()
                stream.close()  # closes the file, so it can be extended or deleted (Windows)
            print(f"  {Path(info.filename).name}: read to {(t + timedelta(seconds=pos / sr)):%H:%M:%S} "
                  f"({n / 1e9:.2f} of {info.compress_size / 1e9:.2f} GB)", flush=True)
            if pos >= keep_b or n >= info.compress_size:
                break
            fetch(part, start, n, info.compress_size)  # fell short: the rest of it
            n = info.compress_size
        part.unlink()
        if not kept:
            continue
        y = resample_long(np.concatenate(kept), sr)
        i = int(round(((t - t0).total_seconds() + keep_a / sr) * SR))  # where its first kept sample goes
        x0, x1 = max(i, 0), min(i + len(y), len(audio))
        if x1 > x0:
            audio[x0:x1] = np.clip(y[x0 - i:x1 - i], -32768, 32767).astype(np.int16)
            covered += (x1 - x0) / SR
    wavfile.write(out_dir / f"{stem}.wav", SR, audio)
    meta = dict(site=site, audio=f"{stem}.wav", start_unix=t0.timestamp(), duration_s=args.hours * 3600,
                covered_s=round(covered), files=len(wanted), day_zip=day)
    (out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=1))
    print(f"{stem}: {covered / 3600:.2f} h of audio in {args.hours} h")


def cmd_projects(args):
    for meta_path in sorted(FOLDER.glob("*/*.json")):
        meta = json.loads(meta_path.read_text())
        if meta.get("project"):
            continue
        s, t0 = meta["site"], meta["start_unix"]
        name = f"eclipse-{s['esid']}-{datetime.fromtimestamp(t0, timezone.utc):%Y-%m-%d-%H%M}"
        rng = np.random.default_rng(zlib.crc32(name.encode()))  # the same offset every time
        angle, dist = rng.uniform(0, 2 * np.pi), rng.uniform(6000, 8000)
        lat, lon = LocalFrame(s["lat"], s["lon"]).to_latlon(np.array([dist * np.sin(angle), dist * np.cos(angle), 0.0]))
        Project.create(Settings(name=name, audio=str((meta_path.parent / meta["audio"]).resolve()),
                                date=datetime.fromtimestamp(t0, timezone.utc).strftime("%Y-%m-%d"),
                                lat=round(float(lat), 5), lon=round(float(lon), 5), truth_lat=s["lat"],
                                truth_lon=s["lon"], start_unix=t0, start_tol_s=120.0), PROJECTS)
        meta["project"] = name
        meta_path.write_text(json.dumps(meta, indent=1))
        print(f"{name}: created")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fetch")
    p.add_argument("record", help="the site's Zenodo record number")
    p.add_argument("--date", default="2024-04-08")
    p.add_argument("--start", default="14:00", help="UTC")
    p.add_argument("--hours", type=float, default=3.0)
    sub.add_parser("projects")
    args = ap.parse_args()
    {"fetch": cmd_fetch, "projects": cmd_projects}[args.cmd](args)


if __name__ == "__main__":
    main()
