"""The search that compares flights first must find what the grid search finds, at any distance."""

from dataclasses import replace

import numpy as np

from doppler_loc import scenarios
from doppler_loc.geo import LocalFrame
from doppler_loc.measure import Event, fit_flyby, straight_flyby, trace_ridge
from doppler_loc.search import observer_grid, search_two_events
from doppler_loc.spectro import compute_spectrogram
from doppler_loc.wide import FitRegion, Legs, QuietIndex, _join, _line_sample, search_anchored_wide, search_pairs_wide, search_two_events_wide


def test_frame_conversions_are_exact_across_a_continent():
    frame = LocalFrame(39.0, -96.0)
    lat = np.array([39.0, 49.0, 25.0, 47.5, 30.0])
    lon = np.array([-96.0, -124.5, -80.0, -68.0, -115.0])
    for h in (0.0, 1500.0):
        got = frame.to_latlon(frame.to_enu(lat, lon, np.full(5, h)))
        assert np.allclose(got[0], lat, atol=1e-7) and np.allclose(got[1], lon, atol=1e-7)  # about a centimetre


def test_a_height_stays_true_far_from_the_frames_centre():
    frame = LocalFrame(39.0, -96.0)
    xyz = frame.to_enu(np.array([47.5, 25.0]), np.array([-68.0, -115.0]), np.array([1500.0, 300.0]))
    assert xyz[0, 2] < -100e3  # what the frame's own z says
    assert np.allclose(frame.height(xyz), [1500.0, 300.0], atol=0.01)


def test_a_ground_point_keeps_its_place_in_the_frame():
    frame = LocalFrame(39.0, -96.0)
    x, y = np.array([0.0, 2300e3, -2400e3, 500e3]), np.array([0.0, 1100e3, -1200e3, -300e3])
    lat, lon = frame.ground_latlon(x, y)
    xyz = frame.to_enu(lat, lon, np.zeros(4))
    assert np.allclose(xyz[:, 0], x, atol=0.01) and np.allclose(xyz[:, 1], y, atol=0.01)


def test_flights_are_matched_only_when_close_at_the_right_gap():
    a = np.array([[0, 1000.0, 0.0, 0.0, 500.0, 60.0], [1, 1000.0, 90e3, 0.0, 500.0, 60.0]])
    b = np.array([[2, 1100.0, 3000.0, 0.0, 500.0, 60.0],  # close, right gap
                  [3, 1100.0, 40e3, 0.0, 500.0, 60.0],  # right gap, too far from either
                  [4, 2000.0, 3000.0, 0.0, 500.0, 60.0],  # close, wrong gap
                  [0, 1100.0, 100.0, 0.0, 500.0, 60.0]])  # the same flight
    ma, mb = _join(a, 2000.0, b, 2000.0, 95.0, 105.0)
    assert ma[:, 0].tolist() == [0] and mb[:, 0].tolist() == [2]


class _Terrain:
    zoom, _tiles = 8, {}

    def prefetch(self, keys):
        pass


class _FlatSite:
    """A site with the scenario's ground: `search_two_events_wide` needs only these."""

    dem = _Terrain()

    def __init__(self, ground):
        self.ground = ground
        self.frame = LocalFrame(40.0, -100.0)

    def place(self, x, y):
        return np.stack([x, y, self.ground(x, y) + 1.5], axis=-1)


def test_same_answer_as_the_grid_search():
    sc = scenarios.two_passes(n_decoys=15)
    spec = compute_spectrogram(sc.audio, sc.sr)
    events = [fit_flyby(*trace_ridge(spec, *m)) for m in sc.marks[:2]]
    half, spacing = 10000.0, 500.0
    grid = observer_grid((0.0, 0.0), half, spacing, sc.ground)
    kw = dict(offset_tol=8.0, shift_tol=6.0, silence_dist=2500.0)
    old = search_two_events(spec, sc.adsb, grid, *events, spec.t[-1], silence_between_only=True, **kw)
    legs = Legs.from_tracks(sc.adsb)
    samples = legs.samples()
    new, *_ = search_two_events_wide(spec, legs, samples, QuietIndex(samples), *events, _FlatSite(sc.ground), half=half, spacing=spacing,
                                     all_rows=True, **kw)
    n = int(round(2 * half / spacing)) + 1
    old_keys = ((old.point % n).astype(np.int64) << 32) + (old.point // n)
    assert len(old.a) > 0
    assert set(zip(old.a.tolist(), old.b.tolist(), old_keys.tolist())) == set(zip(new.a.tolist(), new.b.tolist(), new.point.tolist()))
    a, b = np.lexsort((old_keys, old.b, old.a)), np.lexsort((new.point, new.b, new.a))
    assert np.allclose(old.score1[a], new.score1[b]) and np.allclose(old.score2[a], new.score2[b])


def test_a_flight_between_two_passes_breaks_the_silence_only_if_nothing_was_traced_then():
    sc = scenarios.two_passes(n_decoys=0)
    spec = compute_spectrogram(sc.audio, sc.sr)
    events = [fit_flyby(*trace_ridge(spec, *m)) for m in sc.marks[:2]]
    between = 650.0
    third = scenarios._pass_by("N123AB", sc.truth, heading=250.0, speed=60.0, alt=800.0, side_offset=700.0, t_cpa=sc.t_rec + between)
    tracks = [*sc.adsb, scenarios.adsb_sample(third, rng=np.random.default_rng(1))]
    legs = Legs.from_tracks(tracks)
    samples = legs.samples()
    half, spacing = 8000.0, 500.0

    def true_pair(heard):
        res, *_ = search_two_events_wide(spec, legs, samples, QuietIndex(samples), *events, _FlatSite(sc.ground), half=half, spacing=spacing,
                                         all_rows=True, heard=heard, offset_tol=8.0, shift_tol=6.0, silence_dist=2500.0)
        x, y = (res.point >> 32) * spacing - half, (res.point & 0xFFFFFFFF) * spacing - half
        return int(((res.a == 0) & (res.b == 1) & (np.hypot(x - sc.truth[0], y - sc.truth[1]) < 1500.0)).sum())

    assert true_pair(()) == 0
    assert true_pair((between + 5.0,)) > 0
    assert true_pair((between + 200.0,)) == 0


def test_a_pass_with_wrong_straight_numbers_is_found_by_its_line():
    """A turning plane's fit says nothing about its real speed, distance or closest moment."""
    sc = scenarios.two_passes(n_decoys=15)
    spec = compute_spectrogram(sc.audio, sc.sr)
    straight, other = [fit_flyby(*trace_ridge(spec, *m)) for m in sc.marks[:2]]
    turning = replace(other, speed=other.speed * 0.3, dist=other.dist * 0.1, t_heard=other.t_heard + 40.0, speed_tol=0.6, dist_tol=0.9)
    legs = Legs.from_tracks(sc.adsb)
    samples = legs.samples()
    half, spacing = 10000.0, 500.0
    site = _FlatSite(sc.ground)
    old, *_ = search_two_events_wide(spec, legs, samples, QuietIndex(samples), straight, turning, site, half=half, spacing=spacing, offset_tol=8.0, shift_tol=6.0,
                                     silence_dist=2500.0)
    assert not ((old.a == 0) & (old.b == 1)).any()
    new, followed, xyz = search_anchored_wide(legs, samples, straight, turning, site, half=half, spacing=spacing)
    true_pair = np.flatnonzero((new.a == 0) & (new.b == 1))  # the scenario has decoys fitting as well, elsewhere
    assert true_pair.size == 1
    assert np.hypot(*(xyz[true_pair[0]][:2] - np.asarray(sc.truth)[:2])) < 1000.0
    assert new.score1[true_pair[0]] > 0.8 and new.score2[true_pair[0]] > 0.8


def test_a_known_start_can_search_only_the_pairs_with_a_turning_pass():
    sc = scenarios.two_passes(n_decoys=15)
    spec = compute_spectrogram(sc.audio, sc.sr)
    straight, other = [fit_flyby(*trace_ridge(spec, *m)) for m in sc.marks[:2]]
    turning = replace(other, speed=other.speed * 0.3, dist=other.dist * 0.1, t_heard=other.t_heard + 40.0, speed_tol=0.6, dist_tol=0.9)
    legs = Legs.from_tracks(sc.adsb)
    site = _FlatSite(sc.ground)

    def search(start, marks):
        return search_pairs_wide(spec, legs, [straight, turning], site, half=10000.0, spacing=500.0, f0_tols=[0.15, 0.3], start=start, turning=marks, anchored_only=True)

    found, grid, _ = search((sc.t_rec, 30.0), [False, True])
    assert len(found) == 1 and found[0].anchored and (found[0].i, found[0].j) == (0, 1)
    res = found[0].result
    true_pair = np.flatnonzero((res.a == 0) & (res.b == 1))
    assert true_pair.size == 1 and np.hypot(*(grid[res.point[true_pair[0]]][:2] - np.asarray(sc.truth)[:2])) < 1000.0
    assert search((sc.t_rec, 30.0), [False, False])[0] == []  # nothing for this search to add
    late = search((sc.t_rec + 3000.0, 30.0), [False, True])[0][0].result
    assert not ((late.a == 0) & (late.b == 1)).any()  # the true flights, but not at that time


def test_fit_region_narrows_the_ranges_around_the_true_pass():
    speed, dist, t_heard = 70.0, 1500.0, 100.0
    t = np.arange(40.0, 160.0, 0.5)
    f = straight_flyby(t, 200.0, speed, dist, t_heard)
    ev = Event(t_heard=t_heard, f0=200.0, speed=speed, dist=dist, speed_tol=0.4, dist_tol=0.7, line=(t, f))
    region = FitRegion(ev, _line_sample(ev))
    assert region.allows(np.array([speed]), np.array([dist]))[0]
    assert not region.allows(np.array([speed * 0.62]), np.array([dist * 1.65]))[0]
    assert region.table.mean() < 0.5


def test_matching_a_block_at_a_time_finds_the_same_places():
    from doppler_loc.wide import _cells, _partners, _partners_of
    rng = np.random.default_rng(4)
    n = 4000
    a = np.c_[rng.integers(0, 60, n), rng.uniform(0, 3000, n), rng.uniform(-40e3, 40e3, (n, 2)), rng.uniform(300, 1500, n), rng.uniform(30, 90, n)]
    b = np.c_[rng.integers(0, 60, n), rng.uniform(0, 3000, n), rng.uniform(-40e3, 40e3, (n, 2)), rng.uniform(300, 1500, n), rng.uniform(30, 90, n)]
    args = (3000.0, b, 2500.0, 200.0, 320.0)
    whole = _partners(*_join(a, *args, chunk=1 << 20), 3000.0, 2500.0, 50e3, 500.0)
    blocks = _partners_of(a, 3000.0, b, 2500.0, 200.0, 320.0, 50e3, 500.0)
    assert whole and list(whole) == list(blocks)
    for fa in whole:
        assert [fb for fb, _ in whole[fa]] == [fb for fb, _ in blocks[fa]]
        for (_, r1), (_, r2) in zip(whole[fa], blocks[fa]):
            assert len(r1) == len(r2)
            assert set(zip(*_cells(r1))) == set(zip(*_cells(r2)))

